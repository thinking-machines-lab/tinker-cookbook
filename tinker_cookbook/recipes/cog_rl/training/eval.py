"""Score the production agent app on the held-out Cog families.

Drives the same ``/solve`` endpoint the trainer uses, then grades each rollout's final
program against the held-out expected output (the app never sees it). Point it at the app
configured for any backend:

- **Frontier baseline:** run the app with ``OPENAI_API_KEY`` set (real OpenAI), then
  ``python -m ...training.eval --app-url http://127.0.0.1:8000 --model gpt-5.5``.
- **Trained checkpoint:** serve the checkpoint behind an OpenAI-compatible endpoint (or
  the training proxy) and pass ``--openai-base-url`` so each /solve routes there.

The app stays reward-agnostic; grading is entirely on this side.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
from collections import defaultdict

import httpx

from tinker_cookbook.recipes.cog_rl.training.grading import shaped_reward
from tinker_cookbook.recipes.cog_rl.training.tasks import CogTask, get_tasks
from tinker_cookbook.utils import logtree


def with_example(task: CogTask) -> CogTask:
    """The task with its first hidden test shown in the prompt as a worked example.

    Showing one concrete input/output disambiguates the task semantics (the main failure
    mode on natural-language corpus prompts) and gives the agent something to self-verify
    against. Grading must then exclude that test — see ``split_visible_hidden``.
    """
    a, e = task.tests[0]
    return dataclasses.replace(
        task, prompt=task.prompt + f" For example, solve({a}) should output {e}."
    )


def split_visible_hidden(task: CogTask) -> tuple[CogTask, CogTask]:
    """(visible-only, hidden-only) copies of ``task``, split at the first test.

    The visible task is the self-verification target (the one example the model saw); the
    hidden task is the honest grading set. Requires ``len(task.tests) >= 2``.
    """
    return (
        dataclasses.replace(task, tests=(task.tests[0],)),
        dataclasses.replace(task, tests=tuple(task.tests[1:])),
    )


def log_rollouts_html(path: str, model: str, rollouts: list[dict]) -> None:
    """Write a readable logtree HTML report of every rollout (prompt, each turn's program,
    interpreter output/error, fix requests, and the final reward)."""
    with logtree.init_trace(f"Cog rollouts: {model}", path=path):
        n = len(rollouts)
        passed = sum(r["correct"] for r in rollouts)
        logtree.table_from_dict(
            {
                "model": model,
                "rollouts": n,
                "pass@1": f"{passed / max(n, 1):.3f}",
                "mean_reward": f"{sum(r['reward'] for r in rollouts) / max(n, 1):.3f}",
            },
            caption="summary",
        )
        for r in rollouts:
            verdict = "PASS" if r["correct"] else "fail"
            with logtree.scope_header(
                f"[{r['family']}] {r['name']} — {verdict} (reward {r['reward']:.2f})"
            ):
                logtree.table_from_dict(
                    {"prompt": r["prompt"], "expected": r["expect"], "final program": r["program"]},
                    caption="task",
                )
                for ev in r.get("transcript", []):
                    t = ev.get("type")
                    if t == "assistant":
                        logtree.details(
                            ev["content"], summary=f"turn {ev['turn']}: model message", pre=True
                        )
                        logtree.details(
                            ev["program"], summary=f"turn {ev['turn']}: extracted program", pre=True
                        )
                    elif t == "run":
                        label = "ran clean" if ev["ran_clean"] else (ev.get("error") or "error")
                        logtree.details(
                            ev.get("output") or ev.get("error") or "(no output)",
                            summary=f"turn {ev['turn']}: interpreter — {label}",
                            pre=True,
                        )
                    elif t == "fix_request":
                        logtree.details(
                            ev["content"], summary=f"turn {ev['turn']}: fix request", pre=True
                        )
                    elif t == "api_error":
                        logtree.log_text(f"API error: {ev['message']}")
    print(f"Wrote rollout log to {path}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Score the Cog agent app on held-out tasks.")
    ap.add_argument("--app-url", default="http://127.0.0.1:8000")
    ap.add_argument("--task-source", default="families", help="families | corpus | both")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--repeat", type=int, default=1, help="sample each task this many times (stabilizes pass@1)"
    )
    ap.add_argument(
        "--show-example",
        action="store_true",
        help="show the first test as a worked example in the prompt; grade ONLY on the "
        "remaining hidden tests (a different, easier protocol — don't compare with default runs)",
    )
    ap.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="with --show-example: retry /solve up to N times, keeping the first program "
        "that passes the visible example (self-verification; hidden tests never leak)",
    )
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--openai-base-url", default=None, help="per-request backend override")
    ap.add_argument("--openai-api-key", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    # Anthropic's OpenAI-compatible endpoint requires a token limit; set this for it.
    ap.add_argument("--max-completion-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-html", default=None, help="write a logtree HTML report of all rollouts")
    return ap.parse_args()


async def _main(args: argparse.Namespace) -> None:
    if args.best_of > 1 and not args.show_example:
        raise SystemExit("--best-of needs --show-example (nothing legitimate to verify against)")
    _, eval_tasks = get_tasks(args.task_source, seed=args.seed)
    if args.show_example:
        n0 = len(eval_tasks)
        eval_tasks = [t for t in eval_tasks if len(t.tests) >= 2]
        if len(eval_tasks) < n0:
            print(f"--show-example: dropped {n0 - len(eval_tasks)} tasks with <2 tests")
    if args.limit is not None:
        eval_tasks = eval_tasks[: args.limit]
    if args.repeat > 1:
        eval_tasks = [t for t in eval_tasks for _ in range(args.repeat)]

    sem = asyncio.Semaphore(args.concurrency)
    by_family: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    rollouts: list[dict] = []
    done = 0

    async with httpx.AsyncClient(timeout=600.0) as http:

        async def solve_once(prompt: str, name: str) -> tuple[str, list]:
            body = {"prompt": prompt, "model": args.model, "max_turns": args.max_turns}
            if args.openai_base_url:
                body["openai_base_url"] = args.openai_base_url
            if args.openai_api_key:
                body["openai_api_key"] = args.openai_api_key
            if args.temperature is not None:
                body["temperature"] = args.temperature
            if args.max_completion_tokens is not None:
                body["max_completion_tokens"] = args.max_completion_tokens
            try:
                resp = await http.post(f"{args.app_url}/solve", json=body)
                resp.raise_for_status()
                data = resp.json()
                return data.get("program", ""), data.get("transcript", [])
            except Exception as e:
                print(f"  rollout error on {name}: {e!r}")
                return "", []

        async def one(task):
            nonlocal done
            if args.show_example:
                visible, grade_task = split_visible_hidden(task)
                prompt = with_example(task).prompt
            else:
                visible, grade_task = None, task
                prompt = task.prompt
            async with sem:
                program, transcript = await solve_once(prompt, task.name)
                # Best-of-n with self-verification: retry until the program passes the one
                # example the model was shown. Hidden tests are never consulted here.
                for _ in range(args.best_of - 1):
                    if program and shaped_reward(program, visible)[1]["correct"]:
                        break
                    cand, cand_tr = await solve_once(prompt, task.name)
                    if cand:
                        program, transcript = cand, cand_tr
            reward, info = shaped_reward(program, grade_task)
            by_family[task.family].append((bool(info["correct"]), reward))
            rollouts.append(
                {
                    "family": task.family,
                    "name": task.name,
                    "prompt": prompt,
                    "expect": "; ".join(f"solve({a})={e}" for a, e in grade_task.tests),
                    "program": program,
                    "correct": bool(info["correct"]),
                    "reward": reward,
                    "transcript": transcript,
                }
            )
            done += 1
            if done % 10 == 0 or done == len(eval_tasks):
                print(f"  ... {done}/{len(eval_tasks)}")

        await asyncio.gather(*(one(t) for t in eval_tasks))

    if args.log_html:
        rollouts.sort(key=lambda r: (r["family"], r["name"]))
        log_rollouts_html(args.log_html, args.model, rollouts)

    protocol = ""
    if args.show_example:
        protocol = f" [show-example, best-of-{args.best_of}, graded on hidden tests only]"
    print(f"\n=== Cog eval: {args.model} via {args.app_url}{protocol} ===")
    all_rows = [row for rows in by_family.values() for row in rows]
    n = len(all_rows)
    overall_pass = sum(c for c, _ in all_rows) / max(n, 1)
    overall_reward = sum(r for _, r in all_rows) / max(n, 1)
    print(f"overall (n={n}): pass@1={overall_pass:.3f}  mean_reward={overall_reward:.3f}")
    print("by family:")
    for fam, rows in sorted(by_family.items()):
        k = len(rows)
        p = sum(c for c, _ in rows) / k
        mr = sum(r for _, r in rows) / k
        print(f"  {fam:16s} n={k:<4d} pass@1={p:.3f}  mean_reward={mr:.3f}")


def main() -> None:
    asyncio.run(_main(_parse_args()))


if __name__ == "__main__":
    main()
