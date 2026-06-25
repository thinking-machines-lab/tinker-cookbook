"""Score the production agent app on the held-out RILL families.

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
from collections import defaultdict

import httpx

from tinker_cookbook.recipes.rill_rl.training.grading import shaped_reward
from tinker_cookbook.recipes.rill_rl.training.tasks import build_tasks
from tinker_cookbook.utils import logtree


def log_rollouts_html(path: str, model: str, rollouts: list[dict]) -> None:
    """Write a readable logtree HTML report of every rollout (prompt, each turn's program,
    interpreter output/error, fix requests, and the final reward)."""
    with logtree.init_trace(f"RILL rollouts: {model}", path=path):
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
    ap = argparse.ArgumentParser(description="Score the RILL agent app on held-out tasks.")
    ap.add_argument("--app-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
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
    _, eval_tasks = build_tasks(seed=args.seed)
    if args.limit is not None:
        eval_tasks = eval_tasks[: args.limit]

    sem = asyncio.Semaphore(args.concurrency)
    by_family: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    rollouts: list[dict] = []
    done = 0

    async with httpx.AsyncClient(timeout=600.0) as http:

        async def one(task):
            nonlocal done
            body = {"prompt": task.prompt, "model": args.model, "max_turns": args.max_turns}
            if args.openai_base_url:
                body["openai_base_url"] = args.openai_base_url
            if args.openai_api_key:
                body["openai_api_key"] = args.openai_api_key
            if args.temperature is not None:
                body["temperature"] = args.temperature
            if args.max_completion_tokens is not None:
                body["max_completion_tokens"] = args.max_completion_tokens
            program, transcript = "", []
            async with sem:
                try:
                    resp = await http.post(f"{args.app_url}/solve", json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    program = data.get("program", "")
                    transcript = data.get("transcript", [])
                except Exception as e:
                    print(f"  rollout error on {task.name}: {e!r}")
            reward, info = shaped_reward(program, task)
            by_family[task.family].append((bool(info["correct"]), reward))
            rollouts.append(
                {
                    "family": task.family,
                    "name": task.name,
                    "prompt": task.prompt,
                    "expect": task.expect,
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

    print(f"\n=== RILL eval: {args.model} via {args.app_url} ===")
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
