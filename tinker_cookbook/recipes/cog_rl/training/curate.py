"""Harvest verified gold Cog solutions for SFT warm-start + feasibility filter.

This is **self-distillation** (rejection-sampling fine-tuning / STaR / expert iteration): the
*open model we are training* generates the candidate solutions, and the interpreter + hidden
tests decide which to keep. No external/frontier model is involved — the verifier is what makes
a kept solution trustworthy, not the source.

Runs the agent app (pointed at the open policy via the sampling proxy) over corpus tasks ``n``
times each, grades every attempt against the task's hidden tests with the *same* grader training
uses, and keeps the first program that passes. A task with >= 1 passing attempt is "Cog-feasible";
its passing program is a gold solution. This both filters the corpus to expressible problems and
yields a supervised dataset, with no hand-written Cog. Each RL round raises the solve rate, which
yields more gold, which warm-starts the next round (the expert-iteration loop).

    # serve the current best checkpoint, then harvest from it
    python -m tinker_cookbook.recipes.cog_rl.training.curate \\
        --app-url http://127.0.0.1:8250 --model exp7 --attempts 6 \\
        --out /tmp/dylan/cog_gold_self.jsonl

(``--model`` is just the label the app forwards; behind the proxy it samples the served policy.)

Output JSONL rows: ``{"name", "family", "prompt", "program"}`` (one per solved task).
"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx

from tinker_cookbook.recipes.cog_rl.training.grading import shaped_reward
from tinker_cookbook.recipes.cog_rl.training.tasks import get_tasks


async def _curate(args: argparse.Namespace) -> None:
    train, _ = get_tasks(args.task_source, seed=args.seed)
    if args.limit is not None:
        train = train[: args.limit]
    sem = asyncio.Semaphore(args.concurrency)
    gold: list[dict] = []
    solved = 0
    done = 0

    async with httpx.AsyncClient(timeout=900.0) as http:

        async def one(task):
            nonlocal solved, done
            best = None
            best_response = ""
            for _ in range(args.attempts):
                body = {"prompt": task.prompt, "model": args.model, "max_turns": args.max_turns}
                if args.max_completion_tokens is not None:
                    body["max_completion_tokens"] = args.max_completion_tokens
                async with sem:
                    try:
                        resp = await http.post(f"{args.app_url}/solve", json=body)
                        resp.raise_for_status()
                        data = resp.json()
                        program = data.get("program", "")
                        transcript = data.get("transcript", [])
                    except Exception:
                        program, transcript = "", []
                if program:
                    _, info = shaped_reward(program, task)
                    if info["correct"]:
                        best = program
                        # full final assistant message (reasoning + code) for CoT distillation
                        asst = [e for e in transcript if e.get("type") == "assistant"]
                        best_response = asst[-1].get("content", "") if asst else ""
                        break
            done += 1
            if best is not None:
                solved += 1
                row = {
                    "name": task.name,
                    "family": task.family,
                    "prompt": task.prompt,
                    "program": best,
                }
                if args.keep_response and best_response:
                    row["response"] = best_response
                gold.append(row)
            if done % 25 == 0 or done == len(train):
                print(f"  ... {done}/{len(train)} ({solved} solved)")

        await asyncio.gather(*(one(t) for t in train))

    gold.sort(key=lambda g: g["name"])
    with open(args.out, "w") as f:
        for g in gold:
            f.write(json.dumps(g) + "\n")
    print(
        f"\nfeasible/solved: {solved}/{len(train)} "
        f"({solved / max(len(train), 1):.1%}); gold written to {args.out}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest verified gold Cog solutions.")
    ap.add_argument("--app-url", default="http://127.0.0.1:8260")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--task-source", default="corpus")
    ap.add_argument("--attempts", type=int, default=3, help="samples per task; keep first pass")
    ap.add_argument(
        "--keep-response",
        action="store_true",
        help="also store the full final assistant message (reasoning + code) for CoT distillation",
    )
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--max-completion-tokens", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    asyncio.run(_curate(ap.parse_args()))


if __name__ == "__main__":
    main()
