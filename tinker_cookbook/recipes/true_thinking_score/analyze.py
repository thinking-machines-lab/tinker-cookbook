"""
TTS analysis CLI — compute True-Thinking Scores on math datasets.

Runs TTS computation on MATH-500 or GSM8K problems, with concurrent
problem processing for throughput.  Results are saved as JSONL for
per-problem data and a JSON summary for aggregate statistics.

Usage:
    # Quick test (5 problems)
    python -m tinker_cookbook.recipes.true_thinking_score.analyze \
        n_problems=5

    # 50 problems from MATH-500 (recommended)
    python -m tinker_cookbook.recipes.true_thinking_score.analyze \
        dataset=math n_problems=50

    # GSM8K, larger model
    python -m tinker_cookbook.recipes.true_thinking_score.analyze \
        dataset=gsm8k model_name=Qwen/Qwen3.5-27B n_problems=50

    # Full MATH-500
    python -m tinker_cookbook.recipes.true_thinking_score.analyze \
        dataset=math n_problems=500
"""

import asyncio
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import chz
import tinker
from datasets import Dataset, load_dataset

from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed
from tinker_cookbook.recipes.true_thinking_score.tts import (
    generate_cot_and_compute_tts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_problems(dataset: str, n_problems: int, seed: int) -> list[dict[str, str]]:
    """Load math problems with known answers."""
    if dataset == "math":
        ds = cast(Dataset, load_dataset("HuggingFaceH4/MATH-500", name="default", split="test"))
        problems = []
        for row in ds:
            row = cast(dict[str, str], row)
            try:
                answer = extract_boxed(row["solution"])
                problems.append({"question": row["problem"], "answer": answer})
            except ValueError:
                continue

    elif dataset == "gsm8k":
        ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split="test"))
        problems = []
        for row in ds:
            row = cast(dict[str, str], row)
            try:
                answer = extract_gsm8k_final_answer(row["answer"])
                problems.append({"question": row["question"], "answer": answer})
            except ValueError:
                continue

    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'math' or 'gsm8k'.")

    random.Random(seed).shuffle(problems)
    return problems[:n_problems]


@chz.chz
class CLIConfig:
    """Configuration for TTS analysis."""

    # Model
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str | None = None  # Override renderer (e.g. deepseekv3_thinking)

    # Dataset
    dataset: Literal["math", "gsm8k"] = "math"
    n_problems: int = 50

    # Generation
    max_tokens: int = 4096

    # Parallelism: how many problems to process concurrently
    concurrency: int = 64

    # Output
    log_path: str | None = None

    # Reproducibility
    seed: int = 42


async def analyze_one_problem(
    service_client: tinker.ServiceClient,
    model_name: str,
    problem: dict[str, str],
    problem_idx: int,
    total: int,
    max_tokens: int,
    seed: int,
    renderer_name: str | None = None,
) -> dict | None:
    """Analyze a single problem and return its summary dict."""
    logger.info(f"[{problem_idx + 1}/{total}] {problem['question'][:60]}...")
    try:
        result = await generate_cot_and_compute_tts(
            service_client=service_client,
            model_name=model_name,
            question=problem["question"],
            answer_str=problem["answer"],
            max_tokens=max_tokens,
            seed=seed,
            renderer_name=renderer_name,
        )
        summary = result.summary()
        summary["problem_idx"] = problem_idx
        logger.info(
            f"[{problem_idx + 1}/{total}] Done: {summary['n_steps']} steps, "
            f"mean_tts={summary['mean_tts']:.4f}, correct={summary['model_correct']}"
        )
        return summary
    except Exception as e:
        logger.error(f"[{problem_idx + 1}/{total}] Failed: {e}")
        return {
            "question": problem["question"][:100],
            "answer": problem["answer"],
            "error": str(e),
            "problem_idx": problem_idx,
        }


async def cli_main(config: CLIConfig) -> None:
    # Build log path
    if config.log_path is not None:
        log_dir = Path(config.log_path)
    else:
        model_slug = config.model_name.replace("/", "-")
        run_name = (
            f"tts-{config.dataset}-{model_slug}-"
            f"{config.n_problems}problems-"
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_dir = Path(f"/tmp/tinker-examples/tts/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== TTS Analysis ===")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Problems: {config.n_problems}")
    logger.info(f"Concurrency: {config.concurrency}")
    logger.info(f"Log dir: {log_dir}")

    # Load problems
    problems = _load_problems(config.dataset, config.n_problems, config.seed)
    logger.info(f"Loaded {len(problems)} problems")

    service_client = tinker.ServiceClient()

    # Process problems concurrently with a semaphore to limit parallelism
    sem = asyncio.Semaphore(config.concurrency)

    async def _bounded(problem: dict, idx: int) -> dict | None:
        async with sem:
            return await analyze_one_problem(
                service_client,
                config.model_name,
                problem,
                idx,
                len(problems),
                config.max_tokens,
                config.seed,
                config.renderer_name,
            )

    tasks = [_bounded(p, i) for i, p in enumerate(problems)]
    results = await asyncio.gather(*tasks)

    # Filter out None results and collect stats
    valid_results = [r for r in results if r is not None and "error" not in r]
    error_results = [r for r in results if r is not None and "error" in r]

    # Save per-problem JSONL
    jsonl_path = log_dir / "tts_per_problem.jsonl"
    with open(jsonl_path, "w") as f:
        for r in results:
            if r is not None:
                f.write(json.dumps(r) + "\n")

    # Compute aggregate stats
    all_tts: list[float] = []
    n_sv_total = 0
    n_sv_decorative = 0
    n_correct = 0
    for r in valid_results:
        all_tts.extend(r["per_step_tts"])
        n_sv_total += r["n_self_verification"]
        n_sv_decorative += r["n_sv_decorative"]
        if r["model_correct"]:
            n_correct += 1

    aggregate = {
        "model": config.model_name,
        "dataset": config.dataset,
        "n_problems": len(problems),
        "n_problems_succeeded": len(valid_results),
        "n_problems_failed": len(error_results),
        "n_problems_correct": n_correct,
        "accuracy": n_correct / len(valid_results) if valid_results else 0,
        "n_steps_total": len(all_tts),
        "mean_steps_per_problem": len(all_tts) / len(valid_results) if valid_results else 0,
        "mean_tts": sum(all_tts) / len(all_tts) if all_tts else 0,
        "frac_high_tts_0.7": sum(1 for t in all_tts if t >= 0.7) / len(all_tts) if all_tts else 0,
        "frac_high_tts_0.3": sum(1 for t in all_tts if t >= 0.3) / len(all_tts) if all_tts else 0,
        "frac_decorative_0.005": (
            sum(1 for t in all_tts if t <= 0.005) / len(all_tts) if all_tts else 0
        ),
        "n_self_verification": n_sv_total,
        "n_sv_decorative": n_sv_decorative,
        "frac_sv_decorative": n_sv_decorative / n_sv_total if n_sv_total else 0,
    }

    # Save aggregate summary
    summary_path = log_dir / "tts_summary.json"
    with open(summary_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    # Print results
    logger.info(f"\n{'=' * 60}")
    logger.info("=== RESULTS ===")
    logger.info(
        f"Problems: {aggregate['n_problems_succeeded']}/{aggregate['n_problems']} succeeded"
    )
    logger.info(f"Model accuracy: {aggregate['accuracy']:.1%}")
    logger.info(f"Total steps: {aggregate['n_steps_total']}")
    logger.info(f"Mean steps/problem: {aggregate['mean_steps_per_problem']:.1f}")
    logger.info(f"Mean TTS: {aggregate['mean_tts']:.4f}")
    logger.info(f"High TTS (>=0.7): {aggregate['frac_high_tts_0.7']:.1%}")
    logger.info(f"High TTS (>=0.3): {aggregate['frac_high_tts_0.3']:.1%}")
    logger.info(f"Decorative (<=0.005): {aggregate['frac_decorative_0.005']:.1%}")
    logger.info(f"Self-verification steps: {n_sv_total}")
    logger.info(f"  of which decorative: {n_sv_decorative} ({aggregate['frac_sv_decorative']:.1%})")

    logger.info("\nComparison with paper (AIME, DeepSeek-R1-Distill-Qwen-7B):")
    logger.info(f"  Paper mean TTS:       ~0.03    | Ours: {aggregate['mean_tts']:.4f}")
    logger.info(f"  Paper high TTS>=0.7:   2.3%    | Ours: {aggregate['frac_high_tts_0.7']:.1%}")
    logger.info(f"  Paper high TTS>=0.3:   6.4%    | Ours: {aggregate['frac_high_tts_0.3']:.1%}")
    logger.info(f"  Paper SV decorative:  12-21%   | Ours: {aggregate['frac_sv_decorative']:.1%}")

    logger.info("\nResults saved to:")
    logger.info(f"  Per-problem: {jsonl_path}")
    logger.info(f"  Summary:     {summary_path}")


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig, argv=sys.argv[1:])
    asyncio.run(cli_main(config))
