"""
Small-scale TTS experiment.

Runs TTS computation on a handful of math problems to validate the implementation
and check if we see the paper's key finding: most reasoning steps are decorative.

Usage:
    python -m tinker_cookbook.recipes.true_thinking_score.run_small_experiment

Expected behavior:
    - Generates CoT for each problem
    - Computes TTS for each step
    - Reports distribution of TTS scores
    - Checks if most steps are decorative (TTS < 0.005)
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import tinker

from tinker_cookbook.recipes.true_thinking_score.tts import (
    TTSResult,
    generate_cot_and_compute_tts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Small set of math problems with known answers for initial validation.
# Mix of difficulty levels to see if TTS varies.
PROBLEMS = [
    {
        "question": "What is the sum of the first 10 positive integers?",
        "answer": "55",
    },
    {
        "question": "If a rectangle has length 12 and width 5, what is its area?",
        "answer": "60",
    },
    {
        "question": (
            "A store sells apples for $2 each and oranges for $3 each. "
            "If John buys 4 apples and 3 oranges, how much does he spend in total?"
        ),
        "answer": "17",
    },
    {
        "question": (
            "Let f(x) = x^2 + 3x + 2. What is f(5)?"
        ),
        "answer": "42",
    },
    {
        "question": (
            "In how many ways can 5 people be seated in a row?"
        ),
        "answer": "120",
    },
]

# Use a small model for faster iteration
MODEL_NAME = "Qwen/Qwen3-4B"


async def main():
    log_dir = Path("/tmp/tinker-tts-experiment")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== TTS Small-Scale Experiment ===")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Problems: {len(PROBLEMS)}")
    logger.info(f"Log dir: {log_dir}")

    service_client = tinker.ServiceClient()

    results: list[dict] = []
    all_tts_scores: list[float] = []

    for i, problem in enumerate(PROBLEMS):
        logger.info(f"\n{'='*60}")
        logger.info(f"Problem {i+1}/{len(PROBLEMS)}: {problem['question'][:60]}...")
        logger.info(f"Expected answer: {problem['answer']}")

        try:
            tts_result = await generate_cot_and_compute_tts(
                service_client=service_client,
                model_name=MODEL_NAME,
                question=problem["question"],
                answer_str=problem["answer"],
                max_tokens=2048,  # Smaller for quick experiments
                seed=42,
            )

            summary = tts_result.summary()
            results.append(summary)
            all_tts_scores.extend(summary["per_step_tts"])

            logger.info(f"\nResult: {json.dumps(summary, indent=2)}")

        except Exception as e:
            logger.error(f"Failed on problem {i+1}: {e}", exc_info=True)
            results.append({"question": problem["question"][:100], "error": str(e)})

    # Aggregate statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"=== AGGREGATE RESULTS ===")
    logger.info(f"Total problems: {len(results)}")
    logger.info(f"Total steps analyzed: {len(all_tts_scores)}")

    if all_tts_scores:
        mean_tts = sum(all_tts_scores) / len(all_tts_scores)
        high_tts = sum(1 for t in all_tts_scores if t >= 0.7) / len(all_tts_scores)
        decorative = sum(1 for t in all_tts_scores if t <= 0.005) / len(all_tts_scores)

        logger.info(f"Mean TTS: {mean_tts:.4f}")
        logger.info(f"Fraction high TTS (>= 0.7): {high_tts:.4f}")
        logger.info(f"Fraction decorative (<= 0.005): {decorative:.4f}")
        logger.info(f"TTS distribution: {sorted(all_tts_scores, reverse=True)[:20]}")

        # Paper reports: mean TTS ~ 0.03, only 2.3% >= 0.7
        logger.info(f"\nComparison with paper (AIME, Qwen-7B):")
        logger.info(f"  Paper mean TTS: ~0.03  |  Ours: {mean_tts:.4f}")
        logger.info(f"  Paper high TTS: ~2.3%  |  Ours: {high_tts*100:.1f}%")

    # Save results
    results_path = log_dir / "tts_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "problems": results,
                "aggregate": {
                    "n_problems": len(results),
                    "n_steps": len(all_tts_scores),
                    "mean_tts": sum(all_tts_scores) / len(all_tts_scores) if all_tts_scores else 0,
                    "frac_high_tts": (
                        sum(1 for t in all_tts_scores if t >= 0.7) / len(all_tts_scores)
                        if all_tts_scores
                        else 0
                    ),
                    "frac_decorative": (
                        sum(1 for t in all_tts_scores if t <= 0.005) / len(all_tts_scores)
                        if all_tts_scores
                        else 0
                    ),
                },
            },
            f,
            indent=2,
        )
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
