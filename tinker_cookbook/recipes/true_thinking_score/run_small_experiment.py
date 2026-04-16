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
from pathlib import Path

import tinker

from tinker_cookbook.recipes.true_thinking_score.tts import generate_cot_and_compute_tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Small set of math problems with known answers for initial validation.
# Mix of difficulty levels to see if TTS varies.
PROBLEMS = [
    # AMC-level problems (competition math, should produce longer CoT)
    {
        "question": ("How many positive integers less than 100 are divisible by 3, 5, or 7?"),
        "answer": "54",
    },
    {
        "question": (
            "Let $S$ be the set of all positive integers $n$ such that $n^2 \\equiv 1 \\pmod{24}$. "
            "What is the sum of all elements of $S$ that are less than or equal to 100?"
        ),
        "answer": "2500",
    },
    {
        "question": (
            "A bag contains 5 red balls and 3 blue balls. Two balls are drawn without replacement. "
            "What is the probability that both balls are red? Express your answer as a common fraction."
        ),
        "answer": "5/14",
    },
]

# Use a small model for faster iteration
MODEL_NAME = "Qwen/Qwen3.5-4B"


async def main():
    log_dir = Path("/tmp/tinker-tts-experiment")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== TTS Small-Scale Experiment ===")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Problems: {len(PROBLEMS)}")
    logger.info(f"Log dir: {log_dir}")

    service_client = tinker.ServiceClient()

    results: list[dict] = []
    all_tts_scores: list[float] = []

    for i, problem in enumerate(PROBLEMS):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Problem {i + 1}/{len(PROBLEMS)}: {problem['question'][:60]}...")
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
            logger.error(f"Failed on problem {i + 1}: {e}", exc_info=True)
            results.append({"question": problem["question"][:100], "error": str(e)})

    # Aggregate statistics
    logger.info(f"\n{'=' * 60}")
    logger.info("=== AGGREGATE RESULTS ===")
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
        logger.info("\nComparison with paper (AIME, Qwen-7B):")
        logger.info(f"  Paper mean TTS: ~0.03  |  Ours: {mean_tts:.4f}")
        logger.info(f"  Paper high TTS: ~2.3%  |  Ours: {high_tts * 100:.1f}%")

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
