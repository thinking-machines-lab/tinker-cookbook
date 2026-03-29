"""
Benchmark evaluations for Nemotron-Cascade-2 checkpoints.

Uses Tinker sampling directly with our own grading:
  - GSM8K: Math grading via existing math_grading utilities
  - IFEval: Our 48-type instruction following verifier
  - MMLU-Pro: Multi-task language understanding
  - MATH-500: Hendrycks MATH test set
  - GPQA-Diamond: Graduate-level science QA (multiple choice)
  - AIME 2025: Math competition problems (integer answers 0-999)
  - MBPP: Python code generation with execution-based testing
  - LongBench v2: Long-context comprehension (multiple subtasks)
  - LiveCodeBench: Competitive programming (Pass@1)
  - MMLU-Redux: Verified MMLU subset
  - Arena-Hard: LLM-judge evaluation
  - BFCL: Function-calling accuracy
  - IFBench: Instruction following with objective verifiers

Compares base model vs SFT vs IF-RL checkpoints.
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime

import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.nemotron_cascade.eval.base import EvalResult
from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import BENCHMARKS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backward-compatible aliases: the old names mapped onto the new registry.
# The old ``BENCHMARKS`` dict used slightly different keys in some cases
# (e.g. "mmlu" -> "mmlu_pro", "aime2025" -> "aime").  Keep old keys working.
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {
    "mmlu": "mmlu_pro",
    "aime2025": "aime",
}


def _resolve_benchmark_name(name: str) -> str:
    return _ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------


async def run_eval(
    model_name: str,
    checkpoint_path: str | None,
    benchmarks: list[str],
    limit: int | None = None,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    output_dir: str | None = None,
) -> dict[str, float]:
    """Run evaluation on a single checkpoint.

    Returns a flat dict of metric_name -> value for backward compatibility.
    """
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = tinker.ServiceClient()
    if checkpoint_path:
        sampling_client = await service_client.create_sampling_client_async(model_path=checkpoint_path)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        sampling_client = await service_client.create_sampling_client_async(base_model=model_name)
        logger.info("Using base model")

    all_results: dict[str, float] = {}
    eval_results: list[EvalResult] = []

    for bench_name in benchmarks:
        resolved = _resolve_benchmark_name(bench_name)
        if resolved not in BENCHMARKS:
            logger.warning(f"Unknown benchmark: {bench_name}")
            continue
        logger.info(f"\n--- Running {resolved} ---")
        result = await BENCHMARKS[resolved](
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=max_tokens,
            max_examples=limit,
        )
        eval_results.append(result)
        # Flatten metrics into all_results for backward compat
        all_results.update(result.metrics)
        # Always include the primary score under the benchmark name
        all_results[f"{result.benchmark}/accuracy"] = result.score

    # Print results
    print("\n" + "=" * 50)
    cp_label = checkpoint_path.split("/")[-1] if checkpoint_path else "base"
    print(f"Results for: {cp_label}")
    print("=" * 50)
    for k, v in sorted(all_results.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(
                {
                    "model": model_name,
                    "checkpoint": checkpoint_path,
                    "timestamp": datetime.now().isoformat(),
                    "results": {k: float(v) if isinstance(v, float) else v for k, v in all_results.items()},
                    "eval_results": [
                        {
                            "benchmark": r.benchmark,
                            "score": r.score,
                            "num_examples": r.num_examples,
                            "num_correct": r.num_correct,
                            "metrics": r.metrics,
                        }
                        for r in eval_results
                    ],
                },
                f,
                indent=2,
            )

    return all_results


async def compare_checkpoints(
    model_name: str,
    checkpoints: dict[str, str | None],
    benchmarks: list[str],
    limit: int | None = None,
    output_dir: str | None = None,
):
    """Run evals on multiple checkpoints and print comparison."""
    all_results: dict[str, dict[str, float]] = {}
    for name, cp in checkpoints.items():
        logger.info(f"\n{'=' * 60}\nEvaluating: {name}\n{'=' * 60}")
        cp_out = os.path.join(output_dir, name) if output_dir else None
        results = await run_eval(model_name, cp, benchmarks, limit=limit, output_dir=cp_out)
        all_results[name] = results

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    metrics = sorted(set(k for r in all_results.values() for k in r.keys()))
    names = list(all_results.keys())
    header = f"{'Metric':<35}" + "".join(f"{n:<15}" for n in names)
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = f"{m:<35}"
        for n in names:
            v = all_results[n].get(m, "N/A")
            row += f"{v:<15.4f}" if isinstance(v, float) else f"{str(v):<15}"
        print(row)

    if output_dir:
        with open(os.path.join(output_dir, "comparison.json"), "w") as f:
            json.dump(all_results, f, indent=2)


def main():
    all_benchmark_names = sorted(set(list(BENCHMARKS.keys()) + list(_ALIASES.keys())))
    parser = argparse.ArgumentParser(description="Nemotron-Cascade-2 benchmark evaluation")
    parser.add_argument("--model", default="openai/gpt-oss-120b:peft:131072")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--benchmarks",
        default="gsm8k,ifeval",
        help=f"Comma-separated list. Available: {', '.join(all_benchmark_names)}",
    )
    parser.add_argument("--limit", type=int, default=100, help="Max samples per benchmark")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--sft-checkpoint", default=None)
    parser.add_argument("--ifrl-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    benchmarks = args.benchmarks.split(",")

    if args.compare:
        cps: dict[str, str | None] = {"base": None}
        if args.sft_checkpoint:
            cps["sft"] = args.sft_checkpoint
        if args.ifrl_checkpoint:
            cps["ifrl"] = args.ifrl_checkpoint
        out = args.output_dir or os.path.expanduser(
            f"~/data/nemotron-cascade-2/evals/compare_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        asyncio.run(compare_checkpoints(args.model, cps, benchmarks, limit=args.limit, output_dir=out))
    else:
        out = args.output_dir or os.path.expanduser(
            f"~/data/nemotron-cascade-2/evals/eval_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        asyncio.run(run_eval(args.model, args.checkpoint, benchmarks, limit=args.limit, output_dir=out))


if __name__ == "__main__":
    main()
