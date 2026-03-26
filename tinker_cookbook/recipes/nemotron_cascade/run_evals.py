"""
Run benchmark evaluations on Nemotron-Cascade-2 checkpoints.

Uses Inspect AI for standard benchmarks:
  - GSM8K (math)
  - IFEval (instruction following)
  - MMLU 0-shot (knowledge)
  - GPQA Diamond (hard science)

Can compare base model vs SFT vs IF-RL checkpoints.
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime

import tinker

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.eval.inspect_evaluators import InspectEvaluatorBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard benchmarks for Nemotron-Cascade evaluation
BENCHMARK_CONFIGS = {
    "quick": {
        "tasks": ["inspect_evals/gsm8k", "inspect_evals/ifeval"],
        "limit": 100,
        "description": "Quick eval: GSM8K (100) + IFEval (100)",
    },
    "standard": {
        "tasks": ["inspect_evals/gsm8k", "inspect_evals/ifeval", "inspect_evals/mmlu_0_shot"],
        "limit": 200,
        "description": "Standard eval: GSM8K + IFEval + MMLU (200 each)",
    },
    "full": {
        "tasks": [
            "inspect_evals/gsm8k",
            "inspect_evals/ifeval",
            "inspect_evals/mmlu_0_shot",
            "inspect_evals/gpqa_diamond",
        ],
        "limit": None,
        "description": "Full eval: GSM8K + IFEval + MMLU + GPQA (all examples)",
    },
    "math_only": {
        "tasks": ["inspect_evals/gsm8k"],
        "limit": None,
        "description": "Math only: full GSM8K",
    },
    "ifeval_only": {
        "tasks": ["inspect_evals/ifeval"],
        "limit": None,
        "description": "IF only: full IFEval",
    },
}


async def run_eval(
    model_name: str,
    checkpoint_path: str | None,
    benchmark: str,
    renderer_name: str | None = None,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    output_dir: str | None = None,
    include_reasoning: bool = True,
):
    """Run evaluation on a single checkpoint."""

    config = BENCHMARK_CONFIGS[benchmark]
    logger.info(f"Running {config['description']}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Checkpoint: {checkpoint_path or 'base model'}")

    if renderer_name is None:
        renderer_name = model_info.get_recommended_renderer_name(model_name)

    builder = InspectEvaluatorBuilder(
        tasks=config["tasks"],
        renderer_name=renderer_name,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        limit=config["limit"],
        debug_errors=True,
        log_dir=output_dir,
        max_connections=256,
        log_level="INFO",
        include_reasoning=include_reasoning,
    )

    evaluator = builder()

    # Create sampling client
    service_client = tinker.ServiceClient()

    if checkpoint_path:
        sampling_client = await service_client.create_sampling_client_from_state_async(
            checkpoint_path
        )
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        sampling_client = await service_client.create_sampling_client_async(
            base_model=model_name
        )
        logger.info("Using base model (no checkpoint)")

    # Run evaluation
    results = await evaluator(sampling_client)

    logger.info(f"Results:")
    for key, value in sorted(results.items()):
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "eval_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "model_name": model_name,
                "checkpoint_path": checkpoint_path,
                "benchmark": benchmark,
                "timestamp": datetime.now().isoformat(),
                "results": {k: float(v) if isinstance(v, float) else v for k, v in results.items()},
            }, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    return results


async def compare_checkpoints(
    model_name: str,
    checkpoints: dict[str, str | None],
    benchmark: str,
    renderer_name: str | None = None,
    output_dir: str | None = None,
):
    """Run evaluation on multiple checkpoints and compare."""
    all_results = {}

    for name, cp_path in checkpoints.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {name}")
        logger.info(f"{'='*60}")

        cp_output = os.path.join(output_dir, name) if output_dir else None
        results = await run_eval(
            model_name=model_name,
            checkpoint_path=cp_path,
            benchmark=benchmark,
            renderer_name=renderer_name,
            output_dir=cp_output,
        )
        all_results[name] = results

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    # Collect all metric names
    all_metrics = set()
    for results in all_results.values():
        all_metrics.update(results.keys())

    # Print header
    names = list(all_results.keys())
    header = f"{'Metric':<40}" + "".join(f"{n:<15}" for n in names)
    print(header)
    print("-" * len(header))

    for metric in sorted(all_metrics):
        row = f"{metric:<40}"
        for name in names:
            val = all_results[name].get(metric, "N/A")
            if isinstance(val, float):
                row += f"{val:<15.4f}"
            else:
                row += f"{str(val):<15}"
        print(row)

    # Save comparison
    if output_dir:
        comp_file = os.path.join(output_dir, "comparison.json")
        with open(comp_file, "w") as f:
            json.dump({
                "model_name": model_name,
                "benchmark": benchmark,
                "timestamp": datetime.now().isoformat(),
                "results": {
                    name: {k: float(v) if isinstance(v, float) else v for k, v in res.items()}
                    for name, res in all_results.items()
                },
            }, f, indent=2)
        logger.info(f"Comparison saved to {comp_file}")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluations")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b:peft:131072")
    parser.add_argument("--checkpoint", type=str, default=None, help="Single checkpoint to evaluate")
    parser.add_argument("--benchmark", type=str, default="quick",
                        choices=list(BENCHMARK_CONFIGS.keys()))
    parser.add_argument("--compare", action="store_true",
                        help="Compare base vs SFT vs IF-RL checkpoints")
    parser.add_argument("--sft-checkpoint", type=str, default=None)
    parser.add_argument("--ifrl-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    if args.compare:
        checkpoints = {"base": None}
        if args.sft_checkpoint:
            checkpoints["sft"] = args.sft_checkpoint
        if args.ifrl_checkpoint:
            checkpoints["ifrl"] = args.ifrl_checkpoint

        output_dir = args.output_dir or os.path.expanduser(
            f"~/data/nemotron-cascade-2/evals/{args.benchmark}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        asyncio.run(compare_checkpoints(
            model_name=args.model,
            checkpoints=checkpoints,
            benchmark=args.benchmark,
            output_dir=output_dir,
        ))
    else:
        output_dir = args.output_dir or os.path.expanduser(
            f"~/data/nemotron-cascade-2/evals/{args.benchmark}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        asyncio.run(run_eval(
            model_name=args.model,
            checkpoint_path=args.checkpoint,
            benchmark=args.benchmark,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_dir=output_dir,
        ))


if __name__ == "__main__":
    main()
