"""Run benchmarks on Qwen3.5-35B-A3B and compare with public scores."""

import asyncio
import json
import sys
import time
from pathlib import Path

import tinker

from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.eval.benchmarks import run_benchmark
from tinker_cookbook.eval.benchmarks._types import BenchmarkConfig

MODEL = "Qwen/Qwen3.5-35B-A3B"
SAVE_DIR = Path.home() / "experiments" / "eval_qwen35_35b"

# Public reference scores from HuggingFace model card
PUBLIC_SCORES = {
    "mmlu_pro": 85.3,
    "gpqa": 84.2,
    "ifeval": 91.9,
    "livecodebench": 74.6,
    "longbench": 59.0,
    "bfcl": 67.3,
}

# Phase 1: no sandbox needed
PHASE1_BENCHMARKS = [
    "gsm8k",
    "math500",
    "aime",
    "mmlu_pro",
    "gpqa",
    "ifeval",
    "longbench",
    "bfcl",
]

# Phase 2: sandbox needed
PHASE2_BENCHMARKS = [
    "mbpp",
    "livecodebench",
]


async def run_single(name, sampling_client, renderer, config):
    """Run a single benchmark and print result."""
    print(f"\n{'='*60}")
    print(f"Starting: {name}")
    print(f"{'='*60}")
    t0 = time.monotonic()
    try:
        result = await run_benchmark(name, sampling_client, renderer, config)
        elapsed = time.monotonic() - t0
        public = PUBLIC_SCORES.get(name)
        our_pct = result.score * 100
        delta_str = ""
        if public is not None:
            delta = our_pct - public
            delta_str = f"  (public: {public:.1f}, delta: {delta:+.1f})"

        print(f"\n  {name}: {our_pct:.1f}% "
              f"({result.num_correct}/{result.num_examples}, "
              f"{result.num_errors} errors, {elapsed:.0f}s)"
              f"{delta_str}")
        if result.metrics:
            for k, v in sorted(result.metrics.items()):
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")

        # Save result summary
        result_file = SAVE_DIR / f"{name}_result.json"
        with open(result_file, "w") as f:
            json.dump({
                "benchmark": name,
                "score": result.score,
                "score_pct": our_pct,
                "num_examples": result.num_examples,
                "num_correct": result.num_correct,
                "num_errors": result.num_errors,
                "time_seconds": elapsed,
                "public_score": public,
                "metrics": {k: v for k, v in result.metrics.items() if isinstance(v, (int, float))},
            }, f, indent=2)

        return name, result
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"\n  {name}: FAILED after {elapsed:.0f}s: {e}")
        return name, None


async def main():
    # Parse args
    benchmarks = PHASE1_BENCHMARKS
    if "--phase2" in sys.argv:
        benchmarks = PHASE2_BENCHMARKS
    elif "--all" in sys.argv:
        benchmarks = PHASE1_BENCHMARKS + PHASE2_BENCHMARKS
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        benchmarks = sys.argv[1:]

    print(f"Model: {MODEL}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Save dir: {SAVE_DIR}")

    # Setup
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    renderer_name = get_recommended_renderer_name(MODEL)
    tokenizer = get_tokenizer(MODEL)
    renderer = get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL)
    print("Sampling client created.")

    # Config
    use_sandbox = any(b in PHASE2_BENCHMARKS for b in benchmarks)
    config = BenchmarkConfig(
        save_dir=str(SAVE_DIR),
        concurrency=64,
        agent_concurrency=8,
        timeout_seconds=600,
        max_tokens=32768,
        temperature=0.6,
    )

    # If sandbox benchmarks, set up Modal
    if use_sandbox:
        try:
            from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox
            config.sandbox_factory = ModalSandbox.create
            print("Modal sandbox configured.")
        except ImportError:
            print("WARNING: Modal not available, skipping sandbox benchmarks")
            benchmarks = [b for b in benchmarks if b not in PHASE2_BENCHMARKS]

    # Run benchmarks sequentially (to avoid overloading the API)
    results = {}
    for name in benchmarks:
        bench_config = BenchmarkConfig(
            save_dir=str(SAVE_DIR / name),
            concurrency=config.concurrency,
            agent_concurrency=config.agent_concurrency,
            timeout_seconds=config.timeout_seconds,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            sandbox_factory=config.sandbox_factory,
        )
        (SAVE_DIR / name).mkdir(parents=True, exist_ok=True)
        name, result = await run_single(name, sampling_client, renderer, bench_config)
        if result is not None:
            results[name] = result

    # Print summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Benchmark':20s} {'Our Score':>10s} {'Public':>10s} {'Delta':>10s}")
    print("-" * 55)
    for name in benchmarks:
        r = results.get(name)
        if r is None:
            print(f"{name:20s} {'FAILED':>10s}")
            continue
        our_pct = r.score * 100
        public = PUBLIC_SCORES.get(name)
        if public is not None:
            delta = our_pct - public
            print(f"{name:20s} {our_pct:10.1f} {public:10.1f} {delta:+10.1f}")
        else:
            print(f"{name:20s} {our_pct:10.1f} {'N/A':>10s}")

    # Save combined summary
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "score": r.score,
            "score_pct": r.score * 100,
            "num_examples": r.num_examples,
            "num_correct": r.num_correct,
            "public_score": PUBLIC_SCORES.get(name),
        }
    with open(SAVE_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {SAVE_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
