"""Bridge between BenchmarkBuilder and SamplingClientEvaluator.

Allows any benchmark to be used as an inline evaluator during RL or SFT
training — e.g., evaluate GSM8K every N training steps.

Usage::

    from tinker_cookbook.eval.benchmark_evaluator import BenchmarkEvaluator

    # In training config:
    evaluator_builders=[
        lambda: BenchmarkEvaluator("gsm8k", renderer, max_examples=100),
        lambda: BenchmarkEvaluator("ifeval", renderer, max_examples=50),
    ]
"""

from __future__ import annotations

import logging

import tinker

from tinker_cookbook.eval.benchmarks import run_benchmark
from tinker_cookbook.eval.benchmarks._types import BenchmarkConfig
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.renderers.base import Renderer

logger = logging.getLogger(__name__)


class BenchmarkEvaluator(SamplingClientEvaluator):
    """Run a benchmark as an inline evaluator during training.

    Bridges the benchmark framework with the training loop's evaluator
    interface. When called with a sampling client (at each eval step),
    it runs the benchmark on a small sample and returns metrics as a
    flat ``dict[str, float]`` for logging to TensorBoard / W&B.

    Returned metrics are prefixed with ``eval/{benchmark_name}/``
    (e.g. ``eval/gsm8k/score``, ``eval/gsm8k/num_correct``).

    Args:
        benchmark_name: Name of the benchmark in the registry
            (e.g. ``"gsm8k"``, ``"ifeval"``).
        renderer: Renderer for tokenization and prompt building.
        max_examples: Number of examples to evaluate per call.
            Default 100 — small enough for fast inline eval.
        max_tokens: Maximum generation tokens. Default 32768.
        temperature: Sampling temperature. Default 0.6.

    Example::

        # In training config:
        evaluator_builders = [
            lambda: BenchmarkEvaluator("gsm8k", renderer, max_examples=100),
            lambda: BenchmarkEvaluator("ifeval", renderer, max_examples=50),
        ]

        # Each eval step calls:
        metrics = await evaluator(sampling_client)
        # {"eval/gsm8k/score": 0.85, "eval/gsm8k/num_correct": 85, ...}
    """

    def __init__(
        self,
        benchmark_name: str,
        renderer: Renderer,
        max_examples: int = 100,
        max_tokens: int = 32768,
        temperature: float = 0.6,
    ):
        self.benchmark_name = benchmark_name
        self.renderer = renderer
        self.config = BenchmarkConfig(
            max_examples=max_examples,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """Run the benchmark and return metrics as a flat dict.

        Returns:
            Dict with keys like ``"eval/gsm8k/score"``, ``"eval/gsm8k/num_correct"``,
            ``"eval/gsm8k/num_examples"``, plus any benchmark-specific metrics.
        """
        result = await run_benchmark(
            self.benchmark_name,
            sampling_client,
            self.renderer,
            self.config,
        )

        prefix = f"eval/{self.benchmark_name}"
        metrics = {
            f"{prefix}/score": result.score,
            f"{prefix}/num_correct": float(result.num_correct),
            f"{prefix}/num_examples": float(result.num_examples),
            f"{prefix}/num_errors": float(result.num_errors),
        }
        # Include benchmark-specific metrics
        for key, value in result.metrics.items():
            if isinstance(value, (int, float)):
                metrics[f"{prefix}/{key}"] = float(value)

        return metrics
