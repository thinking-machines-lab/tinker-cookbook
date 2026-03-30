"""Core types for the benchmark evaluation framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env


@dataclass
class BenchmarkConfig:
    """Runtime configuration for benchmark evaluation.

    Controls concurrency, limits, storage, and generation parameters.
    Passed to :meth:`BenchmarkBuilder.make_envs` and the runner.
    """

    max_examples: int | None = None
    """Maximum number of examples to evaluate. ``None`` = all."""
    concurrency: int = 64
    """Maximum number of concurrent rollouts (semaphore bound)."""
    max_tokens: int = 32768
    """Maximum tokens per model generation."""
    temperature: float = 0.6
    """Sampling temperature for model generation."""
    save_dir: str | None = None
    """Directory for saving trajectories and results. ``None`` = no saving."""
    save_every: int = 50
    """Flush partial results to disk every N completed examples."""


@dataclass
class BenchmarkResult:
    """Aggregated result from running a benchmark.

    Attributes:
        name: Benchmark name (e.g. ``"gsm8k"``).
        score: Primary metric normalized to 0–1.
        num_examples: Total examples evaluated (excluding errors).
        num_correct: Examples graded as correct (reward > 0).
        metrics: Benchmark-specific additional metrics.
    """

    name: str
    score: float
    num_examples: int
    num_correct: int
    metrics: dict = field(default_factory=dict)


class BenchmarkBuilder(ABC):
    """Defines a benchmark as a list of Env instances.

    Subclass this to add new benchmarks. The framework handles everything
    else: rollouts, concurrency, storage, aggregation.

    A BenchmarkBuilder creates one :class:`Env` per evaluation example.
    Each Env is single-use — ``initial_observation()`` provides the prompt,
    ``step()`` grades the model's response and returns the reward.

    For single-turn benchmarks (most common), ``step()`` sets
    ``episode_done=True`` after one turn. For multi-turn benchmarks
    (terminal-bench, tau2-bench), ``step()`` drives the conversation.

    The same Env implementation can be used for both evaluation and RL
    training — no separate eval code needed.

    Example::

        class MyBenchmark(BenchmarkBuilder):
            name = "my_benchmark"

            def make_envs(self, renderer, config):
                ds = load_dataset("my/dataset", split="test")
                return [MyEnv(row, renderer) for row in ds]
    """

    name: str
    """Unique benchmark name used in the registry and file paths."""

    @abstractmethod
    def make_envs(
        self,
        renderer: Renderer,
        config: BenchmarkConfig,
    ) -> Sequence[Env]:
        """Create one Env per evaluation example.

        Args:
            renderer: Renderer for tokenization and prompt building.
            config: Runtime configuration (limits, tokens, etc.).

        Returns:
            Sequence of single-use Env instances.
        """
        ...

    def aggregate(
        self,
        rewards: list[float],
        metrics_list: list[dict],
    ) -> BenchmarkResult:
        """Aggregate per-example rewards into a BenchmarkResult.

        Override for custom aggregation (e.g., per-category breakdowns).
        Default: accuracy = fraction with reward > 0.

        Args:
            rewards: Total reward per example.
            metrics_list: Per-example metrics from Env.step().

        Returns:
            Aggregated BenchmarkResult.
        """
        num_correct = sum(1 for r in rewards if r > 0)
        return BenchmarkResult(
            name=self.name,
            score=num_correct / len(rewards) if rewards else 0.0,
            num_examples=len(rewards),
            num_correct=num_correct,
            metrics={},
        )
