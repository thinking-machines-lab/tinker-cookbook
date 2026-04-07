"""Adapter that routes RL test-set evaluation through the benchmark framework.

Bridges :class:`EnvGroupBuilder` (RL) into :class:`BenchmarkBuilder` (eval) so
that RL test datasets produce typed :class:`BenchmarkResult` objects with
trajectory storage, resumability, and consistent aggregation.

All RL recipes use ``group_size=1`` for test splits, so each
:class:`EnvGroupBuilder` produces exactly one :class:`Env`.  This module
leverages that invariant via :class:`_LazyEnv`, which defers the async
``EnvGroupBuilder.make_envs()`` to ``initial_observation()`` time.

Usage in training config::

    from tinker_cookbook.eval.rl_benchmark import RLTestSetBenchmarkEvaluator

    evaluator = RLTestSetBenchmarkEvaluator(
        test_dataset,
        max_tokens=config.max_tokens,
        renderer=renderer,
    )
    metrics = await evaluator(sampling_client)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

import tinker

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.eval.benchmarks._runner import run_benchmark
from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
    Metrics,
)
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.metric_util import dataset_to_env_group_builders
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    RLDataset,
    StepResult,
)
from tinker_cookbook.utils.misc_utils import dict_mean

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy env wrapper — bridges async EnvGroupBuilder.make_envs() into the
# sync BenchmarkBuilder.make_envs() interface.
# ---------------------------------------------------------------------------


class _LazyEnv(Env):
    """Wraps an :class:`EnvGroupBuilder` to produce a single :class:`Env` lazily.

    The async ``make_envs()`` call is deferred to ``initial_observation()``
    time, which the benchmark runner already calls in an async context.
    Asserts ``group_size == 1`` since all RL recipes use that for test splits.
    """

    def __init__(self, builder: EnvGroupBuilder, builder_idx: int) -> None:
        self._builder = builder
        self._builder_idx = builder_idx
        self._inner: Env | None = None

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        envs = await self._builder.make_envs()
        if len(envs) != 1:
            raise ValueError(
                f"RLTestSetBenchmarkBuilder expects group_size=1 for test evaluation, "
                f"but EnvGroupBuilder at index {self._builder_idx} produced {len(envs)} envs. "
                f"Set group_size=1 in your test dataset builder."
            )
        self._inner = envs[0]
        # Forward example_id for trajectory storage / resumability
        example_id = getattr(self._inner, "example_id", None)
        if example_id is not None:
            self.example_id = example_id
        return await self._inner.initial_observation()

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        assert self._inner is not None, "_LazyEnv.step() called before initial_observation()"
        return await self._inner.step(action, extra=extra)

    async def cleanup(self) -> None:
        """Forward cleanup to both the inner env and the builder."""
        inner_cleanup = getattr(self._inner, "cleanup", None)
        if inner_cleanup is not None:
            await inner_cleanup()
        await self._builder.cleanup()


# ---------------------------------------------------------------------------
# Benchmark builder — wraps an RL test dataset as a BenchmarkBuilder.
# ---------------------------------------------------------------------------


class RLTestSetBenchmarkBuilder(BenchmarkBuilder):
    """Adapts a list of :class:`EnvGroupBuilder` instances into a :class:`BenchmarkBuilder`.

    Creates one :class:`_LazyEnv` per builder (each producing one Env).
    Custom :meth:`aggregate` computes RL-style metrics (mean reward, per-tag
    breakdowns) and packs them into :class:`BenchmarkResult.metrics`.

    Args:
        env_group_builders: Flattened list of env group builders from the RL
            test dataset (each expected to have ``group_size=1``).
        name: Benchmark name used in metrics and storage paths.
    """

    def __init__(
        self,
        env_group_builders: Sequence[EnvGroupBuilder],
        name: str = "rl_test",
    ) -> None:
        self.name = name
        self._builders = list(env_group_builders)
        self._tags_per_builder = [b.logging_tags() for b in env_group_builders]

    def make_envs(
        self,
        renderer: Renderer,
        config: BenchmarkConfig,
    ) -> Sequence[Env]:
        """Create one :class:`_LazyEnv` per builder.

        The ``renderer`` and ``config`` arguments are ignored — RL envs have
        their own renderer baked in.  The benchmark runner still needs a
        renderer for trajectory decoding (tokenizer), which is passed
        separately to :func:`run_benchmark`.
        """
        return [_LazyEnv(builder, idx) for idx, builder in enumerate(self._builders)]

    def aggregate(
        self,
        rewards: list[float],
        metrics_list: list[Metrics],
    ) -> BenchmarkResult:
        """Aggregate per-example rewards into a :class:`BenchmarkResult`.

        Computes standard accuracy plus RL-style metrics:

        - ``reward/total``: mean reward across all examples
        - Per-example metric means (``correct``, ``format``, etc.)
        - Per-tag breakdowns (``env/{tag}/reward/total``, etc.) when tags exist

        Index alignment: the benchmark runner processes envs in order and
        ``valid_rewards`` excludes only unrun (``None``) examples.  Since we
        don't use resumability, ``len(rewards) == len(self._builders)`` and
        positional indices align with ``self._tags_per_builder``.
        """
        num_correct = sum(1 for r in rewards if r > 0)
        score = num_correct / len(rewards) if rewards else 0.0

        agg: dict[str, float | int] = {}

        # Emit metrics under env/all/ prefix for backward compatibility with
        # RLTestSetEvaluator's compute_trajectory_metrics() output format.
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        agg["env/all/reward/total"] = mean_reward

        if metrics_list:
            means = dict_mean(metrics_list)
            for k, v in means.items():
                agg[f"env/all/{k}"] = v

        # Per-tag aggregation using positional alignment with self._tags_per_builder
        self._aggregate_per_tag(rewards, metrics_list, agg)

        return BenchmarkResult(
            name=self.name,
            score=score,
            num_examples=len(rewards),
            num_correct=num_correct,
            metrics=agg,
        )

    def _aggregate_per_tag(
        self,
        rewards: list[float],
        metrics_list: list[Metrics],
        out: dict[str, float | int],
    ) -> None:
        """Compute per-tag reward and metric means, writing into ``out``.

        Uses positional alignment: ``rewards[i]`` corresponds to
        ``self._tags_per_builder[i]``.  Only emits per-tag metrics when
        tags select a strict subset of all examples (same convention as
        :func:`compute_trajectory_metrics`).
        """
        tag_rewards: dict[str, list[float]] = defaultdict(list)
        tag_metrics: dict[str, list[Metrics]] = defaultdict(list)

        for i, (reward, per_example_metrics) in enumerate(zip(rewards, metrics_list)):
            if i >= len(self._tags_per_builder):
                break
            for tag in self._tags_per_builder[i]:
                tag_rewards[tag].append(reward)
                tag_metrics[tag].append(per_example_metrics)

        n_total = len(rewards)
        for tag, tag_r in tag_rewards.items():
            if len(tag_r) >= n_total:
                continue
            out[f"env/{tag}/reward/total"] = sum(tag_r) / len(tag_r)
            if tag_metrics[tag]:
                for k, v in dict_mean(tag_metrics[tag]).items():
                    out[f"env/{tag}/{k}"] = v


# ---------------------------------------------------------------------------
# Evaluator — drop-in replacement for RLTestSetEvaluator in training loops.
# ---------------------------------------------------------------------------


class RLTestSetBenchmarkEvaluator(SamplingClientEvaluator):
    """Evaluates an RL test dataset using the benchmark framework.

    Drop-in replacement for :class:`RLTestSetEvaluator` that routes evaluation
    through :func:`run_benchmark`, producing a typed :class:`BenchmarkResult`
    with optional trajectory storage.

    Returns a ``dict[str, float]`` compatible with the training loop's metric
    logging, with keys prefixed by ``name`` (default ``"test"``).

    Args:
        dataset: The RL test dataset to evaluate.
        max_tokens: Maximum generation tokens per example.
        renderer: Renderer for tokenization (needed by the benchmark runner
            for trajectory decoding).  Construct via
            ``renderers.get_renderer(name, tokenizer)``.
        name: Metric key prefix (default ``"test"``).
        save_dir: Directory for trajectory storage.  When set, the benchmark
            runner writes ``StoredTrajectory`` JSONL and ``result.json``.
        temperature: Sampling temperature (default ``1.0``).
        concurrency: Max concurrent rollouts (default ``64``).

    Example::

        evaluator = RLTestSetBenchmarkEvaluator(
            test_dataset,
            max_tokens=1024,
            renderer=renderer,
            save_dir="/path/to/eval_results",
        )
        metrics = await evaluator(sampling_client)
        # {"test/score": 0.85, "test/reward/total": 0.83, "test/correct": 0.85, ...}
    """

    def __init__(
        self,
        dataset: RLDataset,
        max_tokens: int,
        renderer: Renderer,
        name: str = "test",
        save_dir: str | None = None,
        temperature: float = 1.0,
        concurrency: int = 64,
    ) -> None:
        builders = dataset_to_env_group_builders(dataset)
        self._benchmark = RLTestSetBenchmarkBuilder(builders, name=name)
        self._renderer = renderer
        self._name = name
        self._config = BenchmarkConfig(
            max_tokens=max_tokens,
            save_dir=save_dir,
            temperature=temperature,
            concurrency=concurrency,
        )
        self.last_result: BenchmarkResult | None = None
        """The most recent :class:`BenchmarkResult`, available after ``__call__``."""

    async def __call__(
        self,
        sampling_client: tinker.SamplingClient,
    ) -> dict[str, float]:
        """Run the benchmark and return metrics as a flat dict.

        The :class:`BenchmarkResult` is also stored in :attr:`last_result`
        for callers that want the typed result.

        Returns:
            Dict with keys like ``"test/score"``, ``"test/reward/total"``,
            ``"test/correct"``, ``"test/num_errors"``, etc.
        """
        result = await run_benchmark(
            self._benchmark,
            sampling_client,
            self._renderer,
            self._config,
        )
        self.last_result = result

        # Convert BenchmarkResult to flat dict with name prefix
        metrics: dict[str, float] = {
            f"{self._name}/score": result.score,
            f"{self._name}/num_correct": float(result.num_correct),
            f"{self._name}/num_examples": float(result.num_examples),
            f"{self._name}/num_errors": float(result.num_errors),
            f"{self._name}/num_truncated": float(result.num_truncated),
        }

        # Include benchmark-specific metrics (reward/total, correct, format, per-tag, etc.)
        for key, value in result.metrics.items():
            if isinstance(value, (int, float)):
                metrics[f"{self._name}/{key}"] = float(value)

        return metrics
