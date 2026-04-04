"""Utilities for combining multiple reward signals.

Provides combinators that take individual reward functions and produce a
single composite reward. Useful when training with multiple objectives
(e.g. correctness + format compliance + style).

Includes telemetry via ``tinker_cookbook.utils.trace`` (sync spans)
and ``tinker_cookbook.utils.logtree`` (HTML reports).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.trace import scope_span_sync


@dataclass(frozen=True)
class WeightedReward:
    """A named reward function with a weight for linear combination.

    ``fn`` is typed as ``Callable[..., float]`` because composites are
    designed for **already-bound** reward closures.  If your reward
    function needs extra arguments (e.g. a ground-truth answer), use
    :func:`functools.partial` to bind them first::

        from functools import partial
        from tinker_cookbook.rewards.math_rewards import grade_answer

        correctness = WeightedReward(
            name="math",
            fn=partial(grade_answer, ground_truth="42"),
            weight=1.0,
        )

    .. note::

        Async reward functions cannot be used directly in the synchronous
        combinators (:func:`weighted_sum`, :func:`reward_min`, etc.).
        Compose async rewards manually with ``asyncio.gather``.

    Attributes:
        name: Human-readable label (used as a key in metrics dicts).
        fn: Callable that takes text (and any pre-bound args) and returns
            a float reward.
        weight: Multiplicative weight in the linear combination.
    """

    name: str
    fn: Callable[..., float]
    weight: float = 1.0


def combine_weighted(text: str, rewards: Sequence[WeightedReward]) -> tuple[float, dict[str, float]]:
    """Compute a weighted sum of reward functions.

    Args:
        text: The model response text passed to each reward function.
        rewards: Sequence of ``WeightedReward`` entries.

    Returns:
        Tuple of ``(total_reward, per_reward_metrics)`` where the metrics
        dict maps each reward name to its *unweighted* score.
    """
    total = 0.0
    metrics: dict[str, float] = {}
    for r in rewards:
        score = r.fn(text)
        metrics[r.name] = score
        total += r.weight * score
    return total, metrics


def combine_min(text: str, fns: Sequence[Callable[[str], float]]) -> float:
    """Return the minimum reward across all functions.

    Useful as a "gate" -- all criteria must be met.
    """
    return min(fn(text) for fn in fns)


def combine_max(text: str, fns: Sequence[Callable[[str], float]]) -> float:
    """Return the maximum reward across all functions.

    Useful for "any-of" semantics where meeting one criterion suffices.
    """
    return max(fn(text) for fn in fns)


def combine_product(text: str, fns: Sequence[Callable[[str], float]]) -> float:
    """Return the product of all reward values.

    Useful when rewards are in [0, 1] and you want an "AND" gate
    (the product is 1 only if every individual reward is 1).
    """
    result = 1.0
    for fn in fns:
        result *= fn(text)
    return result


def combine_threshold(fn: Callable[[str], float], cutoff: float) -> Callable[[str], float]:
    """Wrap *fn* so it returns 1.0 when the score >= *cutoff*, else 0.0.

    Useful for converting a continuous reward into a binary signal.
    """

    def _thresholded(text: str) -> float:
        return 1.0 if fn(text) >= cutoff else 0.0

    return _thresholded


# ======================================================================
# Telemetry-instrumented composite rewards
# ======================================================================


def combine_weighted_traced(
    text: str,
    rewards: Sequence[WeightedReward],
    *,
    reward_name: str = "composite",
    log_to_logtree: bool = True,
) -> tuple[float, dict[str, float]]:
    """Compute a weighted sum of reward functions with tracing and logtree logging.

    Wraps :func:`weighted_sum` with telemetry:

    - A ``scope_span_sync`` trace span named ``"compute_{reward_name}_reward"``
    - Logtree table showing per-component scores and the total
    - A metrics dict with per-component scores and computation time

    Args:
        text: The model response text passed to each reward function.
        rewards: Sequence of ``WeightedReward`` entries.
        reward_name: Name for the composite reward (used in span name).
        log_to_logtree: Whether to emit logtree output.

    Returns:
        Tuple of ``(total_reward, metrics_dict)`` where ``metrics_dict``
        includes per-component scores, the total, and computation time.
    """
    t_start = time.perf_counter()

    with scope_span_sync(f"compute_{reward_name}_reward"):
        total, per_component = combine_weighted(text, rewards)

    elapsed = time.perf_counter() - t_start

    metrics: dict[str, float] = {}
    for name, score in per_component.items():
        metrics[f"reward/{reward_name}/{name}"] = score
    metrics[f"reward/{reward_name}/total"] = total
    metrics[f"reward/{reward_name}/computation_time"] = elapsed

    if log_to_logtree:
        with logtree.scope_header("Reward Computation"):
            table_data: dict[str, str | float] = {
                "reward_type": "composite_weighted_sum",
            }
            for name, score in per_component.items():
                table_data[f"component/{name}"] = score
            table_data["total"] = total
            table_data["computation_time"] = f"{elapsed:.4f}s"
            logtree.table_from_dict(table_data)

    return total, metrics


# ======================================================================
# Deprecated aliases (backward compatibility)
# ======================================================================

weighted_sum = combine_weighted
weighted_sum_with_trace = combine_weighted_traced
reward_min = combine_min
reward_max = combine_max
reward_product = combine_product
threshold = combine_threshold
