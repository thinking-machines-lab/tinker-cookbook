"""Reinforcement learning: environment protocol, types, and training loops."""

from tinker_cookbook.rl.rollout_strategy import FailFast, RetryOnFailure, RolloutStrategy
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Logs,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    RolloutError,
    StepResult,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

__all__ = [
    # Core protocol and types (types.py)
    "Action",
    "Env",
    "EnvGroupBuilder",
    "Logs",
    "Metrics",
    "Observation",
    "RLDataset",
    "RLDatasetBuilder",
    "RolloutError",
    "StepResult",
    "Trajectory",
    "TrajectoryGroup",
    "Transition",
    # Rollout strategies (rollout_strategy.py)
    "FailFast",
    "RetryOnFailure",
    "RolloutStrategy",
]
