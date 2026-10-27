"""Reinforcement learning: environment protocol, types, and training loops."""

from tinker_cookbook.rl.dataset_utils import (
    EpochCycledRLDataset,
    cycle_permutation,
    epoch_batch_index,
)
from tinker_cookbook.rl.interleaved import InterleavedRLDatasetBuilder
from tinker_cookbook.rl.rollout_strategy import FailFast, RetryOnFailure, RolloutStrategy
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
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
    "ActionExtra",
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
    # Dataset utilities (dataset_utils.py)
    "EpochCycledRLDataset",
    "cycle_permutation",
    "epoch_batch_index",
    # Interleaved datasets (interleaved.py)
    "InterleavedRLDatasetBuilder",
    # Rollout strategies (rollout_strategy.py)
    "FailFast",
    "RetryOnFailure",
    "RolloutStrategy",
]
