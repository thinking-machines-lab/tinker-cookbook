"""Reinforcement learning training and environment abstractions.

Provides the Env protocol, environment builders, dataset types, and the
RL training loop entry point.

Example::

    from tinker_cookbook.rl import Env, StepResult, EnvGroupBuilder, RLDataset
"""

from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    trajectory_to_data,
)
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.train import Config
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Logprobs,
    Logs,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

__all__ = [
    # Core types and type aliases
    "Action",
    "Observation",
    "Logprobs",
    "Metrics",
    "Logs",
    # Environment protocol
    "Env",
    "StepResult",
    "Transition",
    "Trajectory",
    "TrajectoryGroup",
    # Environment builders
    "EnvGroupBuilder",
    "ProblemEnv",
    "ProblemGroupBuilder",
    # Message-level environment abstractions
    "MessageEnv",
    "MessageStepResult",
    "EnvFromMessageEnv",
    # Dataset types
    "RLDataset",
    "RLDatasetBuilder",
    # Training config and entry point
    "Config",
    # Data processing utilities
    "compute_advantages",
    "assemble_training_data",
    "trajectory_to_data",
]
