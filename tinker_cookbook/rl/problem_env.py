import logging
from dataclasses import dataclass
from typing import Callable, Sequence

from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)

logger = logging.getLogger(__name__)


class ProblemEnv(Env):
    def __init__(
        self,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
    ):
        self.renderer = renderer
        self.convo_prefix = convo_prefix or []
        self.format_coef = format_coef

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def get_question(self) -> str:
        raise NotImplementedError

    def check_answer(self, sample_str: str) -> bool:
        raise NotImplementedError

    def check_format(self, sample_str: str) -> bool:
        raise NotImplementedError

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        correct_format = float(parse_success) and float(self.check_format(message["content"]))
        correct_answer = float(self.check_answer(message["content"]))
        total_reward = self.format_coef * (correct_format - 1) + correct_answer
        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "subreward/correct_format": correct_format,
                "subreward/correct_answer": correct_answer,
            },
        )


@dataclass(frozen=True)
class ProblemGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], ProblemEnv]
    num_envs: int

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in range(len(trajectory_group))]
