import logging
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

logger = logging.getLogger(__name__)


class ProblemEnv(Env):
    """A single-turn Q&A environment that rewards correct answers and valid formatting."""

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

    @abstractmethod
    def get_question(self) -> str:
        """Return the question text for this problem."""
        pass

    @abstractmethod
    def check_answer(self, sample_str: str) -> bool:
        """Return a reward (0.0 to 1.0) for the model's response.

        Args:
            sample_str (str): The decoded text of the model's response.

        Returns:
            bool: Whether the answer is correct.
        """
        pass

    @abstractmethod
    def check_format(self, sample_str: str) -> bool:
        """Return a format compliance reward (0.0 to 1.0).

        Args:
            sample_str (str): The decoded text of the model's response.

        Returns:
            bool: Whether the response follows the expected format.
        """
        pass

    @abstractmethod
    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        pass

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Build the initial prompt from the conversation prefix and question."""
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """Score the model's response for correctness and format compliance.

        Args:
            action (Action): Token IDs of the model's response.
            extra (ActionExtra | None): Optional action metadata (unused).
        """
        convo = self.convo_prefix + [{"role": "user", "content": self.get_question()}]
        message, parse_success = self.renderer.parse_response(action)
        content = renderers.get_text_content(message)
        correct_format = float(parse_success) and float(self.check_format(content))
        correct_answer = float(self.check_answer(content))
        total_reward = self.format_coef * (correct_format - 1) + correct_answer

        # Log the attempt in a fixed structure that scales to longer content.
        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=convo))
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "reference_answer": self.get_reference_answer(),
                    "format_valid": bool(correct_format),
                    "correct": bool(correct_answer),
                    "format_coef": self.format_coef,
                    "reward": f"{total_reward:.3f}",
                },
                caption="Reward components",
            )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
            },
        )


@dataclass(frozen=True)
class ProblemGroupBuilder(EnvGroupBuilder):
    """Builds a group of ProblemEnv instances from a factory callable."""

    env_thunk: Callable[[], ProblemEnv]
    num_envs: int
    dataset_name: str = "problems"

    async def make_envs(self) -> Sequence[Env]:
        """Create ``num_envs`` ProblemEnv instances using the factory callable."""
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        """Return zero group rewards (all rewards come from per-step scoring)."""
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]
