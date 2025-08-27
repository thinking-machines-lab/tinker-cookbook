import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Sequence

import chz
import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.preference.preference_datasets import (
    ComparisonDatasetBuilder,
)
from tinker_cookbook.preference.types import (
    Comparison,
    LabeledComparison,
    PreferenceModel,
    PreferenceModelFromChatRenderer,
)
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.utils.misc_utils import safezip

logger = logging.getLogger(__name__)


class PreferenceEnv(Env):
    def __init__(
        self,
        convo_prefix: list[renderers.Message],
        policy_renderer: renderers.Renderer,
    ):
        self.convo_prefix = convo_prefix
        self.policy_renderer = policy_renderer

    @property
    def stop_condition(self) -> StopCondition:
        return self.policy_renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return self.policy_renderer.build_generation_prompt(self.convo_prefix), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """Compute the reward for a given action.

        Args:
            tokens: The tokens to compute the reward for.

        Returns:
            A tuple containing:
                - reward (float): The reward for the given action.
                - metrics (Dict[str, float]): Additional metrics to track.
        """
        return StepResult(
            reward=0,
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )


class TournamentPattern(StrEnum):
    ALL_PAIRS_BOTH_WAYS = "all_pairs_both_ways"
    ALL_PAIRS_ONE_WAY = "all_pairs_one_way"


def get_pairs(n: int, pattern: TournamentPattern) -> list[tuple[int, int]]:
    if pattern == TournamentPattern.ALL_PAIRS_BOTH_WAYS:
        return [(i, j) for i in range(n) for j in range(n) if i != j]
    elif pattern == TournamentPattern.ALL_PAIRS_ONE_WAY:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        raise ValueError(f"Invalid tournament pattern: {pattern}")


@dataclass(frozen=True)
class PairwisePreferenceGroupBuilder(EnvGroupBuilder):
    convo_prefix: list[renderers.Message]
    policy_renderer: renderers.Renderer
    tournament_pattern: TournamentPattern
    preference_model: PreferenceModel
    num_envs: int
    content_preprocessor: Callable[[str], str] | None = None  # e.g. strip out <thinking> tags

    async def make_envs(self) -> Sequence[Env]:
        return [
            PreferenceEnv(self.convo_prefix, self.policy_renderer) for _ in range(self.num_envs)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory]
    ) -> list[tuple[float, Metrics]]:
        assert all(len(trajectory.transitions) == 1 for trajectory in trajectory_group)
        # Get response from each trajectory
        response_tuples = [
            self.policy_renderer.parse_response(trajectory.transitions[0].ac.tokens)
            for trajectory in trajectory_group
        ]
        response_messages, is_valid_list = safezip(*response_tuples)

        pairs = get_pairs(len(response_messages), self.tournament_pattern)

        def preprocess_message(message: renderers.Message) -> renderers.Message:
            if self.content_preprocessor is not None:
                message = {**message, "content": self.content_preprocessor(message["content"])}
            return message

        async def compute_j_reward(i: int, j: int) -> float:
            comparison = Comparison(
                prompt_conversation=self.convo_prefix,
                completion_A=[preprocess_message(response_messages[i])],
                completion_B=[preprocess_message(response_messages[j])],
            )
            return await self.preference_model(comparison)

        pair_rewards = await asyncio.gather(*[compute_j_reward(i, j) for i, j in pairs])
        tournament_rewards = [0.0 for _ in range(len(response_messages))]
        for (i, j), reward in safezip(pairs, pair_rewards):
            tournament_rewards[j] += reward
            tournament_rewards[i] -= reward

        format_rewards = [float(is_valid) - 1.0 for is_valid in is_valid_list]
        format_coef = 1.0
        return [
            (
                tournament_reward + format_coef * format_reward,
                {"reward/tournament": tournament_reward, "reward/format": format_reward},
            )
            for tournament_reward, format_reward in safezip(tournament_rewards, format_rewards)
        ]


class PairwisePreferenceDataset(RLDataset):
    def __init__(
        self,
        comparison_builder: ComparisonDatasetBuilder,
        renderer: renderers.Renderer,
        batch_size: int,
        preference_model: PreferenceModel,
        tournament_pattern: TournamentPattern = TournamentPattern.ALL_PAIRS_BOTH_WAYS,
        group_size: int = 4,
        content_preprocessor: Callable[[str], str] | None = None,
    ):
        self.comparison_builder = comparison_builder
        self.renderer = renderer
        self.batch_size = batch_size
        self.preference_model = preference_model
        self.train_dataset, _ = self.comparison_builder.get_train_and_test_datasets()
        self.tournament_pattern = tournament_pattern
        self.group_size = group_size
        self.content_preprocessor = content_preprocessor

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        rows = self.train_dataset.select(
            range(index * self.batch_size, (index + 1) * self.batch_size)
        )
        lcs = [self.comparison_builder.example_to_labeled_comparison(row) for row in rows]  # type: ignore
        return [self._labeled_comparison_to_env_group(lc) for lc in lcs if lc is not None]

    def _labeled_comparison_to_env_group(self, lc: LabeledComparison) -> EnvGroupBuilder:
        return PairwisePreferenceGroupBuilder(
            convo_prefix=lc.comparison.prompt_conversation,
            policy_renderer=self.renderer,
            preference_model=self.preference_model,
            tournament_pattern=self.tournament_pattern,
            num_envs=self.group_size,
            content_preprocessor=self.content_preprocessor,
        )

    def __len__(self) -> int:
        return len(self.train_dataset) // self.batch_size


@chz.chz
class PairwisePreferenceRLDatasetBuilder(RLDatasetBuilder):
    comparison_builder: ComparisonDatasetBuilder
    renderer_name: str
    model_name_for_tokenizer: str
    batch_size: int
    tournament_pattern: TournamentPattern = TournamentPattern.ALL_PAIRS_BOTH_WAYS
    model_path: str
    group_size: int
    base_url: str | None = None
    content_preprocessor: Callable[[str], str] | None = None

    def __call__(self) -> tuple[PairwisePreferenceDataset, None]:
        print(f"base_url: {self.base_url}")
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        return PairwisePreferenceDataset(
            comparison_builder=self.comparison_builder,
            renderer=renderer,
            batch_size=self.batch_size,
            preference_model=PreferenceModelFromChatRenderer(
                convo_renderer=renderer,
                sampling_client=tinker.ServiceClient(base_url=self.base_url).create_sampling_client(
                    model_path=self.model_path
                ),
            ),
            tournament_pattern=self.tournament_pattern,
            group_size=self.group_size,
            content_preprocessor=self.content_preprocessor,
        ), None
