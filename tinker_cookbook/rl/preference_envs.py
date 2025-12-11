import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Sequence

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.preference.preference_datasets import (
    ComparisonDatasetBuilder,
)
from tinker_cookbook.preference.types import (
    Comparison,
    LabeledComparison,
    PreferenceModel,
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
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter
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
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )


class TournamentPattern(StrEnum):
    ALL_PAIRS_BOTH_WAYS = "all_pairs_both_ways"
    ALL_PAIRS_ONE_WAY = "all_pairs_one_way"


def get_pairs_chunked(n: int, pattern: TournamentPattern, chunk_size: int) -> list[tuple[int, int]]:
    """
    Get pairs of indices of matchups of n players. If chunk_size < n, then we divide the players
    into groups of at most chunk_size and get the matchup indices within each group.
    """
    out = []
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        pairs = get_pairs(chunk_end - chunk_start, pattern)
        for i, j in pairs:
            out.append((chunk_start + i, chunk_start + j))
    return out


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
    matchup_group_size: int = 4  # divide group into smaller groups of this size for matchups
    eval_target_completion_A: list[renderers.Message] | None = None

    async def make_envs(self) -> Sequence[Env]:
        return [
            PreferenceEnv(self.convo_prefix, self.policy_renderer) for _ in range(self.num_envs)
        ]

    def _preprocess_message(self, message: renderers.Message) -> renderers.Message:
        if self.content_preprocessor is not None:
            content = renderers.ensure_text(message["content"])
            message = {**message, "content": self.content_preprocessor(content)}
        return message

    def get_response_message(self, trajectory: Trajectory) -> tuple[list[renderers.Message], bool]:
        response, is_valid = self.policy_renderer.parse_response(
            trajectory.transitions[0].ac.tokens
        )
        return [response], is_valid

    def comparison_reward_for_second_messages(
        self, message_i: list[renderers.Message], message_j: list[renderers.Message]
    ) -> Comparison:
        comparison = Comparison(
            prompt_conversation=self.convo_prefix,
            completion_A=[self._preprocess_message(m) for m in message_i],
            completion_B=[self._preprocess_message(m) for m in message_j],
        )
        return comparison

    @logtree.scope_header_decorator
    async def compute_group_rewards(
        self,
        trajectory_group: list[Trajectory],
        env_group: Sequence[Env],
    ) -> list[tuple[float, Metrics]]:
        assert all(len(trajectory.transitions) == 1 for trajectory in trajectory_group)
        # Get response from each trajectory
        response_tuples = [self.get_response_message(trajectory) for trajectory in trajectory_group]
        response_messages, is_valid_list = safezip(*response_tuples)

        # Log prompt
        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=self.convo_prefix))

        # Log trajectories
        for idx, (messages, is_valid) in enumerate(
            zip(response_messages, is_valid_list, strict=True)
        ):
            with logtree.scope_header(f"Completion {idx}"):
                logtree.log_formatter(ConversationFormatter(messages=messages))
                logtree.log_text(f"Valid format: {is_valid}")

        # if the matching group size is 3 and len(response_messages) is 6
        # then it will return something like
        # [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
        # so we don't end up with O(n^2) comparisons
        comparison_indices_pairs = get_pairs_chunked(
            len(response_messages), self.tournament_pattern, self.matchup_group_size
        )

        logtree.log_text(
            f"Got {len(trajectory_group)} trajectories, doing {len(comparison_indices_pairs)} pairwise matchups."
        )

        j_comparisons = [
            self.comparison_reward_for_second_messages(
                message_i=response_messages[i], message_j=response_messages[j]
            )
            for i, j in comparison_indices_pairs
        ]

        # Log each pairwise comparison with its reward
        with logtree.scope_header("Pairwise Comparisons"):
            j_rewards = []

            # Compute all rewards first
            for comparison in j_comparisons:
                reward = await self.preference_model(comparison)
                j_rewards.append(reward)

            # Log summary of all matchups
            for idx, ((i, j), reward) in enumerate(
                zip(comparison_indices_pairs, j_rewards, strict=True)
            ):
                logtree.log_text(f"Matchup {idx}: ({i} vs {j}) — Reward: {reward:.2f}")

        win_minus_loss_list = [0.0 for _ in range(len(response_messages))]
        matchup_count = [0 for _ in range(len(response_messages))]
        for (i, j), j_reward in safezip(comparison_indices_pairs, j_rewards):
            win_minus_loss_list[j] += j_reward
            win_minus_loss_list[i] -= j_reward
            matchup_count[j] += 1
            matchup_count[i] += 1
        format_coef = 1.0

        return [
            (
                win_minus_loss / matchup_count + format_coef * (float(is_valid) - 1.0),
                {"win_minus_loss": win_minus_loss / matchup_count, "format": is_valid},
            )
            for win_minus_loss, is_valid, matchup_count in safezip(
                win_minus_loss_list, is_valid_list, matchup_count
            )
        ]

    def logging_tags(self) -> list[str]:
        return ["pair_pref"]


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
    batch_size: int
    policy_renderer_name: str
    policy_model_name: str
    tournament_pattern: TournamentPattern = TournamentPattern.ALL_PAIRS_BOTH_WAYS
    group_size: int
    content_preprocessor: Callable[[str], str] | None = None
    preference_model_builder: Callable[[], PreferenceModel]

    async def __call__(self) -> tuple[PairwisePreferenceDataset, None]:
        policy_renderer = renderers.get_renderer(
            self.policy_renderer_name, get_tokenizer(self.policy_model_name)
        )
        return PairwisePreferenceDataset(
            comparison_builder=self.comparison_builder,
            renderer=policy_renderer,
            batch_size=self.batch_size,
            preference_model=self.preference_model_builder(),
            tournament_pattern=self.tournament_pattern,
            group_size=self.group_size,
            content_preprocessor=self.content_preprocessor,
        ), None
