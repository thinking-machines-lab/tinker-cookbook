"""TextArena TicTacToe environment for tinker RL."""

import asyncio
from dataclasses import dataclass
from typing import ClassVar, Sequence

import chz
import textarena as ta
import tinker
from tinker import types
from tinker_cookbook.completers import StopCondition, TinkerMessageCompleter
from tinker_cookbook.renderers import Message, Renderer, get_renderer, get_text_content
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

STOP_CONDITION = ["]\n"]
ILLEGAL_MOVE_REWARD = -2.0


class TwoPlayerCoordinator:
    """Coordinates a single two player game between two players. See README.md in this folder for more details."""

    def __init__(self, shared_env: ta.Env):
        self.shared_env = shared_env  # Should already be reset
        self.condition = asyncio.Condition()
        self.illegal_player_id: int | None = None

    @property
    def state(self) -> ta.State:
        return self.shared_env.state  # type: ignore

    @property
    def current_player_id(self) -> int:
        """Get the current player ID from the environment state."""
        return self.state.current_player_id

    @property
    def game_done(self) -> bool:
        """Check if the game is done. Either the game state is done, or some player made an illegal move"""
        return self.state.done or self.illegal_player_id is not None

    @property
    def rewards(self) -> dict | None:
        """Get rewards from the environment state."""
        return self.state.rewards

    async def wait_across_env(self, player_id: int) -> None:
        """Wait until the opponent has finished their turn"""
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.current_player_id == player_id or self.game_done
            )

    async def make_move(self, player_id: int, move: str) -> None:
        """Make a move and notify waiting players."""
        async with self.condition:
            # Ensure it's actually this player's turn
            before_player_id = self.current_player_id

            if not self.game_done and (self.current_player_id != player_id):
                raise ValueError(
                    f"Not player {player_id}'s turn (current: {self.current_player_id})"
                )

            done, _ = self.shared_env.step(move)

            if done:
                self.shared_env.close()
            else:
                # we will know that the move is illegal if the next player's id has not changed after the move
                if self.current_player_id == before_player_id:
                    self.illegal_player_id = before_player_id

            # Notify all waiting players about the state change
            self.condition.notify_all()


@dataclass
class TwoPlayerEnv(Env):
    """Two player TextArena environment."""

    player_id: int  # 0 or 1
    coordinator: TwoPlayerCoordinator
    self_play: bool
    renderer: Renderer
    opponent_policy: TinkerMessageCompleter | None

    def __post_init__(self):
        assert self.self_play == (self.opponent_policy is None), (
            "If self_play is True, opponent_policy must be None"
        )

    @property
    def stop_condition(self) -> StopCondition:
        return STOP_CONDITION  # TextArena envs look for action in square brackets

    async def wait_for_turn(self) -> None:
        """If the game is not done, wait until the opponent's to finish playing their turn"""
        if not self.coordinator.game_done:
            if self.self_play:
                await self.coordinator.wait_across_env(self.player_id)
            else:
                await self.opponent_step()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        if self.player_id != 0:
            await self.wait_for_turn()
        return self.get_observation(), self.stop_condition

    async def opponent_step(self) -> None:
        """When not self_play, the opponent policy takes a step on the shared environment"""
        assert self.opponent_policy is not None
        opponent_player_id, opponent_observation_str = self.coordinator.shared_env.get_observation()
        assert isinstance(opponent_player_id, int) and isinstance(opponent_observation_str, str)
        assert opponent_player_id == 1 - self.player_id, (
            f"Opponent player ID should be 1 - [the id of the policy player], {opponent_player_id=}, {self.player_id=}"
        )
        opponent_convo: list[Message] = [{"role": "user", "content": opponent_observation_str}]
        opponent_response = await self.opponent_policy(opponent_convo)
        opponent_action_content = get_text_content(opponent_response)
        await self.coordinator.make_move(opponent_player_id, opponent_action_content)

    async def step(self, action: Action) -> StepResult:
        """Take a step in the environment."""
        if self.coordinator.game_done:
            return self.get_done_step()
        assert self.coordinator.current_player_id == self.player_id, "Not the current player's turn"

        # make a move ...
        action_message: Message = self.renderer.parse_response(action)[0]
        action_content = get_text_content(action_message)
        await self.coordinator.make_move(self.player_id, action_content)

        # we wait here rather than the beginning of this function,
        # because we want to know whether this player still needs to make a future move, and give that information to StepResult.
        # it is possible that this player's move is legal and hence make the game continue, kicking off another step in this step; however, the opponent ends the game.
        await self.wait_for_turn()
        return StepResult(
            reward=self.compute_reward(),
            episode_done=self.coordinator.game_done,
            next_observation=self.get_observation(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )

    def get_done_step(self) -> StepResult:
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=types.ModelInput.empty(),
            next_stop_condition=STOP_CONDITION,
            metrics={},
        )

    def compute_reward(self) -> float:
        if self.coordinator.illegal_player_id == self.player_id:
            return ILLEGAL_MOVE_REWARD
        if self.coordinator.rewards:
            return self.coordinator.rewards[self.player_id]
        return 0.0

    def get_observation(self) -> types.ModelInput:
        current_player_id, observation_str = self.coordinator.shared_env.get_observation()
        if not self.coordinator.game_done:
            assert isinstance(current_player_id, int) and isinstance(observation_str, str)
            assert current_player_id == self.player_id, (
                f"Observation should be for the current player, obs: {observation_str}, current_player_id: {current_player_id}, player_id: {self.player_id}"
            )
        return self.renderer.build_generation_prompt([{"role": "user", "content": observation_str}])


@dataclass
class TwoPlayerEnvGroupBuilder(EnvGroupBuilder):
    """Builder for groups of two player TextArena environments sharing the same game."""

    game_name: str
    renderer: Renderer
    num_envs: int
    self_play: bool
    num_players: ClassVar[int] = 2
    opponent_policy: TinkerMessageCompleter | None = None

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of environments sharing the same TextArena game."""
        if self.num_envs % 2 != 0:
            raise ValueError("this env requires an even number of environments (players)")

        def _construct_coordinator() -> TwoPlayerCoordinator:
            """During training, the coordinator performs necessary blocking/synchronization, so that the policys can take turns to make moves on the shared environment, across different Environment objects"""
            shared_env = ta.make(env_id=self.game_name)
            shared_env.reset(num_players=self.num_players)
            return TwoPlayerCoordinator(shared_env=shared_env)

        envs = []
        for _ in range(self.num_envs // 2):
            if self.self_play:
                coordinator = _construct_coordinator()
                # if self_play, then we need to share the same coordinator across all environments
                coordinators = [coordinator for _ in range(self.num_players)]
            else:
                # if not self_play, we can just create a different coordinator for each environment
                coordinators = [_construct_coordinator() for _ in range(self.num_players)]

            envs += [
                TwoPlayerEnv(
                    player_id=i,
                    coordinator=coordinators[i],
                    renderer=self.renderer,
                    self_play=self.self_play,
                    opponent_policy=self.opponent_policy,
                )
                for i in range(2)
            ]
        return envs


class TwoPlayerTextArenaDataset(RLDataset):
    """Dataset for TextArena environments."""

    def __init__(self, batch_size: int, builder: TwoPlayerEnvGroupBuilder, num_datapoints: int):
        self.batch_size = batch_size
        self.builder = builder
        self.num_datapoints = num_datapoints
        assert self.num_datapoints % self.builder.num_players == 0, (
            "num_datapoints must be divisible by num_players"
        )

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        return [
            self.builder
            for i in range(self.batch_size // self.builder.num_players)
            if (index * self.batch_size + self.builder.num_players * i) < self.num_datapoints
        ]

    def __len__(self) -> int:
        return self.num_datapoints // self.batch_size


@chz.chz
class TwoPlayerTextArenaDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    num_train_datapoints: int
    num_test_datapoints: int
    base_url: str | None = None
    model_name: str
    game_name: str
    renderer_name: str

    def _construct_opponent_policy(self, renderer: Renderer) -> TinkerMessageCompleter:
        """Play against a fixed policy during testing."""
        service_client = tinker.ServiceClient(base_url=self.base_url)
        sampling_client = service_client.create_sampling_client(base_model=self.model_name)
        return TinkerMessageCompleter(
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=64,
            stop_condition=STOP_CONDITION,
        )

    async def __call__(self) -> tuple[TwoPlayerTextArenaDataset, TwoPlayerTextArenaDataset | None]:
        """Build the dataset for training and testing."""
        renderer = get_renderer(self.renderer_name, get_tokenizer(self.model_name))

        # The training dataset performs self-play
        train_builder = TwoPlayerEnvGroupBuilder(
            game_name=self.game_name,
            renderer=renderer,
            num_envs=2,
            self_play=True,
        )
        train_dataset = TwoPlayerTextArenaDataset(
            batch_size=self.batch_size,
            builder=train_builder,
            num_datapoints=self.num_train_datapoints,
        )

        # The testing dataset plays against a fixed policy, constructed by self._opponent_policy
        test_builder = TwoPlayerEnvGroupBuilder(
            game_name=self.game_name,
            renderer=renderer,
            num_envs=2,
            self_play=False,
            opponent_policy=self._construct_opponent_policy(renderer),
        )
        test_dataset = TwoPlayerTextArenaDataset(
            batch_size=self.num_test_datapoints,
            builder=test_builder,
            num_datapoints=self.num_test_datapoints,
        )
        return train_dataset, test_dataset
