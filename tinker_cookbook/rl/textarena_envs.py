"""TextArena TicTacToe environment for tinker RL."""

import asyncio
from dataclasses import dataclass
from typing import Sequence

import chz
import textarena as ta
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_public import types

from . import types as rl_types


class TextArenaEnvGroupBuilder(rl_types.EnvGroupBuilder):
    async def compute_group_rewards(
        self, trajectory_group: list[rl_types.Trajectory]
    ) -> list[tuple[float, rl_types.Metrics]]:
        return [(0.0, {}) for trajectory in trajectory_group]
        # ^^^ single-step rewards are handled by the environment


@dataclass
class SinglePlayerEnv(rl_types.Env):
    """Single player TextArena environment."""

    player_id: int  # 0 or 1
    tokenizer: Tokenizer
    env: ta.Env

    @property
    def stop_condition(self) -> StopCondition:
        return ["]\n"]  # TextArena envs look for action in square brackets

    async def initial_observation(self) -> tuple[rl_types.Observation, rl_types.StopCondition]:
        return self.get_observation(), self.stop_condition

    async def step(self, action: rl_types.Action) -> rl_types.StepResult:
        """Take a step in the environment."""
        action_text = self.tokenizer.decode(action).strip()
        self.env.step(action_text)
        if self.env.state.done:  # type: ignore
            self.env.close()
        observation = self.get_observation()
        return rl_types.StepResult(
            reward=self.env.state.rewards[self.player_id] if self.env.state.rewards else 0.0,  # type: ignore
            episode_done=self.env.state.done,  # type: ignore
            next_observation=observation,
            metrics={},
        )

    def get_observation(self) -> types.ModelInput:
        _current_player_id, observation_str = self.env.get_observation()
        observation_tokens = self.tokenizer.encode(observation_str)
        return types.ModelInput.from_ints(tokens=observation_tokens)


@dataclass
class SinglePlayerEnvGroupBuilder(TextArenaEnvGroupBuilder):
    """Builder for groups of single player TextArena environments."""

    game_name: str
    tokenizer: Tokenizer
    num_envs: int

    async def make_envs(self) -> Sequence[rl_types.Env]:
        """Create a group of environments sharing the same TextArena game."""
        envs = []
        for _ in range(self.num_envs):
            env = ta.make(env_id=self.game_name)
            env = ta.wrappers.LLMObservationWrapper(env)
            env.reset(num_players=1)
            envs.append(SinglePlayerEnv(player_id=0, tokenizer=self.tokenizer, env=env))
        return envs


class TwoPlayerCoordinator:
    """Coordinates a single two player game between two players."""

    def __init__(self, shared_env: ta.Env):
        self.shared_env = shared_env  # Should already be resetted
        self.condition = asyncio.Condition()

    @property
    def state(self) -> ta.State:
        return self.shared_env.state  # type: ignore

    @property
    def current_player_id(self) -> int:
        """Get the current player ID from the environment state."""
        return self.state.current_player_id

    @property
    def game_done(self) -> bool:
        """Check if the game is done from the environment state."""
        return self.state.done

    @property
    def rewards(self) -> dict | None:
        """Get rewards from the environment state."""
        return self.state.rewards

    async def wait_for_turn(self, player_id: int) -> None:
        """Wait until it's this player's turn or the game is done."""
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.current_player_id == player_id or self.game_done
            )

    async def make_move(self, player_id: int, move: str) -> None:
        """Make a move and notify waiting players."""
        async with self.condition:
            # Ensure it's actually this player's turn
            if not self.game_done and self.current_player_id != player_id:
                raise ValueError(
                    f"Not player {player_id}'s turn (current: {self.current_player_id})"
                )

            done, _info = self.shared_env.step(move)
            if done:
                self.shared_env.close()

            # Notify all waiting players about the state change
            self.condition.notify_all()


@dataclass
class TwoPlayerEnv(rl_types.Env):
    """Two player TextArena environment."""

    player_id: int  # 0 or 1
    coordinator: TwoPlayerCoordinator
    tokenizer: Tokenizer

    @property
    def stop_condition(self) -> StopCondition:
        return ["]\n"]  # TextArena envs look for action in square brackets

    async def initial_observation(self) -> tuple[rl_types.Observation, rl_types.StopCondition]:
        await self.coordinator.wait_for_turn(self.player_id)
        return self.get_observation(), self.stop_condition

    async def step(self, action: rl_types.Action) -> rl_types.StepResult:
        """Take a step in the environment."""
        assert not self.coordinator.game_done, "Episode already done"
        await self.coordinator.wait_for_turn(self.player_id)
        action_text = self.tokenizer.decode(action).strip()
        await self.coordinator.make_move(self.player_id, action_text)
        await self.coordinator.wait_for_turn(self.player_id)
        return rl_types.StepResult(
            reward=self.coordinator.rewards[self.player_id] if self.coordinator.rewards else 0.0,
            episode_done=self.coordinator.game_done,
            next_observation=self.get_observation(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )

    def get_observation(self) -> types.ModelInput:
        current_player_id, observation_str = self.coordinator.shared_env.get_observation()
        observation_tokens = self.tokenizer.encode(observation_str)
        return types.ModelInput.from_ints(tokens=observation_tokens)


@dataclass
class TwoPlayerEnvGroupBuilder(TextArenaEnvGroupBuilder):
    """Builder for groups of two player TextArena environments sharing the same game."""

    game_name: str
    tokenizer: Tokenizer
    num_envs: int

    async def make_envs(self) -> Sequence[rl_types.Env]:
        """Create a group of environments sharing the same TextArena game."""
        if self.num_envs % 2 != 0:
            raise ValueError("this env requires an even number of environments (players)")

        envs = []
        for i in range(self.num_envs // 2):
            # Create a single shared TextArena environment
            shared_env = ta.make(env_id=self.game_name)
            shared_env = ta.wrappers.LLMObservationWrapper(shared_env)
            shared_env.reset(num_players=2)
            # Create the game coordinator
            coordinator = TwoPlayerCoordinator(shared_env=shared_env)

            # Create two environment wrappers, one for each player
            envs += [
                TwoPlayerEnv(player_id=0, coordinator=coordinator, tokenizer=self.tokenizer),
                TwoPlayerEnv(player_id=1, coordinator=coordinator, tokenizer=self.tokenizer),
            ]
        return envs


class TextArenaDataset(rl_types.RLDataset):
    """Dataset for TextArena environments."""

    def __init__(self, batch_size: int, builder: TextArenaEnvGroupBuilder):
        self.batch_size = batch_size
        self.builder = builder

    def get_batch(self, index: int) -> list[rl_types.EnvGroupBuilder]:
        return [self.builder for _ in range(self.batch_size)]

    def __len__(self) -> int:
        return int(1e9)  # XXX it's infinite; handle this better


@chz.chz
class TextArenaDatasetBuilder(rl_types.RLDatasetBuilder):
    batch_size: int
    builder: TextArenaEnvGroupBuilder

    def __call__(self) -> TextArenaDataset:
        return TextArenaDataset(batch_size=self.batch_size, builder=self.builder)
