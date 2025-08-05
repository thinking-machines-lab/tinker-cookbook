"""Manual test script for TextArena TicTacToe with step-by-step actions."""

import asyncio

import anyio
import chz
from termcolor import colored
from tinker import types
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.textarena_envs import (
    SinglePlayerEnvGroupBuilder,
    TwoPlayerCoordinator,
    TwoPlayerEnvGroupBuilder,
)
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer


async def get_async_input(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)


class ManualPolicy(TokenCompleter):
    def __init__(
        self,
        tokenizer: Tokenizer,
        player_name: str,
        player_id: int,
        coordinator: TwoPlayerCoordinator | None,
    ):
        self.tokenizer = tokenizer
        self.player_name = player_name
        self.player_id = player_id
        self.coordinator = coordinator

    async def __call__(self, ob: types.ModelInput, stop: StopCondition) -> TokensWithLogprobs:
        # Wait for our turn before prompting for input
        print(f"Player {self.player_id} waiting for turn...")
        if self.coordinator:
            await self.coordinator.wait_for_turn(self.player_id)
            print(
                f"Player {self.player_id} got turn! Current player in env: {self.coordinator.current_player_id}"
            )
        observation_str = self.tokenizer.decode(ob.to_ints())
        print(colored(f"\n{self.player_name}'s turn:", "green"))
        print(colored(observation_str, "blue"))
        action_str = await get_async_input(f"{self.player_name} - Enter your action: ")
        action_tokens = self.tokenizer.encode(action_str, add_special_tokens=False)
        return TokensWithLogprobs(tokens=action_tokens, maybe_logprobs=None)


async def do_manual_game(game_name: str, players: int):
    """Test TicTacToe with manual actions."""
    # Setup
    tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B")

    # Create environments
    if players == 1:
        env_group_builder = SinglePlayerEnvGroupBuilder(
            game_name=game_name, tokenizer=tokenizer, num_envs=1
        )
        print("Creating 1 environment...")
        envs = await env_group_builder.make_envs()
        env0 = envs[0]
        policy = ManualPolicy(
            tokenizer=tokenizer, player_name="Player 0", player_id=0, coordinator=None
        )
        trajectory = await do_single_rollout(policy, env0)
        print(f"Rewards: {[transition.reward for transition in trajectory.transitions]}")
    else:
        print("Creating 2 environments...")
        env_group_builder = TwoPlayerEnvGroupBuilder(
            game_name=game_name, tokenizer=tokenizer, num_envs=2
        )
        envs = await env_group_builder.make_envs()
        env0, env1 = envs[0], envs[1]

        # Get the shared coordinator from the environments
        coordinator = env0.coordinator  # type: ignore

        # Create policies with access to the coordinator
        policy0 = ManualPolicy(
            tokenizer=tokenizer, player_name="Player 0", player_id=0, coordinator=coordinator
        )
        policy1 = ManualPolicy(
            tokenizer=tokenizer, player_name="Player 1", player_id=1, coordinator=coordinator
        )

        # Run rollouts in parallel
        async with anyio.create_task_group() as tg:
            tg.start_soon(do_single_rollout, policy0, env0)
            tg.start_soon(do_single_rollout, policy1, env1)


@chz.chz
class Config:
    game_name: str  # GuessTheNumber-v0, TicTacToe-v0, etc.
    players: int


async def main(cfg: Config):
    # asyncio.create_task(debug_tasks())  # Run in background
    await do_manual_game(cfg.game_name, cfg.players)


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
