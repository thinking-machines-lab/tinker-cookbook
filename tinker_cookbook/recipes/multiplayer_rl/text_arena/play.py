"""Interactive play against a trained tic-tac-toe policy.

After training, use this script to play against the trained model interactively
or watch the model play against a random opponent.

Usage:

    # Play against the trained model (you go first as Player 0):
    python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
        checkpoint_path=<path_to_checkpoint>

    # Play as Player 1 (model goes first):
    python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
        checkpoint_path=<path_to_checkpoint> human_player_id=1

    # Watch the model play against a random opponent:
    python -m tinker_cookbook.recipes.multiplayer_rl.text_arena.play \
        checkpoint_path=<path_to_checkpoint> mode=model_vs_random
"""

import asyncio
from typing import Literal

import chz
import tinker
from termcolor import colored

from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.multiplayer_rl.text_arena.env import (
    RandomOpponent,
    TwoPlayerEnvGroupBuilder,
)
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.play_w_env import ManualPolicy, print_trajectory_summary
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer


async def play_game(
    renderer: Renderer,
    model_token_completer: TinkerTokenCompleter,
    human_player_id: int,
    game_name: str,
) -> None:
    """Play an interactive game: human vs trained model."""
    model_player_id = 1 - human_player_id

    # self_play=True so both envs share one coordinator (same board).
    # Both sides run via do_single_rollout — human with ManualPolicy, model with TinkerTokenCompleter.
    builder = TwoPlayerEnvGroupBuilder(
        game_name=game_name,
        renderer=renderer,
        num_envs=2,
        self_play=True,
    )
    envs = await builder.make_envs()
    human_env = envs[human_player_id]
    model_env = envs[model_player_id]
    human_policy = ManualPolicy(renderer.tokenizer, multiline=False, show_observation=True)

    print(colored(f"\nYou are Player {human_player_id}.", "cyan", attrs=["bold"]))
    print(colored("Type moves like: [4]", "cyan"))
    print(colored("=" * 40, "cyan"))

    human_traj, _model_traj = await asyncio.gather(
        do_single_rollout(human_policy, human_env),
        do_single_rollout(model_token_completer, model_env),
    )

    print(colored("\n--- Your Game Summary ---", "cyan", attrs=["bold"]))
    print_trajectory_summary(human_traj)


async def watch_model_vs_random(
    renderer: Renderer,
    model_token_completer: TinkerTokenCompleter,
    game_name: str,
    num_games: int = 5,
) -> None:
    """Watch the trained model play against a random opponent."""
    wins, losses, draws = 0, 0, 0

    for game_idx in range(num_games):
        print(colored(f"\n{'=' * 40} Game {game_idx + 1} {'=' * 40}", "cyan", attrs=["bold"]))

        builder = TwoPlayerEnvGroupBuilder(
            game_name=game_name,
            renderer=renderer,
            num_envs=2,
            self_play=False,
            opponent_policy=RandomOpponent(),
        )
        envs = await builder.make_envs()
        model_env = envs[0]  # Model plays as Player 0

        model_traj = await do_single_rollout(model_token_completer, model_env)

        total_reward = sum(t.reward for t in model_traj.transitions)
        if total_reward > 0:
            result = colored("WIN", "green")
            wins += 1
        elif total_reward < 0:
            result = colored("LOSS", "red")
            losses += 1
        else:
            result = colored("DRAW", "yellow")
            draws += 1
        print(f"  Result: {result} (reward={total_reward:.1f})")

    print(colored(f"\n{'=' * 40} Summary {'=' * 40}", "cyan", attrs=["bold"]))
    print(f"  Wins: {wins}/{num_games}, Draws: {draws}/{num_games}, Losses: {losses}/{num_games}")


@chz.chz
class PlayConfig:
    checkpoint_path: str
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None
    game_name: str = "TicTacToe-v0"
    mode: Literal["human_vs_model", "model_vs_random"] = "human_vs_model"
    human_player_id: int = 0  # 0 = go first, 1 = go second
    num_games: int = 5  # for model_vs_random mode


async def main(config: PlayConfig) -> None:
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    renderer = get_renderer(renderer_name, get_tokenizer(config.model_name))

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        model_path=config.checkpoint_path,
    )
    # Use TinkerTokenCompleter directly — avoids decode/re-encode round-trip
    # that _MessageCompleterAdapter would cause (double chat-template wrapping)
    model_token_completer = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=64,
    )

    if config.mode == "human_vs_model":
        while True:
            await play_game(
                renderer, model_token_completer, config.human_player_id, config.game_name
            )
            again = input(colored("\nPlay again? [y/n]: ", "cyan")).strip().lower()
            if again != "y":
                break
    elif config.mode == "model_vs_random":
        await watch_model_vs_random(
            renderer, model_token_completer, config.game_name, config.num_games
        )


if __name__ == "__main__":
    config = chz.entrypoint(PlayConfig)
    asyncio.run(main(config))
