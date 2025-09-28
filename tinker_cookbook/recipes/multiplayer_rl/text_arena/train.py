import asyncio
from time import time

from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.multiplayer_rl.text_arena.env import TwoPlayerTextArenaDatasetBuilder
from tinker_cookbook.rl import train


def build_config() -> train.Config:
    model_name = "Qwen/Qwen3-8B"
    renderer_name = "qwen3_disable_thinking"

    dataset_builder = TwoPlayerTextArenaDatasetBuilder(
        batch_size=512,
        model_name=model_name,
        game_name="TicTacToe-v0",
        num_train_datapoints=131072,
        num_test_datapoints=128,
        renderer_name=renderer_name,
    )

    return train.Config(
        model_name=model_name,
        log_path=f"/tmp/tinker-examples/text-arena-tic-tac-toe/{int(time())}",
        dataset_builder=dataset_builder,
        learning_rate=3e-5,
        max_tokens=64,
        eval_every=5,
        compute_post_kl=True,
    )


def main():
    config = build_config()
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
