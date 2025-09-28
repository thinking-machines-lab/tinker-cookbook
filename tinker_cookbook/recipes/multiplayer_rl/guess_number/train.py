import asyncio
from time import time

from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.multiplayer_rl.guess_number.env import GuessNumberDatasetBuilder
from tinker_cookbook.rl import train


def build_config() -> train.Config:
    model_name = "Qwen/Qwen3-8B"
    renderer_name = "qwen3_disable_thinking"

    dataset_builder = GuessNumberDatasetBuilder(
        batch_size=32,
        model_name=model_name,
        renderer_name=renderer_name,
        train_group_size=8,
    )

    return train.Config(
        model_name=model_name,
        log_path=f"/tmp/tinker-examples/guess-number/{int(time())}",
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
