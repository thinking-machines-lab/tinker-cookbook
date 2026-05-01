import asyncio
from datetime import datetime

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.multiplayer_rl.text_arena.env import TwoPlayerTextArenaDatasetBuilder
from tinker_cookbook.recipes.multiplayer_rl.text_arena.tictactoe import OpponentType
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str | None = None
    game_name: str = "TicTacToe-v0"
    test_opponent: OpponentType = "base_model"
    batch_size: int = 512
    num_train_datapoints: int = 40960
    num_test_datapoints: int = 128
    learning_rate: float = 3e-5
    max_tokens: int = 512
    eval_every: int = 5
    save_every: int = 20
    wandb_project: str | None = None
    wandb_name: str | None = None
    log_path: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps: int | None = None
    base_url: str | None = None

    fireworks_base_model_name: str | None = None
    fireworks_deployment_id: str | None = None

def build_config(cli_config: CLIConfig) -> train.Config:
    model_name = cli_config.model_name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{model_name}-{cli_config.game_name.lower().replace('-v0', '')}-{cli_config.batch_size}batch-{cli_config.learning_rate}lr-{date_and_time}"
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/text-arena/{run_name}"
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    dataset_builder = TwoPlayerTextArenaDatasetBuilder(
        batch_size=cli_config.batch_size,
        model_name=model_name,
        game_name=cli_config.game_name,
        num_train_datapoints=cli_config.num_train_datapoints,
        num_test_datapoints=cli_config.num_test_datapoints,
        renderer_name=renderer_name,
        test_opponent=cli_config.test_opponent,
    )

    return train.Config(
        model_name=model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        max_steps=cli_config.max_steps,
        base_url=cli_config.base_url,
        fireworks_base_model_name=cli_config.fireworks_base_model_name,
        fireworks_deployment_id=cli_config.fireworks_deployment_id,
    )


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    config = build_config(cli_config)
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(
        config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    asyncio.run(train.main(config))
