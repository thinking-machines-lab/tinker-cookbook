from __future__ import annotations

import asyncio
from datetime import datetime

import chz

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.golf_forecasting.data import GolfForecastDatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Dataset configuration
    dataset_manifest_path: str = "tinker_cookbook/example_data/golf_forecasting/dataset_manifest.json"
    train_jsonl_path: str | None = None
    val_jsonl_path: str | None = None
    include_other_bucket: bool = True

    # Training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 32
    learning_rate: float = 4e-5
    max_tokens: int = 256
    temperature: float = 1.0
    remove_constant_reward_groups: bool = False

    # Logging/eval
    eval_every: int = 20
    save_every: int = 20
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Service configuration
    base_url: str | None = None
    max_steps: int | None = None


async def cli_main(cli_config: CLIConfig) -> None:
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )
    builder = GolfForecastDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        dataset_manifest_path=cli_config.dataset_manifest_path,
        train_jsonl_path=cli_config.train_jsonl_path,
        val_jsonl_path=cli_config.val_jsonl_path,
        include_other_bucket=cli_config.include_other_bucket,
    )

    short_model_name = cli_config.model_name.replace("/", "-")
    run_name = (
        f"golf_forecasting-{short_model_name}-{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cli_config.log_path or f"/tmp/tinker-examples/golf_forecasting/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    config = train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=builder,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        max_tokens=cli_config.max_tokens,
        log_path=log_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        lora_rank=cli_config.lora_rank,
        temperature=cli_config.temperature,
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        base_url=cli_config.base_url,
        max_steps=cli_config.max_steps,
    )
    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
