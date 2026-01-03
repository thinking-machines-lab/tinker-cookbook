import asyncio
from datetime import datetime
from pathlib import Path

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.if_rl.env import IfBenchDatasetBuilder, RewardType
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    # Model parameters
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    seed: int = 42
    renderer_name: str | None = None
    eval_every: int = 0

    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 32
    group_size: int = 8
    max_tokens: int = 2048
    num_epochs: int = 1

    # IFBench-specific
    reward_type: RewardType = RewardType.FULL_STRICT

    # Resume from checkpoint (use state path, not sampler_weights path)
    load_checkpoint_path: str | None = None

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Build dataset builder
    builder = IfBenchDatasetBuilder(
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=cli_config.model_name,
        reward_type=cli_config.reward_type,
        seed=cli_config.seed,
        max_tokens=cli_config.max_tokens,
        num_epochs=cli_config.num_epochs,
    )

    # Build run name
    model_name_short = cli_config.model_name.lower().replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    epoch_suffix = f"_ep{cli_config.num_epochs}" if cli_config.num_epochs > 1 else ""
    run_name = f"ifbench_{model_name_short}_bs{cli_config.batch_size}_gs{cli_config.group_size}_{cli_config.reward_type.value}_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}{epoch_suffix}_{date_and_time}"

    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/ifbench/{run_name}"

    wandb_name = cli_config.wandb_name or run_name

    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    config = train.Config(
        model_name=cli_config.model_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        load_checkpoint_path=cli_config.load_checkpoint_path,
    )

    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
