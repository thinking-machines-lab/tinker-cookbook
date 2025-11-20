import asyncio
import logging
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.code_rl.code_env import DeepcoderDatasetBuilder
from tinker_cookbook.rl.train import AsyncConfig, Config, main

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for DeepCoder RL training."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Data / environment configuration
    seed: int = 0

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 5
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Logging / eval / checkpoints
    log_dir: str | None = None
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False
    eval_every: int = 20
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Async rollout configuration
    max_steps_off_policy: int | None = None


async def cli_main(cli_config: CLIConfig) -> None:
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    model_tag = cli_config.model_name.replace("/", "-")
    run_name = (
        f"deepcoder-{model_tag}-{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-seed{cli_config.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/code_rl/{run_name}"

    wandb_name = cli_config.wandb_name or run_name

    dataset_builder = DeepcoderDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        seed=cli_config.seed,
    )

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
