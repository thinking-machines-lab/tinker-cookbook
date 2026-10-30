"""GEM ❤️ Tinker.

A script for training agents on GEM environments with Tinker-Cookbook.
"""

import asyncio
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import AsyncConfig, Config, main

from .tinker_cookbook_adapter import GemDatasetBuilder


@chz.chz
class CLIConfig:
    # Model
    model_name: str = "meta-llama/Llama-3.2-1B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # GEM env
    env_id: str = "game:GuessTheNumber-v0"
    env_kwargs_json: str | None = None  # e.g., '{"max_turns": 4}'

    # Training
    prompt_type: str = "general"
    group_size: int = 4
    groups_per_batch: int = 64
    n_batches: int = 200
    learning_rate: float = 1e-5
    max_tokens: int = 256
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False
    eval_every: int = 0
    save_every: int = 25

    # Service
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps_off_policy: int | None = None


async def cli_main(cli_config: CLIConfig):
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_name_sanitized = cli_config.model_name.replace("/", "-")
    run_name = (
        f"gem-{cli_config.env_id.replace(':', '_')}-{model_name_sanitized}-r{cli_config.lora_rank}-"
        f"lr{cli_config.learning_rate}-g{cli_config.group_size}-b{cli_config.groups_per_batch}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cli_config.log_path or f"./outputs-tinker/gem/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    dataset_builder = GemDatasetBuilder(
        env_id=cli_config.env_id,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        prompt_type=cli_config.prompt_type,
        group_size=cli_config.group_size,
        groups_per_batch=cli_config.groups_per_batch,
        n_batches=cli_config.n_batches,
        env_kwargs_json=cli_config.env_kwargs_json,
    )

    cfg = Config(
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
        async_config=(
            AsyncConfig(
                max_steps_off_policy=cli_config.max_steps_off_policy,
                groups_per_batch=cli_config.groups_per_batch,
            )
            if cli_config.max_steps_off_policy is not None
            else None
        ),
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(cfg)


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cfg))
