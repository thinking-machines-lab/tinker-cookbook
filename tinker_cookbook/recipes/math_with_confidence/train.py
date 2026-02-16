import asyncio
from datetime import datetime
from typing import Any, Literal

import chz
from tinker.types import LossFnType

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_with_confidence.env import (
    BrierRewardMode,
    DEFAULT_CONSISTENCY_GRADER_MODEL,
    get_dataset_builder,
)
from tinker_cookbook.rl.train import AsyncConfig, Config, main


@chz.chz
class CLIConfig:
    # Model configuration
    model_name: str = "Qwen/Qwen3-30B-A3B-Base"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    dataset_name: Literal["math", "polaris", "deepmath", "gsm8k"] = "math"
    seed: int = 0
    alpha: float = 0.5
    consistency_coef: float = 0.2
    brier_reward_mode: BrierRewardMode = "one_minus_squared_error"
    include_fewshot: bool = True
    consistency_grader_model_name: str = DEFAULT_CONSISTENCY_GRADER_MODEL
    consistency_grader_max_tokens: int = 256

    # Training hyperparameters
    group_size: int = 32
    groups_per_batch: int = 64
    learning_rate: float = 2e-5
    max_tokens: int = 2048
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals/checkpoints
    eval_every: int = 4
    save_every: int = 4

    # Service configuration
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps_off_policy: int | None = None

    # Loss function
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None


async def cli_main(cli_config: CLIConfig):
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    model_name_slug = cli_config.model_name.replace("/", "-")
    default_cfg = CLIConfig()
    run_parts = [f"{cli_config.dataset_name}-{model_name_slug}"]
    if cli_config.alpha != default_cfg.alpha:
        run_parts.append(f"alpha{cli_config.alpha:g}")
    if cli_config.consistency_coef != default_cfg.consistency_coef:
        run_parts.append(f"consis{cli_config.consistency_coef:g}")
    if cli_config.brier_reward_mode != default_cfg.brier_reward_mode:
        run_parts.append(f"brier-{cli_config.brier_reward_mode}")
    if cli_config.group_size != default_cfg.group_size:
        run_parts.append(f"{cli_config.group_size}group")
    if cli_config.groups_per_batch != default_cfg.groups_per_batch:
        run_parts.append(f"{cli_config.groups_per_batch}batch")
    if cli_config.learning_rate != default_cfg.learning_rate:
        run_parts.append(f"{cli_config.learning_rate:g}lr")
    if cli_config.seed != default_cfg.seed:
        run_parts.append(f"seed{cli_config.seed}")
    run_parts.append(datetime.now().strftime("%Y-%m-%d-%H-%M"))
    run_name = "-".join(run_parts)
    log_path = cli_config.log_path or f"/tmp/tinker-examples/math_with_confidence/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    cfg = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            dataset_name=cli_config.dataset_name,
            batch_size=cli_config.groups_per_batch,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            alpha=cli_config.alpha,
            consistency_coef=cli_config.consistency_coef,
            brier_reward_mode=cli_config.brier_reward_mode,
            include_fewshot=cli_config.include_fewshot,
            base_url=cli_config.base_url,
            consistency_grader_model_name=cli_config.consistency_grader_model_name,
            consistency_grader_max_tokens=cli_config.consistency_grader_max_tokens,
            seed=cli_config.seed,
        ),
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
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(cfg)


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cfg))
