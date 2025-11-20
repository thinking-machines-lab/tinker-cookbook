"""
On-policy distillation for reasoning and chat tasks.

This script implements on-policy distillation where a student model learns from
a teacher model by minimizing KL divergence. No correctness or format rewards
are used - only KL penalty provides supervision.

Example usage:
    # For reasoning tasks (DeepMath)
    python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
        model_name=Qwen/Qwen3-8B-Base \
        dataset=deepmath \
        learning_rate=1e-4 \
        groups_per_batch=1024 \
        lora_rank=128 \
        wandb_project=cookbook_distillation

    # For chat tasks (Tulu3)
    python -m tinker_cookbook.recipes.distillation.on_policy_distillation \
        model_name=Qwen/Qwen3-8B-Base \
        dataset=tulu3 \
        learning_rate=1e-4 \
        groups_per_batch=1024 \
        lora_rank=128 \
        wandb_project=cookbook_distillation
"""

import asyncio
import logging
import os
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDatasetBuilder,
    TeacherConfig,
)

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for on-policy distillation."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B-Base"  # Student model
    lora_rank: int = 128
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None  # Student checkpoint

    # Teacher configuration
    teacher_model: str = "Qwen/Qwen3-8B"
    teacher_checkpoint: str | None = None

    # Dataset configuration
    dataset: str = "deepmath"  # Options: deepmath, tulu3

    # Training hyperparameters
    group_size: int = 4  # Number of rollouts per prompt
    groups_per_batch: int = 1024
    learning_rate: float = 1e-4
    max_tokens: int = 4096
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Optimizer configuration
    num_substeps: int = 1
    loss_fn: str = "importance_sampling"

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evaluation and checkpointing
    eval_every: int = 20
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Create log path if not specified
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_name = cli_config.model_name.replace("/", "-")
        run_name = (
            f"distill-{cli_config.dataset}-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/distillation/{run_name}")

    # Create wandb name if not specified
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = os.path.basename(log_path)

    # Create dataset builder
    dataset_builder = PromptOnlyDatasetBuilder(
        dataset_name=cli_config.dataset,
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
    )

    # Create teacher config
    teacher_config = TeacherConfig(
        base_model=cli_config.teacher_model,
        load_checkpoint_path=cli_config.teacher_checkpoint,
    )

    # Create distillation dataset config
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=cli_config.groups_per_batch,
    )

    # Create full config
    config = train_on_policy.Config(
        learning_rate=cli_config.learning_rate,
        dataset_configs=[dataset_config],
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,  # type: ignore
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await train_on_policy.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
