"""
Multi-teacher on-policy distillation example.

This script demonstrates on-policy distillation with multiple datasets and
different teacher models for each dataset. It uses:
- DeepMath dataset with Qwen3-32B as teacher
- Tulu3 dataset with Qwen3-235B-A22B-Instruct-2507 as teacher
- Qwen3-8B as student model
- qwen3_instruct renderer

Example usage:
    python -m tinker_cookbook.recipes.distillation.on_policy_multi_teacher \
        learning_rate=1e-4 \
        deepmath_groups_per_batch=256 \
        tulu3_groups_per_batch=256 \
        lora_rank=128 \
        wandb_project=cookbook_distillation
"""

import asyncio
import logging
import os
from datetime import datetime

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDatasetBuilder,
    TeacherConfig,
)

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for multi-teacher on-policy distillation."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B"  # Student model
    lora_rank: int = 128
    renderer_name: str = "qwen3_instruct"
    load_checkpoint_path: str | None = None  # Student checkpoint

    # Teacher configurations
    deepmath_teacher_model: str = "Qwen/Qwen3-32B"
    deepmath_teacher_checkpoint: str | None = None
    tulu3_teacher_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    tulu3_teacher_checkpoint: str | None = None

    # Dataset configuration
    deepmath_groups_per_batch: int = 512
    tulu3_groups_per_batch: int = 512

    # Training hyperparameters
    group_size: int = 4  # Number of rollouts per prompt
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

    # Create log path if not specified
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_name = cli_config.model_name.replace("/", "-")
        run_name = (
            f"distill-multi-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"dm{cli_config.deepmath_groups_per_batch}-t3{cli_config.tulu3_groups_per_batch}-"
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/distillation/{run_name}")

    # Create wandb name if not specified
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = os.path.basename(log_path)

    # Create DeepMath dataset builder
    deepmath_builder = PromptOnlyDatasetBuilder(
        dataset_name="deepmath",
        groups_per_batch=cli_config.deepmath_groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=cli_config.renderer_name,
    )

    # Create Tulu3 dataset builder
    tulu3_builder = PromptOnlyDatasetBuilder(
        dataset_name="tulu3",
        groups_per_batch=cli_config.tulu3_groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=cli_config.renderer_name,
    )

    # Create teacher configs
    deepmath_teacher_config = TeacherConfig(
        base_model=cli_config.deepmath_teacher_model,
        load_checkpoint_path=cli_config.deepmath_teacher_checkpoint,
    )

    tulu3_teacher_config = TeacherConfig(
        base_model=cli_config.tulu3_teacher_model,
        load_checkpoint_path=cli_config.tulu3_teacher_checkpoint,
    )

    # Create distillation dataset configs
    deepmath_dataset_config = DistillationDatasetConfig(
        dataset_builder=deepmath_builder,
        teacher_config=deepmath_teacher_config,
        groups_per_batch=cli_config.deepmath_groups_per_batch,
    )

    tulu3_dataset_config = DistillationDatasetConfig(
        dataset_builder=tulu3_builder,
        teacher_config=tulu3_teacher_config,
        groups_per_batch=cli_config.tulu3_groups_per_batch,
    )

    # Create full config with both datasets
    config = train_on_policy.Config(
        learning_rate=cli_config.learning_rate,
        dataset_configs=[deepmath_dataset_config, tulu3_dataset_config],
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
