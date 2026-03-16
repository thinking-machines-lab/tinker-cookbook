"""
Multi-turn on-policy distillation with Harbor sandbox environments.

The student interacts with a harbor sandbox over multiple turns (tool calls),
and the training signal comes from KL divergence against a teacher model.
Environment responses are masked out; only student-generated tokens contribute.

Example usage:
    python -m tinker_cookbook.recipes.distillation.on_policy_distillation_harbor_multi_turn \
        model_name=moonshotai/Kimi-K2-Thinking \
        teacher_model=moonshotai/Kimi-K2-Thinking \
        max_turns=10 \
        group_size=4 \
        groups_per_batch=8 \
        learning_rate=1e-4 \
        lora_rank=8 \
        max_tokens=2048 \
        max_trajectory_tokens=24576 \
        temperature=1.0 \
        kl_penalty_coef=1.0 \
        sandbox_timeout=600 \
        command_timeout=120 \
        save_every=5 \
        eval_every=5 \
        wandb_name=cookbook-multiturn-onpodi
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import chz
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import DistillationDatasetConfig, TeacherConfig
from tinker_cookbook.recipes.distillation.harbor_multiturn import (
    HarborDistillationDatasetBuilder,
    zero_reward,
)
from tinker_cookbook.recipes.harbor_rl.harbor_env import default_sandbox_factory
from tinker_cookbook.recipes.harbor_rl.launch_terminal_bench import load_terminal_bench_tasks

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for multi-turn harbor distillation."""

    # Student model
    model_name: str = "moonshotai/Kimi-K2-Thinking"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Teacher model
    teacher_model: str = "moonshotai/Kimi-K2-Thinking"
    teacher_checkpoint: str | None = None

    # Harbor environment
    max_turns: int = 10
    sandbox_timeout: int = 600
    command_timeout: int = 120
    max_trajectory_tokens: int = 32 * 1024

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 8
    learning_rate: float = 1e-4
    max_tokens: int = 8192
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Optimizer configuration
    num_substeps: int = 1

    # Loss function
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Evaluation and checkpointing
    eval_every: int = 20
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    max_steps: int | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    """Load harbor tasks, build distillation config, and run training."""

    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    # Build log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_tag = cli_config.model_name.replace("/", "-")
        run_name = (
            f"distill-harbor-{model_tag}-{cli_config.lora_rank}rank-"
            f"{cli_config.learning_rate}lr-{cli_config.groups_per_batch}batch-"
            f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = str(Path(f"~/tinker-examples/distillation/{run_name}").expanduser())

    wandb_name = cli_config.wandb_name or Path(log_path).name

    # Load harbor tasks
    tasks = load_terminal_bench_tasks()
    logger.info("Loaded %d harbor tasks", len(tasks))

    # Build dataset
    dataset_builder = HarborDistillationDatasetBuilder(
        tasks=tasks,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        max_turns=cli_config.max_turns,
        sandbox_timeout=cli_config.sandbox_timeout,
        command_timeout=cli_config.command_timeout,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        sandbox_factory=default_sandbox_factory,
        reward_fn=zero_reward,
    )

    teacher_config = TeacherConfig(
        base_model=cli_config.teacher_model,
        load_checkpoint_path=cli_config.teacher_checkpoint,
    )

    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=cli_config.groups_per_batch,
    )

    config = train_on_policy.Config(
        learning_rate=cli_config.learning_rate,
        dataset_configs=[dataset_config],
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        max_steps=cli_config.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    await train_on_policy.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
