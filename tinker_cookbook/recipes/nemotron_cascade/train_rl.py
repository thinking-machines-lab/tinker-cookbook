"""
Nemotron-Cascade-2 RL training CLI.

Supports IF-RL and multi-domain RL stages from the Nemotron-Cascade-2 paper.

Paper hyperparameters:
  IF-RL:
    - Batch size: 128, Rollouts: 16, Temp: 1.0
    - LR: 3e-6, KL coeff: 0
    - Max response: 49K tokens
    - Steps: ~180

  Multi-domain RL:
    - Same hyperparameters, ~70 steps
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import chz
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.nemotron_cascade.if_rl_env import IFRLDatasetBuilder
from tinker_cookbook.recipes.nemotron_cascade.mcqa_rl_env import MCQARLDatasetBuilder
from tinker_cookbook.recipes.nemotron_cascade.structured_output_rl_env import StructuredOutputRLDatasetBuilder
from tinker_cookbook.rl.train import AsyncConfig, Config, StreamMinibatchConfig, main
from tinker_cookbook.rl.types import RLDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """CLI configuration for Nemotron-Cascade-2 RL training."""

    # Model configuration
    model_name: str = "openai/gpt-oss-120b:peft:131072"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    env: str = "if_rl"  # Options: if_rl, mcqa, structured_output

    # Training hyperparameters (paper defaults for IF-RL)
    group_size: int = 16
    groups_per_batch: int = 128
    learning_rate: float = 3e-6
    max_tokens: int = 49152  # 49K tokens
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1
    seed: int = 0

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals and checkpointing
    eval_every: int = 20
    save_every: int = 20

    # Service configuration
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Loss function
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Remove constant reward groups (paper's dynamic filtering)
    remove_constant_reward_groups: bool = True

    max_steps: int | None = None
    max_steps_off_policy: int | None = None
    stream_minibatch_config: StreamMinibatchConfig | None = None


def get_dataset_builder(
    env: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
) -> RLDatasetBuilder:
    if env == "if_rl":
        return IFRLDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    elif env == "mcqa":
        return MCQARLDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    elif env == "structured_output":
        return StructuredOutputRLDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown environment: {env}. Available: if_rl, mcqa, structured_output")


async def cli_main(cli_config: CLIConfig):
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    model_name_short = cli_config.model_name.replace("/", "-").replace(":", "-")
    run_name = (
        f"nemotron-cascade-{cli_config.env}-{model_name_short}-"
        f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
        f"{cli_config.group_size}group-{cli_config.groups_per_batch}batch-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/nemotron_cascade_rl/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            env=cli_config.env,
            batch_size=cli_config.groups_per_batch,
            model_name=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            seed=cli_config.seed,
        ),
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
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
        remove_constant_reward_groups=cli_config.remove_constant_reward_groups,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        stream_minibatch_config=StreamMinibatchConfig(
            groups_per_batch=cli_config.groups_per_batch,
            num_minibatches=cli_config.stream_minibatch_config.num_minibatches,
        )
        if cli_config.stream_minibatch_config is not None
        else None,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        max_steps=cli_config.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
