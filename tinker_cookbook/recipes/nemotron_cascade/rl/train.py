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
from tinker_cookbook.recipes.nemotron_cascade.rl.envs.if_rl import IFRLDatasetBuilder
from tinker_cookbook.recipes.nemotron_cascade.rl.envs.longctx import LongContextRLDatasetBuilder
from tinker_cookbook.recipes.nemotron_cascade.rl.envs.mcqa import MCQARLDatasetBuilder
from tinker_cookbook.recipes.nemotron_cascade.rl.envs.rlhf import RLHFDatasetBuilder
from tinker_cookbook.recipes.nemotron_cascade.rl.envs.structured_output import StructuredOutputRLDatasetBuilder
from tinker_cookbook.recipes.nemotron_cascade.rl.envs.workbench import WorkbenchRLDatasetBuilder
from tinker_cookbook.recipes.nemotron_cascade.rl.envs.code_rl import CodeRLDatasetBuilder
from tinker_cookbook.recipes.nemotron_cascade.rl.envs.swe_agentless import SWERLDatasetBuilder
from tinker_cookbook.recipes.nemotron_cascade.rl.envs.swe_agentic import SWEAgenticDatasetBuilder
from tinker_cookbook.rl.rollout_strategy import RetryOnFailure
from tinker_cookbook.rl.interleaved import InterleavedRLDatasetBuilder
from tinker_cookbook.rl.train import AsyncConfig, Config, KLReferenceConfig, StreamMinibatchConfig, main
from tinker_cookbook.rl.types import RLDatasetBuilder

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """CLI configuration for Nemotron-Cascade-2 RL training."""

    # Model configuration
    model_name: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None
    # TODO: warm_start support requires core library change to add
    # load_checkpoint_with_optimizer_path to rl.train.Config.
    # For now, all RL stages use cold start (fresh optimizer).
    # Paper likely uses warm start between RL stages.

    # Environment configuration
    env: str = "if_rl"  # Options: if_rl, longctx_rl, mcqa, structured_output, workbench, swe_rl, swe_agentic, rlhf, code_rl

    # Training hyperparameters (paper defaults for IF-RL)
    group_size: int = 16
    groups_per_batch: int = 128
    learning_rate: float = 3e-5
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

    # Model context window size. Dynamically caps max_tokens per-request.
    # Needed for Cascade SWE prompts (~24K tokens) where prompt + max_tokens
    # can exceed the model's 65K context limit.
    context_window: int | None = None

    # SWE environment reward mode (only used when env=swe_rl)
    swe_reward_mode: str = "llm_judge"  # "llm_judge" or "execution"
    # SWE data source (only used when env=swe_rl)
    swe_use_cascade_data: bool = True
    swe_use_r2e_gym: bool = True


def get_dataset_builder(
    env: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
    swe_reward_mode: str = "llm_judge",
    swe_use_cascade_data: bool = True,
    swe_use_r2e_gym: bool = True,
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
    elif env == "workbench":
        return WorkbenchRLDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    elif env == "swe_rl":
        return SWERLDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=min(group_size, 4),  # Cap at 4 (each test is expensive)
            reward_mode=swe_reward_mode,  # type: ignore[arg-type]
            use_cascade_swe_data=swe_use_cascade_data,
            use_r2e_gym=swe_use_r2e_gym,
            seed=seed,
        )
    elif env == "rlhf":
        return RLHFDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    elif env == "code_rl":
        return CodeRLDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    elif env == "longctx_rl":
        return LongContextRLDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    elif env == "swe_agentic":
        return SWEAgenticDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    elif env == "multi_domain":
        # Paper Table 8: MCQA 55%, Workbench 30%, Structured Output 15%
        mcqa = MCQARLDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
        workbench = WorkbenchRLDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
        structured = StructuredOutputRLDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
        return InterleavedRLDatasetBuilder(
            sources=[mcqa, workbench, structured],
            weights=[0.55, 0.30, 0.15],
            groups_per_batch=batch_size,
            total_batches=None,  # Use max_steps from Config
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown environment: {env}. "
            "Available: if_rl, longctx_rl, mcqa, structured_output, workbench, "
            "swe_rl, swe_agentic, rlhf, code_rl, multi_domain"
        )


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
            swe_reward_mode=cli_config.swe_reward_mode,
            swe_use_cascade_data=cli_config.swe_use_cascade_data,
            swe_use_r2e_gym=cli_config.swe_use_r2e_gym,
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
        kl_reference_config=(
            KLReferenceConfig(base_model=cli_config.model_name.split(":peft:")[0])
            if cli_config.kl_penalty_coef > 0
            else None
        ),
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
        context_window=cli_config.context_window,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
