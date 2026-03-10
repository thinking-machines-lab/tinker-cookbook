"""
CLI for SDPO training. Supports math environments and SciKnowEval (the paper's
primary benchmark). For more complex setups, construct an sdpo.train.Config
directly and call sdpo.train.main().
"""

import asyncio
from datetime import datetime

import chz

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.math_rl.math_env import get_math_dataset_builder
from tinker_cookbook.recipes.sdpo.sciknoweval_env import (
    SciKnowEvalDatasetBuilder,
    SciKnowEvalDomain,
)
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.sdpo.train import Config, main


@chz.chz
class CLIConfig:
    """Command-line configuration for SDPO training."""

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment: math datasets or SciKnowEval (paper's benchmark)
    env: str = "math"  # math, gsm8k, polaris, deepmath, sciknoweval
    sciknoweval_domain: SciKnowEvalDomain = "chemistry"
    seed: int = 0

    # Training
    group_size: int = 8
    groups_per_batch: int = 64
    learning_rate: float = 1e-5
    max_tokens: int = 2048
    temperature: float = 1.0

    # SDPO-specific
    success_reward_threshold: float = 0.5
    reprompt_suffix: str = "Correctly solve the original question."
    dont_reprompt_on_self_success: bool = True
    remove_thinking_from_demonstration: bool = True

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Checkpointing / eval
    eval_every: int = 10
    save_every: int = 10

    # Service
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _get_dataset_builder(cli_config: CLIConfig, renderer_name: str) -> RLDatasetBuilder:
    if cli_config.env == "sciknoweval":
        return SciKnowEvalDatasetBuilder(
            batch_size=cli_config.groups_per_batch,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            domain=cli_config.sciknoweval_domain,
            seed=cli_config.seed,
        )
    else:
        return get_math_dataset_builder(
            dataset_name=cli_config.env,
            batch_size=cli_config.groups_per_batch,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            seed=cli_config.seed,
        )


async def cli_main(cli_config: CLIConfig):
    """Build full config from CLI args and run training."""
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    model_slug = cli_config.model_name.replace("/", "-")
    env_label = (
        f"sciknoweval-{cli_config.sciknoweval_domain}"
        if cli_config.env == "sciknoweval"
        else cli_config.env
    )
    run_name = (
        f"sdpo-{env_label}-{model_slug}-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-seed{cli_config.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cli_config.log_path or f"/tmp/tinker-examples/sdpo/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    config = Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        dataset_builder=_get_dataset_builder(cli_config, renderer_name),
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        lora_rank=cli_config.lora_rank,
        success_reward_threshold=cli_config.success_reward_threshold,
        reprompt_suffix=cli_config.reprompt_suffix,
        dont_reprompt_on_self_success=cli_config.dont_reprompt_on_self_success,
        remove_thinking_from_demonstration=cli_config.remove_thinking_from_demonstration,
        renderer_name=renderer_name,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
    )

    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
