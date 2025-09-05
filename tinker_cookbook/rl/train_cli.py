"""
DEPRECATED: we recommend using a standalone script like what's in recipes/rl_basic.py instead. We will remove this script in the future.

Command-line interface for RL general training.

This provides a simple entry point for common RL training scenarios.
For more advanced use cases, use train.py directly.
"""

import asyncio
import logging
import os

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.preference import preference_datasets
from tinker_cookbook.rl import (
    arithmetic_env,
    math_env,
    preference_envs,
    textarena_envs,
)
from tinker_cookbook.rl.train import Config, main
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Simple command-line configuration for RL training."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    env: str = "arithmetic"  # Options: arithmetic, guess_the_number

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 5
    kl_penalty_coef: float = 0.0

    # Logging configuration
    log_path: str = chz.field(
        default="/tmp/tinker-examples/rl",
        munger=lambda _, s: os.path.expanduser(s),
    )
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Checkpointing
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def get_dataset_builder(
    env: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
    base_url: str | None = None,
) -> RLDatasetBuilder:
    if env == "arithmetic":
        return arithmetic_env.ArithmeticDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            n_batches=100,
            include_fewshot=True,
            group_size=group_size,
        )
    elif env in ["math", "polaris", "deepmath"]:
        return math_env.get_math_dataset_builder(
            dataset_name=env,
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
        )
    elif env == "guess_the_number":
        return textarena_envs.TextArenaDatasetBuilder(
            batch_size=batch_size,
            builder=textarena_envs.SinglePlayerEnvGroupBuilder(
                game_name="GuessTheNumber-v0",
                tokenizer=get_tokenizer(model_name),
                num_envs=1,
            ),
        )
    elif env == "hhh":
        comparison_builder = preference_datasets.HHHComparisonBuilder(swap=False)
        return preference_envs.PairwisePreferenceRLDatasetBuilder(
            batch_size=batch_size,
            comparison_builder=comparison_builder,
            renderer_name=renderer_name,
            model_name_for_tokenizer=model_name,
            model_path="tinker://40e97ac0-99ea-4a84-a8c8-3b319db7cd2b/sampler_weights/checkpoint_final",
            # ^^^ 8b instruct trained on anthropic-hhh dataset
            group_size=group_size,
            base_url=base_url,
        )
    else:
        raise ValueError(f"Unknown environment: {env}")


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get tokenizer for stop sequences
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    # Create full config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            env=cli_config.env,
            batch_size=cli_config.groups_per_batch,
            model_name=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            base_url=cli_config.base_url,
        ),
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name,
        log_path=cli_config.log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        save_every=cli_config.save_every,
    )

    cli_utils.check_log_dir(
        cli_config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )

    # Run training
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
