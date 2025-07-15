"""
Command-line interface for RL general training.

This provides a simple entry point for common RL training scenarios.
For more advanced use cases, use train.py directly.
"""

import asyncio
import logging

import chz
from tinker_cookbook import model_info, renderers
from tinker_cookbook.preference import preference_datasets
from tinker_cookbook.rl import (
    arithmetic_env,
    math_env,
    polaris_math_env,
    preference_envs,
    textarena_envs,
    twenty_questions_env,
)
from tinker_cookbook.rl.train import Config, main
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Simple command-line configuration for RL training."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.2-1B"
    renderer_name: str | None = None

    # Environment configuration
    env: str = "arithmetic"  # Options: arithmetic, guess_the_number

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 5

    # Logging configuration
    log_relpath: str = "tmp/rl-general"
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Service configuration
    base_url: str | None = None


def get_dataset_builder(
    env: str, batch_size: int, model_name: str, renderer_name: str, group_size: int
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
    elif env == "math":
        return math_env.MathDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
        )
    elif env == "polaris_math":
        return polaris_math_env.PolarisDatasetBuilder(
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
        common_config = ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            max_length=8192,
            batch_size=batch_size,
        )
        comparison_dataset_builder = preference_datasets.HHHBuilder(
            common_config=common_config, swap=False
        )
        return preference_envs.PairwisePreferenceRLDatasetBuilder(
            batch_size=batch_size,
            comparison_dataset_builder=comparison_dataset_builder,
            model_path="tinker://40e97ac0-99ea-4a84-a8c8-3b319db7cd2b/sampler_weights/checkpoint_final",
            # ^^^ 8b instruct trained on anthropic-hhh dataset
            group_size=group_size,
        )
    elif env == "twenty_questions":
        return twenty_questions_env.TwentyQuestionsDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
        )
    else:
        raise ValueError(f"Unknown environment: {env}")


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get tokenizer for stop sequences
    tokenizer = get_tokenizer(cli_config.model_name)
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    renderers.get_renderer(renderer_name, tokenizer)

    # Create full config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            env=cli_config.env,
            batch_size=cli_config.groups_per_batch,
            model_name=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
        ),
        model_name=cli_config.model_name,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name,
        log_relpath=cli_config.log_relpath,
        base_url=cli_config.base_url,
    )

    # Run training
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
