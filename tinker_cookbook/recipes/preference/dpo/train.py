"""
Basic CLI for training with Direct Preference Optimization (DPO). It only supports a few datasets and configuration options; if you want to do something more complicated, please write a new script and call the train_dpo.main function directly.
"""

from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import (
    DPODatasetBuilderFromComparisons,
)
from tinker_cookbook.recipes.preference.datasets import (
    HelpSteer3ComparisonBuilder,
    HHHComparisonBuilder,
    UltraFeedbackComparisonBuilder,
)
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule


@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    dataset: str = "hhh"  # or path like tinker_cookbook.preference.preference_datasets:HHHBuilder
    load_checkpoint_path: str | None = None
    renderer_name: str | None = None

    # Training parameters
    learning_rate: float = 1e-5
    lr_schedule: LRSchedule = "linear"
    dpo_beta: float = 0.1
    max_length: int | None = 8192
    batch_size: int = 256

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Service configuration
    base_url: str | None = None

    # DPO-specific parameters
    reference_model_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def get_dataset_builder(
    dataset: str,
    model_name: str,
    renderer_name: str,
    max_length: int | None,
    batch_size: int,
) -> ChatDatasetBuilder:
    """Get the appropriate dataset builder for DPO training."""
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    if dataset == "hhh":
        return DPODatasetBuilderFromComparisons(
            common_config=common_config, comparison_builder=HHHComparisonBuilder()
        )
    elif dataset == "helpsteer3":
        return DPODatasetBuilderFromComparisons(
            common_config=common_config, comparison_builder=HelpSteer3ComparisonBuilder()
        )
    elif dataset == "ultrafeedback":
        return DPODatasetBuilderFromComparisons(
            common_config=common_config, comparison_builder=UltraFeedbackComparisonBuilder()
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def cli_main(cli_config: CLIConfig):
    """Main CLI function that builds the full config and calls the training function."""
    # Build full config
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_name = cli_config.model_name.replace("/", "-")
    run_name = f"{cli_config.dataset}-{model_name}-{cli_config.learning_rate}lr-{cli_config.batch_size}batch-{date_and_time}"
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/dpo/{run_name}"
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    config = train_dpo.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        dataset_builder=get_dataset_builder(
            cli_config.dataset,
            cli_config.model_name,
            renderer_name,
            cli_config.max_length,
            cli_config.batch_size,
        ),
        load_checkpoint_path=cli_config.load_checkpoint_path,
        evaluator_builders=[],
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        dpo_beta=cli_config.dpo_beta,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        reference_model_name=cli_config.reference_model_name,
    )

    train_dpo.main(config)


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
