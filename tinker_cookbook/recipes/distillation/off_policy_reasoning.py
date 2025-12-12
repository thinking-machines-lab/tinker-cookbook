"""
Supervised fine-tuning for reasoning tasks using OpenThoughts3.

This script implements standard supervised learning on the OpenThoughts3 dataset,
which contains reasoning traces with chain-of-thought style responses.

Example usage:
    python -m tinker_cookbook.recipes.distillation.off_policy_reasoning \
        model_name=Qwen/Qwen3-8B-Base \
        learning_rate=1e-4 \
        batch_size=128 \
        lora_rank=128 \
        wandb_project=cookbook_distillation
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import cast

import chz
import datasets
import tinker
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import Message, TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    StreamingSupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)
from tinker_cookbook.utils.lr_scheduling import LRSchedule

logger = logging.getLogger(__name__)


@chz.chz
class OpenThoughts3Builder(ChatDatasetBuilder):
    """Builder for OpenThoughts3 dataset with streaming support."""

    buffer_size: int = 128 * 3000  # Buffer for shuffle
    max_prompts: int = 128 * 3000  # Maximum number of prompts to train on

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load streaming dataset
        ds = datasets.load_dataset(
            "open-thoughts/OpenThoughts3-1.2M", split="train", streaming=True
        )
        ds = cast(datasets.IterableDataset, ds)

        # Use train_on_what from common_config if provided, otherwise default to ALL_ASSISTANT_MESSAGES
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        def map_fn(row: dict) -> tinker.Datum:
            # Convert OpenThoughts3 format (from/value) to standard format (role/content)
            conversations = row.get("conversations", [])
            messages: list[Message] = [
                {
                    "role": "user" if msg["from"] == "human" else "assistant",
                    "content": msg["value"],
                }
                for msg in conversations
            ]
            return conversation_to_datum(
                messages, self.renderer, self.common_config.max_length, train_on_what
            )

        train_dataset = StreamingSupervisedDatasetFromHFDataset(
            hf_dataset=ds,
            batch_size=self.common_config.batch_size,
            length=self.max_prompts,
            map_fn=map_fn,
            buffer_size=self.buffer_size,
        )

        # No test dataset for OpenThoughts3
        return train_dataset, None


@chz.chz
class CLIConfig:
    """Command-line configuration for SFT on OpenThoughts3."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B-Base"
    lora_rank: int = 128
    renderer_name: str | None = "qwen3"
    load_checkpoint_path: str | None = None

    # Training hyperparameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1
    max_length: int = 16384

    # Dataset configuration
    buffer_size: int = 128 * 3000  # Buffer for randomized shuffle
    max_prompts: int = 128 * 3000  # Maximum number of prompts to train on

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Evaluation and checkpointing
    eval_every: int = 50
    save_every: int = 50

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Create log path if not specified
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
        run_name = os.path.basename(log_path)
    else:
        model_name = cli_config.model_name.replace("/", "-")
        run_name = (
            f"sft-openthoughts3-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{cli_config.batch_size}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/distillation/{run_name}")

    # Create wandb name if not specified
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Create dataset builder
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        max_length=cli_config.max_length,
        batch_size=cli_config.batch_size,
        train_on_what=None,  # Use default in OpenThoughts3Builder
    )

    dataset_builder = OpenThoughts3Builder(
        common_config=common_config,
        buffer_size=cli_config.buffer_size,
        max_prompts=cli_config.max_prompts,
    )

    # Create full config
    config = train.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        dataset_builder=dataset_builder,
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        num_epochs=cli_config.num_epochs,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
    )

    # Run training
    asyncio.run(train.main(config))


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)
