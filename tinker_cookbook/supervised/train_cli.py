"""
DEPRECATED: we recommend using a standalone script like what's in recipes/sl_basic.py instead. We will remove this script in the future.

Basic CLI for training with supervised learning. It only supports a few datasets and configuration options; if you want to do something more complicated, please write a new script and call the train.main function directly.
"""

import asyncio
import os

import chz
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.evaluators import EvaluatorBuilder
from tinker_cookbook.supervised import chat_datasets, train, train_pipelined
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig


@chz.chz
class CLIConfig:
    # Required parameters
    log_path: str = chz.field(
        default="/tmp/tinker-examples/supervised",
        munger=lambda _, s: os.path.expanduser(s),
    )
    model_name: str = "meta-llama/Llama-3.1-8B"
    load_checkpoint_path: str | None = None
    dataset: str = "no_robots"

    # Training parameters
    learning_rate: float = 1e-4
    lr_schedule: str = "linear"
    num_epochs: int = 1

    # Model parameters
    lora_rank: int = 32

    # Infrastructure parameters
    base_url: str | None = None
    pipelined: bool = False  # Use faster pipelined

    # Checkpointing and evaluation
    save_every: int = 20
    eval_every: int = 20
    infrequent_eval_every: int = 100
    inline_evals: str | None = None

    # Dataset-specific parameters
    renderer_name: str | None = None
    train_on_what: renderers.TrainOnWhat | None = None  # TrainOnWhat option
    max_length: int | None = 16384
    batch_size: int = 256

    # Logging parameters
    wandb_project: str | None = None
    wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def get_dataset_builder(
    dataset: str,
    model_name: str,
    renderer_name: str,
    max_length: int | None,
    batch_size: int,
    train_on_what: renderers.TrainOnWhat | None = None,
) -> ChatDatasetBuilder:
    # Note that sft/train can work with non-chat datasets, but this CLI only supports chat datasets
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
        train_on_what=train_on_what,
    )

    if dataset == "tulu3":
        return chat_datasets.Tulu3Builder(common_config=common_config)
    elif dataset == "no_robots":
        return chat_datasets.NoRobotsBuilder(common_config=common_config)
    elif dataset == "hhh":  # a pairwise comparison dataset
        from tinker_cookbook.preference.preference_datasets import (
            ChatDatasetBuilderFromComparisons,
            HHHComparisonBuilder,
        )

        return ChatDatasetBuilderFromComparisons(
            common_config=common_config, comparison_builder=HHHComparisonBuilder()
        )
    elif dataset.endswith(".jsonl"):
        # Load conversations from a JSONL file
        return chat_datasets.FromConversationFileBuilder(
            common_config=common_config,
            file_path=dataset,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_infrequent_evaluator_builders(
    inline_evals: str | None, renderer_name: str, model_name: str
) -> list[EvaluatorBuilder]:
    if inline_evals is None:
        return []
    elif inline_evals == "inspect":
        from tinker_cookbook.inspect_evaluators import InspectEvaluatorBuilder

        builder = InspectEvaluatorBuilder(
            tasks=["inspect_evals/gsm8k", "inspect_evals/ifeval"],
            renderer_name=renderer_name,
            model_name=model_name,
            temperature=0.6,
            max_tokens=1000,
            limit=None,
            debug_errors=True,
            log_dir=None,
            max_connections=512,
            log_level="INFO",
        )
        return [builder]
    else:
        raise ValueError(f"Unknown inline evaluator: {inline_evals}")


def cli_main(cli_config: CLIConfig):
    # build full config
    cli_utils.check_log_dir(
        cli_config.log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    config = train.Config(
        log_path=cli_config.log_path,
        model_name=cli_config.model_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        dataset_builder=get_dataset_builder(
            cli_config.dataset,
            cli_config.model_name,
            renderer_name,
            cli_config.max_length,
            cli_config.batch_size,
            cli_config.train_on_what,
        ),
        evaluator_builders=[],
        infrequent_evaluator_builders=get_infrequent_evaluator_builders(
            cli_config.inline_evals,
            renderer_name,
            cli_config.model_name,
        ),
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        num_epochs=cli_config.num_epochs,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name,
        lora_rank=cli_config.lora_rank,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
        infrequent_eval_every=cli_config.infrequent_eval_every,
    )
    if cli_config.pipelined:
        asyncio.run(train_pipelined.main(config))
    else:
        train.main(config)


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
