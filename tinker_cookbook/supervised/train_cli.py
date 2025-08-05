"""
Basic CLI for training with supervised learning. It only supports a few datasets and configuration options; if you want to do something more complicated, please write a new script and call the train.main function directly.
"""

import asyncio

import chz
from tinker_cookbook import model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import chat_datasets, train, train_pipelined
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.misc_utils import lookup_func


@chz.chz
class CLIConfig:
    model_name: str = "meta-llama/Llama-3.2-1B"
    load_checkpoint_path: str | None = None
    dataset: str = "no_robots"
    renderer_name: str | None = None

    # Training parameters
    learning_rate: float = 1e-4
    lr_schedule: str = "linear"
    max_length: int | None = 8192
    batch_size: int = 256
    lora_rank: int = 32
    pipelined: bool = False  # Use faster pipelined

    # Logging parameters
    log_relpath: str = "tmp/supervised"
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Service configuration
    base_url: str | None = None


def get_dataset_builder(
    dataset: str,
    model_name: str,
    renderer_name: str,
    max_length: int | None,
    batch_size: int,
) -> ChatDatasetBuilder:
    # Note that sft/train can work with non-chat datasets, but this CLI only supports chat datasets
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )
    if dataset == "tulu3":
        return chat_datasets.Tulu3Builder(common_config=common_config)
    elif dataset == "tulu3_user_sim":
        return chat_datasets.Tulu3Builder(
            common_config=common_config, train_on_what=TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES
        )
    elif dataset == "no_robots":
        return chat_datasets.NoRobotsBuilder(common_config=common_config)
    elif dataset == "hhh":  # a pairwise comparison dataset
        from tinker_cookbook.preference.preference_datasets import HHHBuilder

        return HHHBuilder(common_config=common_config)
    elif dataset.endswith(".jsonl"):
        # Load conversations from a JSONL file
        return chat_datasets.FromConversationFileBuilder(
            common_config=common_config,
            file_path=dataset,
        )
    else:
        # Can pass in path to callable like
        # tinker_cookbook.supervised.chat_datasets:Tulu3Builder
        # tinker_cookbook.preference.preference_datasets:HHHBuilder
        try:
            builder_func = lookup_func(dataset)
        except ValueError:
            raise ValueError(f"Unknown dataset: {dataset}")
        else:
            return builder_func(common_config=common_config)


def cli_main(cli_config: CLIConfig):
    # build full config
    config = train.Config(
        log_relpath=cli_config.log_relpath,
        model_name=cli_config.model_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        dataset_builder=get_dataset_builder(
            cli_config.dataset,
            cli_config.model_name,
            cli_config.renderer_name
            or model_info.get_recommended_renderer_name(cli_config.model_name),
            cli_config.max_length,
            cli_config.batch_size,
        ),
        evaluator_builders=[],
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name,
        lora_rank=cli_config.lora_rank,
    )
    if cli_config.pipelined:
        asyncio.run(train_pipelined.main(config))
    else:
        train.main(config)


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
