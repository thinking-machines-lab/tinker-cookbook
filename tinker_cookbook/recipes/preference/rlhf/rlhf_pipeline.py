import asyncio
import logging
import os

import chz
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.preference.comparison_policy_evaluator import ComparisonEvaluator
from tinker_cookbook.preference.preference_datasets import ChatDatasetBuilderFromComparisons
from tinker_cookbook.preference.types import PreferenceModelBuilderFromChatRenderer
from tinker_cookbook.recipes.chat_sl.chat_datasets import NoRobotsBuilder
from tinker_cookbook.recipes.preference.datasets import HHHComparisonBuilder
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.rl import preference_envs, train
from tinker_cookbook.supervised import train as supervised_train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    base_model: str = "meta-llama/Llama-3.2-3B"
    short_name: str = "llama3b"
    run_sft: bool = True
    run_rm: bool = True
    run_rl: bool = True
    wandb_project: str | None = None
    wandb_name: str | None = "rlhf"
    lora_rank: int = 64
    max_length: int = 16384
    batch_size: int = 256

    sft_learning_rate: float = 2e-4
    rm_learning_rate: float = 3e-4
    rl_learning_rate: float = 1e-5
    rl_max_tokens: int = 1024
    rl_group_size: int = 4

    save_every: int = 100
    eval_every: int = 20

    # Logtree configuration - number of groups to log per iteration (0 = disable)
    num_groups_to_log: int = 4


def sft_stage(
    log_path: str,
    base_model: str,
    wandb_project: str | None,
    wandb_name: str | None,
    lora_rank: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    save_every: int,
    eval_every: int,
):
    """
    Train base policy on NoRobots dataset
    """
    # Create renderer for the model
    renderer_name = model_info.get_recommended_renderer_name(base_model)

    # Create common config for the dataset builder
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=base_model,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    # Use NoRobots dataset for SFT
    dataset_builder = NoRobotsBuilder(common_config=common_config)

    # Create training config
    config = supervised_train.Config(
        log_path=log_path,
        model_name=base_model,
        dataset_builder=dataset_builder,
        evaluator_builders=[],  # Could add evaluators here
        num_epochs=1,
        learning_rate=learning_rate,
        lr_schedule="linear",
        save_every=save_every,
        eval_every=eval_every,
        lora_rank=lora_rank,
        wandb_project=wandb_project,
        wandb_name=f"{wandb_name}-sft",
    )

    # Run training
    asyncio.run(supervised_train.main(config))


def train_rm(
    log_path: str,
    base_model: str,
    wandb_project: str | None,
    wandb_name: str | None,
    lora_rank: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    save_every: int,
    eval_every: int,
):
    """Train reward model using Anthropic HHH preference comparisons."""
    # Use HHH comparison builder for Anthropic data
    comparison_builder = HHHComparisonBuilder()

    # Get renderer name for the model
    renderer_name = model_info.get_recommended_renderer_name(base_model)

    # Create common config for the dataset builder
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=base_model,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    # Create the dataset builder that wraps comparisons with rendering
    dataset_builder = ChatDatasetBuilderFromComparisons(
        common_config=common_config, comparison_builder=comparison_builder
    )

    # Create training config
    config = supervised_train.Config(
        log_path=log_path,
        model_name=base_model,
        dataset_builder=dataset_builder,
        evaluator_builders=[],  # Could add evaluators here
        num_epochs=1,
        learning_rate=learning_rate,
        lr_schedule="linear",
        save_every=save_every,
        eval_every=eval_every,
        lora_rank=lora_rank,
        wandb_project=wandb_project,
        wandb_name=f"{wandb_name}-rm",
    )

    # Run training
    asyncio.run(supervised_train.main(config))


async def train_rl(
    log_path: str,
    sft_log_path: str,
    rm_log_path: str,
    base_model: str,
    wandb_project: str | None,
    wandb_name: str | None,
    lora_rank: int,
    group_size: int,
    batch_size: int,
    learning_rate: float,
    max_tokens: int,
    save_every: int,
    eval_every: int,
    num_groups_to_log: int = 4,
):
    """Train policy using RL with prompts from Anthropic HHH data."""
    # Get checkpoints from previous stages
    sft_checkpoint_dict = checkpoint_utils.get_last_checkpoint(sft_log_path)
    rm_checkpoint_dict = checkpoint_utils.get_last_checkpoint(rm_log_path)

    if sft_checkpoint_dict is None:
        raise ValueError(f"No SFT checkpoint found in {sft_log_path}")
    if rm_checkpoint_dict is None:
        raise ValueError(f"No RM checkpoint found in {rm_log_path}")

    sft_checkpoint = sft_checkpoint_dict["state_path"]
    rm_weights_path = rm_checkpoint_dict["sampler_path"]

    # Use HHH comparison builder for prompts
    comparison_builder = HHHComparisonBuilder()
    renderer_name = model_info.get_recommended_renderer_name(base_model)

    preference_model_builder = PreferenceModelBuilderFromChatRenderer(
        renderer_name=renderer_name,
        model_name=base_model,
        rm_weights_path=rm_weights_path,
    )

    rl_dataset_builder = preference_envs.PairwisePreferenceRLDatasetBuilder(
        comparison_builder=comparison_builder,
        policy_renderer_name=renderer_name,
        policy_model_name=base_model,
        preference_model_builder=preference_model_builder,
        batch_size=batch_size,
        group_size=group_size,
        tournament_pattern=preference_envs.TournamentPattern.ALL_PAIRS_BOTH_WAYS,
    )

    def get_evaluator_builder() -> ComparisonEvaluator:
        comparison_builder_eval = HHHComparisonBuilder(test_size=256)
        _, test_dataset = comparison_builder_eval.get_train_and_test_datasets()
        assert test_dataset is not None
        test_labeled_comparisons = [
            comparison_builder_eval.example_to_labeled_comparison(example)  # type: ignore
            for example in test_dataset
        ]
        test_comparisons = [lc.comparison for lc in test_labeled_comparisons if lc is not None]
        return ComparisonEvaluator(
            preference_model_builder=preference_model_builder,
            comparisons=test_comparisons,
            renderer_name=renderer_name,
            model_name_for_tokenizer=base_model,
        )

    config = train.Config(
        model_name=base_model,
        dataset_builder=rl_dataset_builder,
        load_checkpoint_path=sft_checkpoint,
        learning_rate=learning_rate,
        max_tokens=max_tokens,
        log_path=log_path,
        evaluator_builders=[get_evaluator_builder],
        wandb_project=wandb_project,
        wandb_name=f"{wandb_name}-rl",
        lora_rank=lora_rank,
        save_every=save_every,
        eval_every=eval_every,
        num_groups_to_log=num_groups_to_log,
    )
    await train.main(config)


def cli_main(cli_config: CLIConfig):
    log_path_root = os.path.expanduser(f"~/experiments/rlhf-{cli_config.short_name}")
    sft_log_path = os.path.join(log_path_root, "sft")
    rm_log_path = os.path.join(log_path_root, "rm")
    rl_log_path = os.path.join(log_path_root, "rl")
    if cli_config.run_sft:
        sft_stage(
            sft_log_path,
            cli_config.base_model,
            cli_config.wandb_project,
            cli_config.wandb_name,
            cli_config.lora_rank,
            cli_config.batch_size,
            cli_config.sft_learning_rate,
            cli_config.max_length,
            cli_config.save_every,
            cli_config.eval_every,
        )
    if cli_config.run_rm:
        train_rm(
            rm_log_path,
            cli_config.base_model,
            cli_config.wandb_project,
            cli_config.wandb_name,
            cli_config.lora_rank,
            cli_config.batch_size,
            cli_config.rm_learning_rate,
            cli_config.max_length,
            cli_config.save_every,
            cli_config.eval_every,
        )
    if cli_config.run_rl:
        asyncio.run(
            train_rl(
                rl_log_path,
                sft_log_path,
                rm_log_path,
                cli_config.base_model,
                cli_config.wandb_project,
                cli_config.wandb_name,
                cli_config.lora_rank,
                cli_config.rl_group_size,
                cli_config.batch_size,
                cli_config.rl_learning_rate,
                cli_config.rl_max_tokens,
                cli_config.save_every,
                cli_config.eval_every,
                cli_config.num_groups_to_log,
            )
        )


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)
