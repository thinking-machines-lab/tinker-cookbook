"""
Nemotron-Cascade-2 SFT training CLI.

Replicates the SFT stage of NVIDIA's Nemotron-Cascade-2 paper (arxiv:2603.19220).

Paper hyperparameters (full fine-tuning on Nemotron-3-Nano-30B-A3B):
  - Global batch size: 64
  - Packed sequence length: 256K tokens
  - Max LR: 5e-5, Min LR: 5e-6
  - Warmup: 200 steps
  - Scheduler: Cosine
  - Optimizer: AdamW (beta1=0.9, beta2=0.98)
  - Weight decay: 0.1
  - Training steps: ~33,000 (out of max 40,000)
  - Epochs: ~1.5

For LoRA fine-tuning on gpt-oss-120b, we adapt these hyperparameters:
  - Higher LR for LoRA (~10x)
  - Per-sequence batching instead of packed sequences
"""

import asyncio
import logging
from datetime import datetime

import chz

from tinker_cookbook import checkpoint_utils, cli_utils, renderers
from tinker_cookbook.recipes.nemotron_cascade.sft_datasets import (
    NemotronCascadeSFTBuilder,
    NemotronCascadeSFTFromFileBuilder,
    SFTSubset,
)
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    # Required parameters
    log_path: str | None = None
    model_name: str = "openai/gpt-oss-120b:peft:131072"
    load_checkpoint_path: str | None = None

    # Dataset parameters
    dataset: str = "hf"  # "hf" for HuggingFace, or path to JSONL file
    subsets: tuple[SFTSubset, ...] = ("math",)
    max_examples: int | None = None
    streaming: bool = False

    # Training parameters
    learning_rate: float = 5e-4
    lr_schedule: LRSchedule = "cosine"
    num_epochs: int = 1

    # Model parameters
    lora_rank: int = 32

    # Optimizer parameters (paper uses beta2=0.98)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-8

    # Infrastructure parameters
    base_url: str | None = None

    # Checkpointing and evaluation
    save_every: int = 50
    eval_every: int = 20
    infrequent_eval_every: int = 100

    # Dataset-specific parameters
    renderer_name: str | None = None
    train_on_what: renderers.TrainOnWhat | None = None
    max_length: int | None = 16384
    batch_size: int = 64

    # Logging parameters
    wandb_project: str | None = None
    wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    max_steps: int | None = None

    test_size: int = 1024
    seed: int = 0


def get_dataset_builder(
    cli_config: CLIConfig,
    renderer_name: str,
) -> ChatDatasetBuilder:
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        max_length=cli_config.max_length,
        batch_size=cli_config.batch_size,
        train_on_what=cli_config.train_on_what,
    )

    if cli_config.dataset == "hf":
        return NemotronCascadeSFTBuilder(
            common_config=common_config,
            subsets=cli_config.subsets,
            max_examples=cli_config.max_examples,
            test_size=cli_config.test_size,
            seed=cli_config.seed,
            streaming=cli_config.streaming,
        )
    elif cli_config.dataset.endswith(".jsonl"):
        return NemotronCascadeSFTFromFileBuilder(
            common_config=common_config,
            file_path=cli_config.dataset,
            test_size=cli_config.test_size,
            seed=cli_config.seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {cli_config.dataset}. Use 'hf' or a .jsonl path.")


def cli_main(cli_config: CLIConfig):
    model_name_short = cli_config.model_name.replace("/", "-").replace(":", "-")
    subsets_str = "+".join(cli_config.subsets) if cli_config.dataset == "hf" else "file"
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"nemotron-cascade-sft-{subsets_str}-{model_name_short}-"
        f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
        f"{cli_config.batch_size}batch-{date_and_time}"
    )

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/nemotron_cascade_sft/{run_name}"

    wandb_name = cli_config.wandb_name if cli_config.wandb_name is not None else run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    config = train.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        dataset_builder=get_dataset_builder(cli_config, renderer_name),
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
        infrequent_eval_every=cli_config.infrequent_eval_every,
        max_steps=cli_config.max_steps,
        adam_beta1=cli_config.adam_beta1,
        adam_beta2=cli_config.adam_beta2,
        adam_eps=cli_config.adam_eps,
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    cli_main(cli_config)
