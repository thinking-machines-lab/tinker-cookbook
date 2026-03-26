"""
LR sweep for Nemotron-Cascade-2 SFT replication.

Runs multiple LR experiments in parallel to find optimal learning rate
for LoRA fine-tuning on gpt-oss-120b and Qwen3-8B-Base.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import tinker

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.recipes.nemotron_cascade.sft_datasets import NemotronCascadeSFTFromFileBuilder
from tinker_cookbook.utils.lr_scheduling import LRSchedule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_experiment(
    model_name: str,
    learning_rate: float,
    data_path: str,
    log_base: str,
    batch_size: int = 16,
    max_length: int = 8192,
    lora_rank: int = 32,
    max_steps: int | None = None,
):
    """Run a single SFT experiment."""
    model_short = model_name.replace("/", "-").replace(":", "-")
    log_path = f"{log_base}/{model_short}_lr{learning_rate}"

    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=model_name,
        explicit_renderer_name=None,
        load_checkpoint_path=None,
        base_url=None,
    )

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    dataset_builder = NemotronCascadeSFTFromFileBuilder(
        common_config=common_config,
        file_path=data_path,
        test_size=0,
        seed=0,
    )

    config = train.Config(
        log_path=log_path,
        model_name=model_name,
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        evaluator_builders=[],
        learning_rate=learning_rate,
        lr_schedule="cosine",
        num_epochs=1,
        lora_rank=lora_rank,
        save_every=0,  # Don't save checkpoints during sweep
        eval_every=0,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-8,
        max_steps=max_steps,
    )

    logger.info(f"Starting: model={model_name}, lr={learning_rate}, log_path={log_path}")
    asyncio.run(train.main(config))
    logger.info(f"Completed: model={model_name}, lr={learning_rate}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--data", type=str, default=os.path.expanduser("~/data/nemotron-cascade-2/sft_instruction_following_500.jsonl"))
    parser.add_argument("--log-base", type=str, default="/tmp/tinker-examples/nemotron_cascade_lr_sweep")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    run_single_experiment(
        model_name=args.model,
        learning_rate=args.lr,
        data_path=args.data,
        log_base=args.log_base,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_steps=args.max_steps,
    )
