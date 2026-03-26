"""
Full-scale SFT training for Nemotron-Cascade-2 replication.

Uses combined math + science + instruction_following + safety data.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import chz

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.recipes.nemotron_cascade.sft_datasets import NemotronCascadeSFTFromFileBuilder
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_jsonl_files(input_files: list[str], output_file: str, max_per_file: int | None = None):
    """Combine multiple JSONL files into one, optionally limiting per file."""
    total = 0
    with open(output_file, "w") as out:
        for f in input_files:
            count = 0
            with open(f) as inp:
                for line in inp:
                    out.write(line)
                    count += 1
                    total += 1
                    if max_per_file and count >= max_per_file:
                        break
            logger.info(f"  Added {count} examples from {os.path.basename(f)}")
    logger.info(f"Combined dataset: {total} total examples -> {output_file}")
    return total


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b:peft:131072")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--data-dir", type=str, default=os.path.expanduser("~/data/nemotron-cascade-2"))
    parser.add_argument("--combined-file", type=str, default=None, help="Pre-combined JSONL file to use")
    args = parser.parse_args()

    # Combine data files or use pre-combined
    if args.combined_file and os.path.exists(args.combined_file):
        data_path = args.combined_file
        logger.info(f"Using pre-combined data: {data_path}")
    else:
        data_files = []
        for name in ["sft_math_100k.jsonl", "sft_science_50k.jsonl",
                      "sft_instruction_following_10k.jsonl", "sft_safety_all.jsonl"]:
            fpath = os.path.join(args.data_dir, name)
            if os.path.exists(fpath):
                data_files.append(fpath)
            else:
                logger.warning(f"Data file not found: {fpath}")

        data_path = os.path.join(args.data_dir, "sft_combined.jsonl")
        combine_jsonl_files(data_files, data_path)

    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=args.model, explicit_renderer_name=None,
        load_checkpoint_path=None, base_url=None,
    )

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    dataset_builder = NemotronCascadeSFTFromFileBuilder(
        common_config=common_config,
        file_path=data_path,
        test_size=512,
        seed=42,
    )

    model_short = args.model.replace("/", "-").replace(":", "-")
    if args.log_path:
        log_path = args.log_path
    else:
        log_path = f"/tmp/tinker-examples/nemotron_cascade_full_sft/{model_short}_lr{args.lr}"

    config = train.Config(
        log_path=log_path,
        model_name=args.model,
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        evaluator_builders=[],
        learning_rate=args.lr,
        lr_schedule="cosine",
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-8,
        max_steps=args.max_steps,
    )

    logger.info(f"Starting full SFT: model={args.model}, lr={args.lr}, log_path={log_path}")
    asyncio.run(train.main(config))
    logger.info("Full SFT completed!")


if __name__ == "__main__":
    main()
