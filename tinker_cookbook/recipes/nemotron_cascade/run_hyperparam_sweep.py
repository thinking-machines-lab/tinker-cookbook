"""
Hyperparameter sweep for SFT v2 on Nemotron-3-Nano.

Creates a proportional 50K sample from the full dataset and tests:
- Learning rates: 1e-4, 3e-4, 5e-4
- LoRA ranks: 32, 64, 128
- Max lengths: 16384, 49152

Run after full data is downloaded.
"""

import argparse
import asyncio
import json
import logging
import os
import random
from pathlib import Path

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.recipes.nemotron_cascade.sft_datasets import NemotronCascadeSFTFromFileBuilder
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_proportional_sample(
    data_dir: str,
    output_file: str,
    total_sample: int = 50000,
    seed: int = 42,
) -> int:
    """Sample proportionally from all SFT subsets."""
    subsets = {
        "sft_math_full.jsonl": 5_226_364,
        "sft_science_full.jsonl": 2_717_163,
        "sft_chat_full.jsonl": 13_972_873,
        "sft_instruction_following_full.jsonl": 820_263,
        "sft_safety_full.jsonl": 3_570,
        "sft_conversational_agent_full.jsonl": 822_213,
        "sft_swe_full.jsonl": 439_610,
        "sft_terminal_agent_full.jsonl": 822_213,
    }

    # Fallbacks for subsets not yet fully downloaded
    fallbacks = {
        "sft_math_full.jsonl": "sft_math_100k.jsonl",
        "sft_science_full.jsonl": "sft_science_50k.jsonl",
        "sft_instruction_following_full.jsonl": "sft_instruction_following_10k.jsonl",
        "sft_safety_full.jsonl": "sft_safety_all.jsonl",
    }

    total_paper = sum(subsets.values())
    rng = random.Random(seed)
    total_written = 0

    with open(output_file, "w") as out:
        for subset_name, paper_size in subsets.items():
            fpath = os.path.join(data_dir, subset_name)
            if not os.path.exists(fpath):
                fb = fallbacks.get(subset_name)
                if fb:
                    fpath = os.path.join(data_dir, fb)
                if not os.path.exists(fpath):
                    logger.warning(f"Skipping {subset_name}: not available")
                    continue

            # Proportional sample size
            target = max(1, int(total_sample * paper_size / total_paper))

            # Count available lines
            with open(fpath) as f:
                available = sum(1 for _ in f)
            target = min(target, available)

            # Reservoir sampling
            selected = []
            with open(fpath) as f:
                for i, line in enumerate(f):
                    if i < target:
                        selected.append(line)
                    else:
                        j = rng.randint(0, i)
                        if j < target:
                            selected[j] = line

            rng.shuffle(selected)
            for line in selected:
                out.write(line)
                total_written += 1

            logger.info(f"  {os.path.basename(fpath)}: {len(selected)} samples (target {target})")

    logger.info(f"Total: {total_written} samples -> {output_file}")
    return total_written


def run_single_sweep(
    model: str,
    lr: float,
    rank: int,
    max_length: int,
    data_path: str,
    log_base: str,
    batch_size: int = 32,
    max_steps: int = 200,
):
    """Run a single SFT experiment."""
    renderer_name = checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default(
        model_name=model, explicit_renderer_name=None,
        load_checkpoint_path=None, base_url=None,
    )

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    dataset_builder = NemotronCascadeSFTFromFileBuilder(
        common_config=common_config,
        file_path=data_path,
        test_size=256,
        seed=0,
    )

    log_path = f"{log_base}/rank{rank}_lr{lr}_maxlen{max_length}"

    config = train.Config(
        log_path=log_path,
        model_name=model,
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        evaluator_builders=[],
        learning_rate=lr,
        lr_schedule="cosine",
        num_epochs=1,
        lora_rank=rank,
        save_every=0,
        eval_every=0,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-8,
        max_steps=max_steps,
    )

    logger.info(f"Sweep: rank={rank}, lr={lr}, max_len={max_length}")
    asyncio.run(train.main(config))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--data-dir", default=os.path.expanduser("~/data/nemotron-cascade-2"))
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--log-base", default="/tmp/tinker-examples/nemotron_cascade_sweep_v2")

    # What to sweep
    parser.add_argument("--lr", type=float, default=None, help="Single LR (or sweep all)")
    parser.add_argument("--rank", type=int, default=None, help="Single rank (or sweep all)")
    parser.add_argument("--max-length", type=int, default=None, help="Single max_len (or sweep all)")
    args = parser.parse_args()

    # Create sample if needed
    sample_path = os.path.join(args.data_dir, f"sft_sample_{args.sample_size}.jsonl")
    if not os.path.exists(sample_path):
        logger.info(f"Creating {args.sample_size} sample...")
        create_proportional_sample(args.data_dir, sample_path, args.sample_size)

    # Determine sweep dimensions
    lrs = [args.lr] if args.lr else [1e-4, 3e-4, 5e-4]
    ranks = [args.rank] if args.rank else [32, 64, 128]
    max_lengths = [args.max_length] if args.max_length else [16384, 49152]

    logger.info(f"Sweep: {len(lrs)} LRs × {len(ranks)} ranks × {len(max_lengths)} lengths = {len(lrs)*len(ranks)*len(max_lengths)} configs")

    for lr in lrs:
        for rank in ranks:
            for max_len in max_lengths:
                run_single_sweep(
                    model=args.model,
                    lr=lr,
                    rank=rank,
                    max_length=max_len,
                    data_path=sample_path,
                    log_base=args.log_base,
                    batch_size=args.batch_size,
                    max_steps=args.max_steps,
                )


if __name__ == "__main__":
    main()
