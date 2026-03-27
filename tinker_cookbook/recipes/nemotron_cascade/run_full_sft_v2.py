"""
Full-scale SFT for Nemotron-Cascade-2 replication (v2).

Matches the paper's setup more closely:
- All 8 SFT subsets (~24.8M examples)
- LoRA rank 64 (higher capacity)
- Max sequence length 49152 (49K tokens)
- Cosine schedule with warmup
- AdamW (beta1=0.9, beta2=0.98)
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.recipes.nemotron_cascade.sft_datasets import NemotronCascadeSFTFromFileBuilder
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_all_sft_data(data_dir: str, output_file: str) -> int:
    """Combine all SFT data files into one JSONL file."""
    # Priority order: full files first, then partial files
    subsets = [
        "sft_math_full.jsonl",
        "sft_science_full.jsonl",
        "sft_chat_full.jsonl",
        "sft_instruction_following_full.jsonl",
        "sft_safety_full.jsonl",
        "sft_conversational_agent_full.jsonl",
        "sft_swe_full.jsonl",
        "sft_terminal_agent_full.jsonl",
    ]

    # Fallback to partial files if full not available
    fallbacks = {
        "sft_math_full.jsonl": "sft_math_100k.jsonl",
        "sft_science_full.jsonl": "sft_science_50k.jsonl",
        "sft_instruction_following_full.jsonl": "sft_instruction_following_10k.jsonl",
        "sft_safety_full.jsonl": "sft_safety_all.jsonl",
    }

    total = 0
    with open(output_file, "w") as out:
        for subset in subsets:
            fpath = os.path.join(data_dir, subset)
            if not os.path.exists(fpath):
                fallback = fallbacks.get(subset)
                if fallback:
                    fpath = os.path.join(data_dir, fallback)
                if not os.path.exists(fpath):
                    logger.warning(f"Missing: {subset} (no fallback)")
                    continue

            count = 0
            with open(fpath) as inp:
                for line in inp:
                    out.write(line)
                    count += 1
            total += count
            logger.info(f"  Added {count:>10,} from {os.path.basename(fpath)}")

    logger.info(f"Combined: {total:,} total examples -> {output_file}")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=49152,
                        help="Max sequence length (paper uses 256K packed, we use 49K per-sequence)")
    parser.add_argument("--lora-rank", type=int, default=64,
                        help="LoRA rank (higher = more capacity, paper uses full FT)")
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Paper uses ~1.5 epochs")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.expanduser("~/data/nemotron-cascade-2"))
    parser.add_argument("--combined-file", type=str, default=None)
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming for very large datasets")
    parser.add_argument("--packing", action="store_true",
                        help="Pack multiple short examples into each sequence for efficiency")
    parser.add_argument("--max-packed-length", type=int, default=49152,
                        help="Max tokens per packed sequence (only used with --packing)")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="Wandb project for logging (e.g. nemotron-cascade-2-replication)")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Wandb run name")
    args = parser.parse_args()

    # Combine data
    if args.combined_file and os.path.exists(args.combined_file):
        data_path = args.combined_file
    else:
        data_path = os.path.join(args.data_dir, "sft_all_combined.jsonl")
        if not os.path.exists(data_path):
            logger.info("Combining SFT data files...")
            combine_all_sft_data(args.data_dir, data_path)
        else:
            logger.info(f"Using existing combined file: {data_path}")

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
        test_size=1024,
        seed=42,
        packing=args.packing,
        max_packed_length=args.max_packed_length,
    )

    model_short = args.model.replace("/", "-").replace(":", "-")
    log_path = args.log_path or (
        f"/tmp/tinker-examples/nemotron_cascade_sft_v2/"
        f"{model_short}_rank{args.lora_rank}_lr{args.lr}_maxlen{args.max_length}"
    )

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
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name or f"sft_v2_rank{args.lora_rank}_lr{args.lr}{'_packed' if args.packing else ''}",
    )

    logger.info(f"SFT v2: model={args.model}, rank={args.lora_rank}, "
                f"lr={args.lr}, max_len={args.max_length}, log={log_path}")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
