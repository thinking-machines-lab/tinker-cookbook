"""
Run IF-RL training for Nemotron-Cascade-2 replication.

Loads from the SFT checkpoint and runs instruction-following RL
using GRPO with dynamic filtering (paper's approach).

Paper hyperparameters:
  - Batch size: 128, Rollouts: 16, Temp: 1.0
  - LR: 3e-6 (AdamW), KL coeff: 0
  - Max response: 49K tokens
  - Steps: ~180 with dynamic filtering
"""

import argparse
import asyncio
import glob
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_latest_checkpoint(log_dir: str) -> str | None:
    """Find the latest checkpoint in a training log directory."""
    checkpoints_file = os.path.join(log_dir, "checkpoints.jsonl")
    if not os.path.exists(checkpoints_file):
        # Try to find final checkpoint from the metrics
        return None

    latest = None
    with open(checkpoints_file) as f:
        for line in f:
            try:
                cp = json.loads(line)
                latest = cp
            except json.JSONDecodeError:
                continue

    if latest and "state_path" in latest:
        return latest["state_path"]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b:peft:131072")
    parser.add_argument("--sft-log-dir", type=str, default=None,
                        help="Path to SFT log directory to load checkpoint from")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Direct path to checkpoint (overrides --sft-log-dir)")
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=49152)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--no-dynamic-filter", action="store_true",
                        help="Disable dynamic filtering (keep all groups)")
    args = parser.parse_args()

    # Resolve checkpoint path
    checkpoint_path = args.load_checkpoint
    if checkpoint_path is None and args.sft_log_dir:
        checkpoint_path = find_latest_checkpoint(args.sft_log_dir)
        if checkpoint_path:
            logger.info(f"Found SFT checkpoint: {checkpoint_path}")
        else:
            logger.warning(f"No checkpoint found in {args.sft_log_dir}")

    from tinker_cookbook.recipes.nemotron_cascade.train_rl import CLIConfig, cli_main

    config = CLIConfig(
        model_name=args.model,
        env="if_rl",
        group_size=args.group_size,
        groups_per_batch=args.batch_size,
        learning_rate=args.lr,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        kl_penalty_coef=0.0,
        max_steps=args.max_steps,
        save_every=args.save_every,
        eval_every=args.eval_every,
        load_checkpoint_path=checkpoint_path,
        log_path=args.log_path or "/tmp/tinker-examples/nemotron_cascade_ifrl",
        behavior_if_log_dir_exists="overwrite",
        remove_constant_reward_groups=not args.no_dynamic_filter,
    )

    logger.info(f"Starting IF-RL: model={args.model}, lr={args.lr}, checkpoint={checkpoint_path}")
    asyncio.run(cli_main(config))
    logger.info("IF-RL completed!")


if __name__ == "__main__":
    main()
