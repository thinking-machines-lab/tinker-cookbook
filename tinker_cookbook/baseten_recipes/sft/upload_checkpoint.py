"""
Upload a trained checkpoint to HuggingFace Hub without retraining.

This script downloads a checkpoint from Tinker, merges it with the base model,
and uploads the merged model to HuggingFace Hub.

Usage:
    uv run python -m tinker_cookbook.baseten_recipes.sft.upload_checkpoint \\
        --checkpoint-path tinker://xxx/sampler_weights/final \\
        --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --hf-repo-id username/model-name \\
        --hf-private

    Or load from a training log directory:
    uv run python -m tinker_cookbook.baseten_recipes.sft.upload_checkpoint \\
        --log-dir /tmp/tinker-examples/sft/my-run \\
        --base-model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --hf-repo-id username/model-name
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.baseten_recipes.sft import post_training_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_environment():
    """Load environment variables from .env file."""
    # Look for .env in project root
    current = Path.cwd()
    for _ in range(5):
        if (current / "pyproject.toml").exists():
            dotenv_path = current / ".env"
            if dotenv_path.exists():
                logger.info(f"Loading environment from {dotenv_path.absolute()}")
                load_dotenv(dotenv_path)
                break
        if current.parent == current:
            break
        current = current.parent

    # Verify HF token
    if not os.environ.get("HF_TOKEN"):
        logger.error("HF_TOKEN not found in environment. Set it in .env or export it.")
        sys.exit(1)


async def main():
    parser = argparse.ArgumentParser(
        description="Upload a trained checkpoint to HuggingFace Hub"
    )

    # Checkpoint location (one of these required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint-path",
        type=str,
        help="Tinker checkpoint path (e.g., tinker://xxx/sampler_weights/final)"
    )
    group.add_argument(
        "--log-dir",
        type=str,
        help="Training log directory (will use last checkpoint from checkpoints.jsonl)"
    )

    # Model and upload config
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model name (e.g., Qwen/Qwen3-30B-A3B-Instruct-2507)"
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., username/model-name)"
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        default=True,
        help="Make HuggingFace repo private (default: true)"
    )
    parser.add_argument(
        "--hf-public",
        action="store_true",
        help="Make HuggingFace repo public (overrides --hf-private)"
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="/tmp/tinker-merged-models",
        help="Local directory for downloading and merging (default: /tmp/tinker-merged-models)"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank used during training (for model card, default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate used during training (for model card)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size used during training (for model card)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="custom",
        help="Dataset name used during training (for model card)"
    )

    args = parser.parse_args()

    # Setup environment
    setup_environment()

    # Get checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        logger.info(f"Using checkpoint: {checkpoint_path}")
    else:
        # Load from log directory
        log_dir = args.log_dir
        logger.info(f"Loading checkpoint from log directory: {log_dir}")
        final_checkpoint = checkpoint_utils.get_last_checkpoint(log_dir, required_key="sampler_path")
        if final_checkpoint is None:
            logger.error(f"No checkpoint found in {log_dir}/checkpoints.jsonl")
            sys.exit(1)
        checkpoint_path = final_checkpoint["sampler_path"]
        logger.info(f"Found checkpoint: {checkpoint_path}")

    # Determine if repo should be private
    hf_private = args.hf_private and not args.hf_public

    # Create minimal training config for model card generation
    class MinimalConfig:
        def __init__(self):
            self.lora_rank = args.lora_rank
            self.learning_rate = args.learning_rate or "auto"
            self.lr_schedule = "linear"
            self.num_epochs = 1
            self.dataset_builder = type('obj', (object,), {
                'common_config': type('obj', (object,), {
                    'batch_size': args.batch_size or "N/A"
                })()
            })()

    training_config = MinimalConfig()

    logger.info("\n" + "=" * 80)
    logger.info("Starting post-training pipeline")
    logger.info("=" * 80)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"HuggingFace repo: {args.hf_repo_id}")
    logger.info(f"Private: {hf_private}")
    logger.info(f"Download dir: {args.download_dir}")
    logger.info("=" * 80)

    # Confirm before proceeding
    print("\n⚠️  WARNING: This will require significant disk space!")
    print(f"   - Base model download: ~60GB (for 30B models)")
    print(f"   - Checkpoint download: ~5-10GB")
    print(f"   - Merged model: ~60GB")
    print(f"   - Total: ~125-130GB\n")

    response = input("Continue with upload? [y/N]: ")
    if response.lower() != 'y':
        logger.info("Upload cancelled")
        return

    try:
        result = await post_training_pipeline.run_post_training_pipeline(
            checkpoint_path=checkpoint_path,
            base_model=args.base_model,
            repo_id=args.hf_repo_id,
            private=hf_private,
            download_dir=args.download_dir,
            training_config=training_config,
            metrics={},
        )

        logger.info("\n" + "=" * 80)
        logger.info("✅ Upload completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Model URL: {result['hf_url']}")
        logger.info(f"Merged model saved locally: {result['merged_model_dir']}")

    except Exception as e:
        logger.error(f"❌ Upload failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
