"""
SFT Recipe - Complete supervised fine-tuning pipeline with optional HuggingFace upload.

This script trains models using a YAML configuration file and loads API keys from .env.

Usage:
    # Copy example config
    cp configs/example.yaml configs/my_config.yaml

    # Edit configs/my_config.yaml with your settings
    # Make sure .env in project root has TINKER_API_KEY and HF_TOKEN

    # Run training with uv (recommended)
    uv run python -m tinker_cookbook.baseten_recipes.sft.train configs/my_config.yaml

    # Or without uv
    python -m tinker_cookbook.baseten_recipes.sft.train configs/my_config.yaml
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from tinker_cookbook import cli_utils, hyperparam_utils, model_info, renderers
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule
from tinker_cookbook.baseten_recipes.sft import sft_datasets

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    # If path doesn't exist and is relative, try resolving relative to script location
    if not config_file.exists() and not config_file.is_absolute():
        # Try relative to the sft directory
        script_dir = Path(__file__).parent
        config_file = script_dir / config_path

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Try using an absolute path or run from tinker_cookbook/baseten_recipes/sft/\n"
            f"Example: configs/phare.yaml or {Path(__file__).parent}/configs/phare.yaml"
        )

    with open(config_file) as f:
        config = yaml.safe_load(f)

    return config


def setup_environment():
    """Load environment variables from .env file."""
    # Look for .env in project root first, then current directory and parent directories
    current = Path.cwd()

    # Try to find project root by looking for pyproject.toml
    search_path = current
    for _ in range(5):  # Search up to 5 levels up
        if (search_path / "pyproject.toml").exists():
            dotenv_path = search_path / ".env"
            if dotenv_path.exists():
                logger.info(f"Loading environment from {dotenv_path.absolute()}")
                load_dotenv(dotenv_path)
                break
        if search_path.parent == search_path:
            break
        search_path = search_path.parent
    else:
        # Fallback to checking current directory and parents
        dotenv_path = Path(".env")
        if not dotenv_path.exists():
            dotenv_path = Path("../.env")
        if not dotenv_path.exists():
            dotenv_path = Path("../../.env")
        if not dotenv_path.exists():
            dotenv_path = Path("../../../.env")

        if dotenv_path.exists():
            logger.info(f"Loading environment from {dotenv_path.absolute()}")
            load_dotenv(dotenv_path)
        else:
            logger.warning("No .env file found. API keys must be set in environment.")

    # Verify required keys
    if not os.environ.get("TINKER_API_KEY"):
        logger.warning("TINKER_API_KEY not set in environment")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set in environment (required for HuggingFace datasets/upload)")
    else:
        # Login to HuggingFace Hub with the token
        try:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("Successfully logged in to HuggingFace Hub")
        except Exception as e:
            logger.warning(f"Failed to login to HuggingFace Hub: {e}")


def get_dataset_builder(
    config: dict[str, Any],
    model_name: str,
    renderer_name: str,
) -> ChatDatasetBuilder:
    """Factory function to create dataset builders from config.

    Args:
        config: Full configuration dictionary
        model_name: Model name for tokenizer
        renderer_name: Renderer name for chat formatting

    Returns:
        ChatDatasetBuilder instance
    """
    dataset_config = config["dataset"]
    dataset_name = dataset_config["name"]
    max_length = dataset_config.get("max_length")
    batch_size = dataset_config["batch_size"]
    train_on_what_str = dataset_config.get("train_on_what")
    num_examples = dataset_config.get("num_examples")

    # Convert train_on_what string to enum
    train_on_what = None
    if train_on_what_str:
        train_on_what = getattr(renderers.TrainOnWhat, train_on_what_str)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=max_length,
        batch_size=batch_size,
        train_on_what=train_on_what,
    )

    if dataset_name == "no_robots":
        builder = sft_datasets.NoRobotsBuilder(common_config=common_config)
    elif dataset_name == "tulu3":
        builder = sft_datasets.Tulu3Builder(common_config=common_config)
    elif dataset_name.endswith(".jsonl"):
        # Load conversations from a JSONL file
        builder = FromConversationFileBuilder(
            common_config=common_config,
            file_path=dataset_name,
        )
    else:
        # Treat as a HuggingFace dataset with 'messages' column
        builder = sft_datasets.HuggingFaceMessagesBuilder(
            common_config=common_config,
            dataset_name=dataset_name,
        )

    # Wrap with example limiter if num_examples is set
    if num_examples is not None:
        from tinker_cookbook.baseten_recipes.sft.dataset_utils import LimitedDatasetBuilder

        builder = LimitedDatasetBuilder(builder=builder, num_examples=num_examples)

    return builder


async def main(config_path: str) -> None:
    """Main entry point for SFT training.

    Args:
        config_path: Path to config.yaml file
    """
    # Setup environment and load config
    setup_environment()
    config = load_config(config_path)

    # Extract config sections
    model_config = config["model"]
    dataset_config = config["dataset"]
    training_config = config["training"]
    checkpoint_config = config["checkpointing"]
    post_training_config = config["post_training"]
    logging_config = config["logging"]
    infra_config = config["infrastructure"]

    # Auto-detect renderer if not specified
    model_name = model_config["name"]
    renderer_name = model_config.get("renderer_name") or model_info.get_recommended_renderer_name(
        model_name
    )

    # Auto-compute learning rate if not specified
    learning_rate = training_config.get("learning_rate")
    if learning_rate is None:
        learning_rate = hyperparam_utils.get_lr(model_name, is_lora=True)
        logger.info(f"Auto-computed learning rate: {learning_rate}")

    # Build run name and paths
    model_name_sanitized = model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"{dataset_config['name']}-{model_name_sanitized}-"
        f"{model_config['lora_rank']}rank-{learning_rate}lr-"
        f"{dataset_config['batch_size']}batch-{date_and_time}"
    )

    log_path = logging_config.get("log_path")
    if log_path is None:
        log_path = f"/tmp/tinker-examples/sft/{run_name}"

    wandb_name = logging_config.get("wandb_name") or run_name

    # Set wandb entity in environment if specified in config
    wandb_entity = logging_config.get("wandb_entity")
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity
        logger.info(f"Setting WANDB_ENTITY to: {wandb_entity}")

    # Check log directory
    behavior = infra_config.get("behavior_if_log_dir_exists", "ask")
    cli_utils.check_log_dir(log_path, behavior_if_exists=behavior)

    # Build dataset
    dataset_builder = get_dataset_builder(config, model_name, renderer_name)

    # Build training configuration
    train_config = train.Config(
        log_path=log_path,
        model_name=model_name,
        load_checkpoint_path=infra_config.get("load_checkpoint_path"),
        dataset_builder=dataset_builder,
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        learning_rate=learning_rate,
        lr_schedule=training_config.get("lr_schedule", "linear"),
        num_epochs=training_config["num_epochs"],
        base_url=infra_config.get("base_url"),
        wandb_project=logging_config.get("wandb_project"),
        wandb_name=wandb_name,
        print_examples_every=logging_config.get("print_examples_every", 1),
        lora_rank=model_config["lora_rank"],
        save_every=checkpoint_config["save_every"],
        eval_every=checkpoint_config["eval_every"],
        ttl_seconds=checkpoint_config.get("ttl_seconds", 604800),
    )

    # Log training info
    logger.info("=" * 80)
    logger.info(f"SFT Training: {run_name}")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_config['name']}")
    logger.info(f"LoRA rank: {model_config['lora_rank']}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {dataset_config['batch_size']}")
    logger.info(f"Renderer: {renderer_name}")
    logger.info(f"Num epochs: {training_config['num_epochs']}")
    if dataset_config.get("num_examples"):
        logger.info(f"Num examples: {dataset_config['num_examples']}")
    logger.info(f"Log path: {log_path}")
    logger.info("=" * 80)

    # Run training
    await train.main(train_config)

    # Post-training pipeline
    if post_training_config.get("enabled"):
        logger.info("\n" + "=" * 80)
        logger.info("Starting post-training pipeline")
        logger.info("=" * 80)

        hf_repo_id = post_training_config.get("hf_repo_id")
        if hf_repo_id is None:
            logger.error("hf_repo_id is required when post_training.enabled is true")
            return

        try:
            from tinker_cookbook.baseten_recipes.sft import post_training_pipeline
            from tinker_cookbook import checkpoint_utils

            # Get the final checkpoint path from checkpoints.jsonl
            final_checkpoint = checkpoint_utils.get_last_checkpoint(
                log_path, required_key="sampler_path"
            )
            if final_checkpoint is None:
                logger.error("No checkpoint found in checkpoints.jsonl")
                return

            checkpoint_path = final_checkpoint["sampler_path"]
            logger.info(f"Using checkpoint: {checkpoint_path}")

            # Run the post-training pipeline
            result = await post_training_pipeline.run_post_training_pipeline(
                checkpoint_path=checkpoint_path,
                base_model=model_name,
                repo_id=hf_repo_id,
                private=post_training_config.get("hf_private", True),
                download_dir=post_training_config.get("download_dir", "/tmp/tinker-merged-models"),
                training_config=train_config,
                metrics={},  # TODO: Extract from metrics.jsonl
            )

            logger.info("\n" + "=" * 80)
            logger.info("Post-training pipeline completed successfully!")
            logger.info("=" * 80)
            logger.info(f"Model uploaded to: {result['hf_url']}")

        except ImportError as e:
            logger.error(f"Post-training pipeline not available: {e}")
            logger.error("Install required dependencies: uv pip install pyyaml python-dotenv")
        except Exception as e:
            logger.error(f"Post-training pipeline failed: {e}", exc_info=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m tinker_cookbook.baseten_recipes.sft.train <config.yaml>")
        print("\nExample:")
        print("  cp configs/example.yaml configs/my_config.yaml")
        print("  # Edit configs/my_config.yaml with your settings")
        print("  uv run python -m tinker_cookbook.baseten_recipes.sft.train configs/my_config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    asyncio.run(main(config_path))
