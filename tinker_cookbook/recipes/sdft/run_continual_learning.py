"""
Continual learning experiment: SFT vs SDFT (old) vs SDFT (top-K).

Runs a 2-stage sequential learning experiment:
  Stage 1: Train on tool-use task, evaluate both tool-use and science
  Stage 2: Train on science task (from Stage 1 checkpoint), evaluate both

Three methods are compared from the same base model:
  - SFT: Standard supervised fine-tuning (cross_entropy on golden answers)
  - SDFT-IS: Old SDFT with importance_sampling loss (topk=0)
  - SDFT-TopK: New SDFT with top-K distillation (topk=20)

Usage:
    # Run everything (all methods, all LRs, both stages)
    python -m tinker_cookbook.recipes.sdft.run_continual_learning

    # Run a single configuration
    python -m tinker_cookbook.recipes.sdft.run_continual_learning \
        methods=sdft_topk learning_rates=1e-3 stages=1

    # Dry run (print configs without running)
    python -m tinker_cookbook.recipes.sdft.run_continual_learning dry_run=true
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Literal

import chz

from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.distillation import sdft
from tinker_cookbook.recipes.sdft.datasets import (
    SDFTDataset,
    load_science_from_arrow,
    load_tooluse_from_arrow,
)
from tinker_cookbook.supervised import train as sl_train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

Method = Literal["sft", "sdft_is", "sdft_topk", "sdft_hybrid"]

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DATA_DIR = "~/Self-Distillation/data"
LORA_RANK = 128
BATCH_SIZE = 128
MAX_TOKENS = 2048
EVAL_EVERY = 10  # Eval every N batches
SAVE_EVERY = 100  # Save infrequently, we'll save at end


@chz.chz
class ExperimentConfig:
    """Configuration for the continual learning experiment."""

    model_name: str = MODEL_NAME
    data_dir: str = DATA_DIR
    lora_rank: int = LORA_RANK
    batch_size: int = BATCH_SIZE
    max_tokens: int = MAX_TOKENS

    # What to run
    methods: str = "sft,sdft_is,sdft_topk"  # comma-separated
    learning_rates: str = "5e-4,1e-3"  # comma-separated
    stages: str = "1,2"  # comma-separated

    # SDFT-TopK specific
    topk: int = 20
    thinking_format: bool = False

    # Logging
    log_root: str = "/tmp/tinker-sdft-continual-v3"
    wandb_project: str | None = None
    base_url: str | None = None

    # Control
    dry_run: bool = False

    def get_methods(self) -> list[Method]:
        return [m.strip() for m in self.methods.split(",")]  # type: ignore[return-value]

    def get_learning_rates(self) -> list[float]:
        return [float(lr.strip()) for lr in self.learning_rates.split(",")]

    def get_stages(self) -> list[int]:
        return [int(s.strip()) for s in self.stages.split(",")]


# ---------------------------------------------------------------------------
# Training: SFT
# ---------------------------------------------------------------------------


async def run_sft_stage(
    config: ExperimentConfig,
    task: str,  # "tooluse" or "science"
    learning_rate: float,
    renderer_name: str,
    log_path: str,
    wandb_name: str | None = None,
    load_checkpoint_path: str | None = None,
) -> str | None:
    """Run SFT training on a task and return the checkpoint path."""
    from tinker_cookbook.recipes.sdft.datasets import (
        ScienceArrowSFTBuilder,
        TooluseArrowSFTBuilder,
    )

    expanded_data_dir = str(Path(config.data_dir).expanduser())

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        max_length=config.max_tokens,
        batch_size=config.batch_size,
    )

    if task == "tooluse":
        dataset_builder = TooluseArrowSFTBuilder(
            common_config=common_config,
            data_dir=f"{expanded_data_dir}/tooluse_data",
        )
    elif task == "science":
        dataset_builder = ScienceArrowSFTBuilder(
            common_config=common_config,
            data_dir=f"{expanded_data_dir}/science_data",
            thinking_format=config.thinking_format,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    sl_config = sl_train.Config(
        model_name=config.model_name,
        renderer_name=renderer_name,
        learning_rate=learning_rate,
        lora_rank=config.lora_rank,
        log_path=log_path,
        wandb_project=config.wandb_project,
        wandb_name=wandb_name,
        base_url=config.base_url,
        eval_every=EVAL_EVERY,
        save_every=SAVE_EVERY,
        load_checkpoint_path=load_checkpoint_path,
        dataset_builder=dataset_builder,
    )

    await sl_train.main(sl_config)

    ckpt = checkpoint_utils.get_last_checkpoint(log_path)
    if ckpt and ckpt.state_path:
        return ckpt.state_path
    return None


# ---------------------------------------------------------------------------
# Training: SDFT
# ---------------------------------------------------------------------------


async def run_sdft_stage(
    config: ExperimentConfig,
    task: str,
    learning_rate: float,
    renderer_name: str,
    log_path: str,
    topk: int,
    wandb_name: str | None = None,
    load_checkpoint_path: str | None = None,
) -> str | None:
    """Run SDFT training on a task and return the checkpoint path."""
    expanded_data_dir = str(Path(config.data_dir).expanduser())
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    if task == "tooluse":
        train_q, train_a, _, _ = load_tooluse_from_arrow(f"{expanded_data_dir}/tooluse_data")
    elif task == "science":
        train_q, train_a, _, _ = load_science_from_arrow(
            f"{expanded_data_dir}/science_data",
            thinking_format=config.thinking_format,
            for_sft=False,
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    train_dataset = SDFTDataset(
        questions=train_q,
        golden_answers=train_a,
        batch_size=config.batch_size,
        group_size=1,
        renderer=renderer,
        dataset_name=f"sdft_{task}",
    )

    loss_fn: str = "cross_entropy" if topk > 0 else "importance_sampling"

    sdft_config = sdft.Config(
        model_name=config.model_name,
        renderer_name=renderer_name,
        lora_rank=config.lora_rank,
        base_url=config.base_url,
        learning_rate=learning_rate,
        max_tokens=config.max_tokens,
        topk=topk,
        loss_fn=loss_fn,  # type: ignore[arg-type]
        eval_every=EVAL_EVERY,
        save_every=SAVE_EVERY,
        log_path=log_path,
        wandb_project=config.wandb_project,
        wandb_name=wandb_name,
        load_checkpoint_path=load_checkpoint_path,
    )

    await sdft.main(sdft_config, train_dataset, test_dataset=None)

    ckpt = checkpoint_utils.get_last_checkpoint(log_path)
    if ckpt and ckpt.state_path:
        return ckpt.state_path
    return None


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


async def run_single(
    config: ExperimentConfig,
    method: Method,
    learning_rate: float,
    stage: int,
    renderer_name: str,
    stage1_checkpoints: dict[str, str | None],
) -> dict:
    """Run a single (method, lr, stage) configuration and return results."""
    lr_str = f"{learning_rate:.0e}".replace("+", "")
    run_id = f"stage{stage}_{method}_lr{lr_str}"
    log_path = f"{config.log_root}/{run_id}"

    task = "tooluse" if stage == 1 else "science"

    # For stage 2, load from stage 1 checkpoint
    load_checkpoint_path = None
    if stage == 2:
        s1_key = f"stage1_{method}_lr{lr_str}"
        load_checkpoint_path = stage1_checkpoints.get(s1_key)
        if not load_checkpoint_path:
            logger.warning(f"No stage 1 checkpoint for {s1_key}, skipping {run_id}")
            return {"run_id": run_id, "error": "no_stage1_checkpoint"}

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Running: {run_id}")
    logger.info(f"  Method: {method}, LR: {learning_rate}, Stage: {stage}, Task: {task}")
    if load_checkpoint_path:
        logger.info(f"  Loading from: {load_checkpoint_path}")
    logger.info(f"  Log path: {log_path}")
    logger.info(f"{'=' * 60}\n")

    if config.dry_run:
        return {"run_id": run_id, "dry_run": True}

    # Train
    if method == "sft":
        checkpoint_path = await run_sft_stage(
            config,
            task,
            learning_rate,
            renderer_name,
            log_path,
            wandb_name=run_id,
            load_checkpoint_path=load_checkpoint_path,
        )
    elif method == "sdft_is":
        checkpoint_path = await run_sdft_stage(
            config,
            task,
            learning_rate,
            renderer_name,
            log_path,
            topk=0,
            wandb_name=run_id,
            load_checkpoint_path=load_checkpoint_path,
        )
    elif method in ("sdft_topk", "sdft_hybrid"):
        checkpoint_path = await run_sdft_stage(
            config,
            task,
            learning_rate,
            renderer_name,
            log_path,
            topk=config.topk,
            wandb_name=run_id,
            load_checkpoint_path=load_checkpoint_path,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    result = {
        "run_id": run_id,
        "method": method,
        "learning_rate": learning_rate,
        "stage": stage,
        "task": task,
        "checkpoint_path": checkpoint_path,
        "log_path": log_path,
    }

    # Save result
    Path(log_path).mkdir(parents=True, exist_ok=True)
    with open(f"{log_path}/run_result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


async def cli_main(config: ExperimentConfig) -> None:
    """Run the full continual learning experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    methods = config.get_methods()
    learning_rates = config.get_learning_rates()
    stages = config.get_stages()

    logger.info(f"Methods: {methods}")
    logger.info(f"Learning rates: {learning_rates}")
    logger.info(f"Stages: {stages}")
    logger.info(f"Log root: {config.log_root}")

    # Resolve renderer
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=config.model_name,
        explicit_renderer_name=None,
        load_checkpoint_path=None,
        base_url=config.base_url,
    )

    # Run stage 1
    stage1_checkpoints: dict[str, str | None] = {}
    if 1 in stages:
        for method in methods:
            for lr in learning_rates:
                result = await run_single(
                    config,
                    method,
                    lr,
                    stage=1,
                    renderer_name=renderer_name,
                    stage1_checkpoints={},
                )
                lr_str = f"{lr:.0e}".replace("+", "")
                key = f"stage1_{method}_lr{lr_str}"
                stage1_checkpoints[key] = result.get("checkpoint_path")

        # Save stage 1 checkpoint map
        Path(config.log_root).mkdir(parents=True, exist_ok=True)
        with open(f"{config.log_root}/stage1_checkpoints.json", "w") as f:
            json.dump(stage1_checkpoints, f, indent=2)
        logger.info(f"Stage 1 checkpoints: {json.dumps(stage1_checkpoints, indent=2)}")

    # Load stage 1 checkpoints if we're only running stage 2
    if 1 not in stages and 2 in stages:
        ckpt_file = f"{config.log_root}/stage1_checkpoints.json"
        if Path(ckpt_file).exists():
            with open(ckpt_file) as f:
                stage1_checkpoints = json.load(f)
            logger.info(f"Loaded stage 1 checkpoints from {ckpt_file}")
        else:
            logger.error(f"No stage 1 checkpoints found at {ckpt_file}")
            return

    # Run stage 2
    if 2 in stages:
        for method in methods:
            for lr in learning_rates:
                await run_single(
                    config,
                    method,
                    lr,
                    stage=2,
                    renderer_name=renderer_name,
                    stage1_checkpoints=stage1_checkpoints,
                )

    logger.info("\n" + "=" * 60)
    logger.info("  EXPERIMENT COMPLETE")
    logger.info(f"  Results in: {config.log_root}")
    logger.info("=" * 60)


if __name__ == "__main__":
    config = chz.entrypoint(ExperimentConfig)
    asyncio.run(cli_main(config))
