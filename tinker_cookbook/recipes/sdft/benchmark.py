"""
SDFT Benchmark Script.

Runs independent SFT and SDFT comparisons from the same base model.
Uses the paper's eval data from the SDFT repo for exact reproducibility.

Paper target numbers (Qwen2.5-7B-Instruct):
    SciKnowEval Chemistry L3: Base=32.1, SFT=66.2, SDFT=70.2
    ToolAlpaca Tool Use:       Base=42.9, SFT=63.2, SDFT=70.6

We use Qwen3-8B (different model), so exact numbers will differ,
but the pattern (SDFT > SFT) should hold.

Usage:
    # Run all comparisons (base eval, SFT, SDFT — SFT and SDFT are independent)
    python -m tinker_cookbook.recipes.sdft.benchmark \
        dataset=science \
        sdft_repo_path=~/Repos/Self-Distillation

    # Eval base model only
    python -m tinker_cookbook.recipes.sdft.benchmark \
        dataset=science \
        sdft_repo_path=~/Repos/Self-Distillation \
        phase=base_eval

    # SDFT training only (independent from SFT)
    python -m tinker_cookbook.recipes.sdft.benchmark \
        dataset=science \
        sdft_repo_path=~/Repos/Self-Distillation \
        phase=sdft

    # Eval an existing checkpoint
    python -m tinker_cookbook.recipes.sdft.benchmark \
        dataset=science \
        sdft_repo_path=~/Repos/Self-Distillation \
        phase=eval_checkpoint \
        checkpoint_path=tinker://...
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import chz

from tinker_cookbook import checkpoint_utils, cli_utils

logger = logging.getLogger(__name__)

Phase = Literal["all", "base_eval", "sft", "sdft", "eval_checkpoint"]


@chz.chz
class BenchmarkConfig:
    """Configuration for running SDFT benchmarks."""

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 128
    renderer_name: str | None = None

    # Data source: path to the cloned SDFT paper repo
    sdft_repo_path: str = "~/Repos/Self-Distillation"

    # Benchmark dataset
    dataset: str = "science"  # science | tooluse

    # Which phase to run. SFT and SDFT are independent comparisons from the
    # same base model (not a sequential pipeline).
    phase: Phase = "all"

    # For eval_checkpoint phase: path to evaluate
    checkpoint_path: str | None = None

    # SDFT training hyperparameters (paper defaults)
    learning_rate: float = 5e-5
    groups_per_batch: int = 32
    max_tokens: int = 1024
    max_steps: int | None = None
    eval_every: int = 20
    save_every: int = 20

    # SFT training hyperparameters
    sft_learning_rate: float = 5e-5
    sft_batch_size: int = 32
    sft_max_steps: int | None = None

    # Logging
    log_root: str | None = None
    wandb_project: str | None = None
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _resolve_data_paths(config: BenchmarkConfig) -> tuple[str, str, str]:
    """Resolve train and eval data paths from the SDFT repo."""
    repo = Path(config.sdft_repo_path).expanduser()
    if config.dataset == "science":
        train_path = str(repo / "data" / "science_data" / "train_data")
        eval_path = str(repo / "data" / "science_data" / "eval_data")
        eval_dataset = "science"
    elif config.dataset == "tooluse":
        train_path = str(repo / "data" / "tooluse_data" / "train_data")
        eval_path = str(repo / "data" / "tooluse_data" / "eval_data")
        eval_dataset = "tooluse"
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    # Verify paths exist
    for path in [train_path, eval_path]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Data path not found: {path}. "
                f"Clone the SDFT repo to {repo} or set sdft_repo_path correctly."
            )
    return train_path, eval_path, eval_dataset


def _make_log_path(config: BenchmarkConfig, phase: str) -> str:
    if config.log_root:
        root = config.log_root
    else:
        root = "/tmp/tinker-examples/sdft-benchmark"
    model_slug = config.model_name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    return f"{root}/{config.dataset}-{model_slug}-{phase}-{timestamp}"


async def run_base_eval(config: BenchmarkConfig) -> dict[str, float]:
    """Phase 1: Evaluate the base model (no training)."""
    from tinker_cookbook.recipes.sdft.eval import run_eval

    _, eval_path, eval_dataset = _resolve_data_paths(config)
    renderer_name = await _resolve_renderer(config)

    log_path = _make_log_path(config, "base-eval")
    output_path = f"{log_path}/eval_results.json"

    logger.info(f"=== Phase: Base model eval on {config.dataset} ===")
    logger.info(f"Model: {config.model_name}")

    metrics = await run_eval(
        model_name=config.model_name,
        eval_dataset=eval_dataset,
        eval_data_path=eval_path,
        renderer_name=renderer_name,
        base_url=config.base_url,
        model_path=None,
        max_tokens=2048 if config.dataset == "science" else 1024,
        output_path=output_path,
    )

    _print_metrics("Base model", metrics)
    return metrics


async def run_sft(config: BenchmarkConfig) -> str | None:
    """Phase 2: Run SFT training and return checkpoint path."""
    from datasets import load_from_disk

    from tinker_cookbook.supervised import train as sl_train
    from tinker_cookbook.supervised.data import conversation_to_datum

    train_path, eval_path, eval_dataset = _resolve_data_paths(config)
    renderer_name = await _resolve_renderer(config)

    log_path = _make_log_path(config, "sft")
    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)

    logger.info(f"=== Phase: SFT training on {config.dataset} ===")

    # Load training data
    ds = load_from_disk(train_path)

    from tinker_cookbook import renderers as rmod
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(config.model_name)
    renderer = rmod.get_renderer(renderer_name, tokenizer=tokenizer)

    # Convert to SL datums
    datums = []
    if config.dataset == "science":
        for row in ds:  # type: ignore[union-attr]
            messages = row["messages"]  # type: ignore[index]
            golden = row["output_text"]  # type: ignore[index]
            # SFT: train on the golden answer directly
            sft_messages = list(messages) + [{"role": "assistant", "content": golden}]
            datum = conversation_to_datum(sft_messages, renderer, max_length=2048)  # type: ignore[arg-type]
            if datum is not None:
                datums.append(datum)
    elif config.dataset == "tooluse":
        for row in ds:  # type: ignore[union-attr]
            prompt = row["prompt"]  # type: ignore[index]
            golden = "\n".join(row["golden_response"])  # type: ignore[index]
            sft_messages: list[dict[str, str]] = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": golden},
            ]
            datum = conversation_to_datum(sft_messages, renderer, max_length=2048)  # type: ignore[arg-type]
            if datum is not None:
                datums.append(datum)

    logger.info(f"Prepared {len(datums)} SFT training datums")

    # SFT training uses the supervised train loop
    sl_config = sl_train.Config(
        model_name=config.model_name,
        renderer_name=renderer_name,
        learning_rate=config.sft_learning_rate,
        lora_rank=config.lora_rank,
        log_path=log_path,
        wandb_project=config.wandb_project,
        wandb_name=f"sdft-bench-sft-{config.dataset}",
        base_url=config.base_url,
        eval_every=config.eval_every,
        save_every=config.save_every,
    )

    from tinker_cookbook.supervised.data import SupervisedDatasetFromList

    train_dataset = SupervisedDatasetFromList(datums, batch_size=config.sft_batch_size)
    await sl_train.main(sl_config, train_dataset, test_dataset=None)

    # Find final checkpoint
    ckpt = checkpoint_utils.get_last_checkpoint(log_path)
    if ckpt:
        logger.info(f"SFT checkpoint: {ckpt.state_path}")
        return ckpt.state_path
    return None


async def run_sdft(config: BenchmarkConfig) -> str | None:
    """Phase 3: Run SDFT training and return checkpoint path."""
    from tinker_cookbook.distillation import sdft
    from tinker_cookbook.recipes.sdft.datasets import SDFTDataset, load_sdft_from_arrow

    train_path, eval_path, eval_dataset = _resolve_data_paths(config)
    renderer_name = await _resolve_renderer(config)

    log_path = _make_log_path(config, "sdft")
    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)

    logger.info(f"=== Phase: SDFT training on {config.dataset} ===")

    from tinker_cookbook import renderers as rmod
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(config.model_name)
    renderer = rmod.get_renderer(renderer_name, tokenizer=tokenizer)

    # Load from Arrow data (paper's exact data)
    train_q, train_a = load_sdft_from_arrow(train_path, config.dataset)

    train_dataset = SDFTDataset(
        questions=train_q,
        golden_answers=train_a,
        batch_size=config.groups_per_batch,
        group_size=1,
        renderer=renderer,
        dataset_name=f"sdft_{config.dataset}",
    )

    sdft_config = sdft.Config(
        model_name=config.model_name,
        renderer_name=renderer_name,
        lora_rank=config.lora_rank,
        base_url=config.base_url,
        learning_rate=config.learning_rate,
        max_tokens=config.max_tokens,
        eval_every=config.eval_every,
        save_every=config.save_every,
        log_path=log_path,
        wandb_project=config.wandb_project,
        wandb_name=f"sdft-bench-sdft-{config.dataset}",
        max_steps=config.max_steps,
    )

    await sdft.main(sdft_config, train_dataset, test_dataset=None)

    ckpt = checkpoint_utils.get_last_checkpoint(log_path)
    if ckpt:
        logger.info(f"SDFT checkpoint: {ckpt.state_path}")
        return ckpt.state_path
    return None


async def run_eval_checkpoint(config: BenchmarkConfig) -> dict[str, float]:
    """Evaluate a specific checkpoint."""
    from tinker_cookbook.recipes.sdft.eval import run_eval

    _, eval_path, eval_dataset = _resolve_data_paths(config)
    renderer_name = await _resolve_renderer(config)

    if not config.checkpoint_path:
        raise ValueError("checkpoint_path must be set for eval_checkpoint phase")

    log_path = _make_log_path(config, "eval-checkpoint")
    output_path = f"{log_path}/eval_results.json"

    logger.info(f"=== Phase: Eval checkpoint on {config.dataset} ===")
    logger.info(f"Checkpoint: {config.checkpoint_path}")

    metrics = await run_eval(
        model_name=config.model_name,
        eval_dataset=eval_dataset,
        eval_data_path=eval_path,
        renderer_name=renderer_name,
        base_url=config.base_url,
        model_path=config.checkpoint_path,
        max_tokens=2048 if config.dataset == "science" else 1024,
        output_path=output_path,
    )

    _print_metrics("Checkpoint", metrics)
    return metrics


async def _resolve_renderer(config: BenchmarkConfig) -> str:
    return await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=config.model_name,
        explicit_renderer_name=config.renderer_name,
        load_checkpoint_path=None,
        base_url=config.base_url,
    )


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  {label} Results:")
    for k, v in metrics.items():
        if "accuracy" in k:
            logger.info(f"    {k}: {v:.4f} ({v * 100:.2f}%)")
        else:
            logger.info(f"    {k}: {v}")
    logger.info(f"{'=' * 60}\n")


async def cli_main(config: BenchmarkConfig) -> None:
    """Run the configured benchmark phase(s)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    results: dict[str, dict[str, float]] = {}

    if config.phase in ("all", "base_eval"):
        results["base"] = await run_base_eval(config)

    if config.phase in ("all", "sft"):
        sft_path = await run_sft(config)
        if sft_path:
            # Eval SFT checkpoint
            sft_eval_config = BenchmarkConfig(
                **{**chz.to_dict(config), "checkpoint_path": sft_path, "phase": "eval_checkpoint"}  # type: ignore[arg-type]
            )
            results["sft"] = await run_eval_checkpoint(sft_eval_config)

    if config.phase in ("all", "sdft"):
        sdft_path = await run_sdft(config)
        if sdft_path:
            sdft_eval_config = BenchmarkConfig(
                **{**chz.to_dict(config), "checkpoint_path": sdft_path, "phase": "eval_checkpoint"}  # type: ignore[arg-type]
            )
            results["sdft"] = await run_eval_checkpoint(sdft_eval_config)

    if config.phase == "eval_checkpoint":
        results["checkpoint"] = await run_eval_checkpoint(config)

    # Summary
    if len(results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("  BENCHMARK SUMMARY")
        logger.info("=" * 60)
        for label, metrics in results.items():
            acc_key = [k for k in metrics if "accuracy" in k]
            if acc_key:
                acc = metrics[acc_key[0]]
                logger.info(f"  {label:>12s}: {acc:.4f} ({acc * 100:.2f}%)")
        logger.info("=" * 60)

    # Save summary
    if config.log_root:
        summary_path = Path(config.log_root) / f"benchmark_summary_{config.dataset}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    config = chz.entrypoint(BenchmarkConfig)
    asyncio.run(cli_main(config))
