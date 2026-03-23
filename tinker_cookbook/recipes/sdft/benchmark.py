"""
SDFT Benchmark Script.

Runs independent SFT and SDFT comparisons from the same base model.
All data is loaded from HuggingFace — no external repo needed.

Paper target numbers (Qwen2.5-7B-Instruct):
    SciKnowEval Chemistry L3: Base=32.1, SFT=66.2, SDFT=70.2
    ToolAlpaca Tool Use:       Base=42.9, SFT=63.2, SDFT=70.6

We use Qwen3-8B (different model), so exact numbers will differ,
but the pattern (SDFT > SFT) should hold.

Usage:
    # Run all comparisons (base eval, SFT, SDFT — independent from same base)
    python -m tinker_cookbook.recipes.sdft.benchmark \
        dataset=sciknoweval

    # Eval base model only
    python -m tinker_cookbook.recipes.sdft.benchmark \
        dataset=sciknoweval phase=base_eval

    # SFT training only (independent from SDFT)
    python -m tinker_cookbook.recipes.sdft.benchmark \
        dataset=sciknoweval phase=sft

    # SDFT training only (independent from SFT)
    python -m tinker_cookbook.recipes.sdft.benchmark \
        dataset=sciknoweval phase=sdft

    # Eval an existing checkpoint
    python -m tinker_cookbook.recipes.sdft.benchmark \
        dataset=sciknoweval phase=eval_checkpoint \
        checkpoint_path=tinker://...
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import chz
import tinker

from tinker_cookbook import checkpoint_utils, cli_utils, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

Phase = Literal["all", "base_eval", "sft", "sdft", "eval_checkpoint"]


@chz.chz
class BenchmarkConfig:
    """Configuration for running SDFT benchmarks."""

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 128
    renderer_name: str | None = None

    # Benchmark dataset
    dataset: str = "sciknoweval"  # sciknoweval | toolalpaca
    sciknoweval_domain: str = "Chemistry"

    # Which phase to run. SFT and SDFT are independent comparisons from the
    # same base model (not a sequential pipeline).
    phase: Phase = "all"

    # For eval_checkpoint phase: path to evaluate
    checkpoint_path: str | None = None

    # SDFT training hyperparameters (paper defaults)
    learning_rate: float = 2e-5
    groups_per_batch: int = 32
    max_tokens: int = 2048
    max_steps: int | None = None
    eval_every: int = 20
    save_every: int = 20

    # SFT training hyperparameters
    sft_learning_rate: float = 2e-5
    sft_batch_size: int = 32
    sft_max_steps: int | None = None

    # Logging
    log_root: str | None = None
    wandb_project: str | None = None
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def _make_log_path(config: BenchmarkConfig, phase: str) -> str:
    if config.log_root:
        root = config.log_root
    else:
        root = "/tmp/tinker-examples/sdft-benchmark"
    model_slug = config.model_name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    return f"{root}/{config.dataset}-{model_slug}-{phase}-{timestamp}"


# ---------------------------------------------------------------------------
# Phase: Base eval
# ---------------------------------------------------------------------------


async def run_base_eval(config: BenchmarkConfig) -> dict[str, float]:
    """Evaluate the base model (no training)."""
    renderer_name = await _resolve_renderer(config)
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    log_path = _make_log_path(config, "base-eval")

    logger.info(f"=== Phase: Base model eval on {config.dataset} ===")
    logger.info(f"Model: {config.model_name}")

    service_client = tinker.ServiceClient(base_url=config.base_url)
    sampling_client = service_client.create_sampling_client(base_model=config.model_name)

    evaluator = _build_evaluator(config, renderer)
    metrics = await evaluator(sampling_client)

    _print_metrics("Base model", metrics)
    _save_results(log_path, metrics, config, model_path=None)
    return metrics


# ---------------------------------------------------------------------------
# Phase: SFT
# ---------------------------------------------------------------------------


async def run_sft(config: BenchmarkConfig) -> str | None:
    """Run SFT training from the base model and return checkpoint path."""
    from tinker_cookbook.recipes.sdft.datasets import SciKnowEvalSFTBuilder
    from tinker_cookbook.supervised import train as sl_train
    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    renderer_name = await _resolve_renderer(config)

    log_path = _make_log_path(config, "sft")
    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)

    logger.info(f"=== Phase: SFT training on {config.dataset} ===")

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        max_length=2048,
        batch_size=config.sft_batch_size,
    )

    if config.dataset == "sciknoweval":
        dataset_builder = SciKnowEvalSFTBuilder(
            common_config=common_config,
            domain=config.sciknoweval_domain,
        )
    else:
        raise ValueError(f"SFT benchmark not yet implemented for dataset: {config.dataset}")

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
        max_steps=config.sft_max_steps,
        dataset_builder=dataset_builder,
    )

    await sl_train.main(sl_config)

    ckpt = checkpoint_utils.get_last_checkpoint(log_path)
    if ckpt and ckpt.state_path:
        logger.info(f"SFT checkpoint: {ckpt.state_path}")
        return ckpt.state_path
    return None


# ---------------------------------------------------------------------------
# Phase: SDFT
# ---------------------------------------------------------------------------


async def run_sdft(config: BenchmarkConfig) -> str | None:
    """Run SDFT training from the base model and return checkpoint path."""
    from tinker_cookbook.distillation import sdft
    from tinker_cookbook.recipes.sdft.datasets import SDFTDataset

    renderer_name = await _resolve_renderer(config)
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    log_path = _make_log_path(config, "sdft")
    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)

    logger.info(f"=== Phase: SDFT training on {config.dataset} ===")

    train_q, train_a, _, _ = _load_data(config)

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
    if ckpt and ckpt.state_path:
        logger.info(f"SDFT checkpoint: {ckpt.state_path}")
        return ckpt.state_path
    return None


# ---------------------------------------------------------------------------
# Phase: Eval checkpoint
# ---------------------------------------------------------------------------


async def run_eval_checkpoint(config: BenchmarkConfig) -> dict[str, float]:
    """Evaluate a specific checkpoint."""
    renderer_name = await _resolve_renderer(config)
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    if not config.checkpoint_path:
        raise ValueError("checkpoint_path must be set for eval_checkpoint phase")

    log_path = _make_log_path(config, "eval-checkpoint")

    logger.info(f"=== Phase: Eval checkpoint on {config.dataset} ===")
    logger.info(f"Checkpoint: {config.checkpoint_path}")

    service_client = tinker.ServiceClient(base_url=config.base_url)
    sampling_client = service_client.create_sampling_client(
        base_model=config.model_name, model_path=config.checkpoint_path
    )

    evaluator = _build_evaluator(config, renderer)
    metrics = await evaluator(sampling_client)

    _print_metrics("Checkpoint", metrics)
    _save_results(log_path, metrics, config, model_path=config.checkpoint_path)
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_data(
    config: BenchmarkConfig,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load train and test data from HuggingFace."""
    from tinker_cookbook.recipes.sdft.datasets import load_sciknoweval, load_toolalpaca

    if config.dataset == "sciknoweval":
        return load_sciknoweval(domain=config.sciknoweval_domain)
    elif config.dataset == "toolalpaca":
        return load_toolalpaca()
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}. Options: sciknoweval, toolalpaca")


def _build_evaluator(
    config: BenchmarkConfig,
    renderer: renderers.Renderer,
):  # type: ignore[return]
    """Build the appropriate evaluator for the dataset."""
    from tinker_cookbook.recipes.sdft.eval import SciKnowEvalEvaluator, ToolUseEvaluator

    _, _, test_q, test_a = _load_data(config)

    if config.dataset == "sciknoweval":
        # SciKnowEval eval: test questions are plain strings, wrap as user messages
        prompts = [[{"role": "user", "content": q}] for q in test_q]
        return SciKnowEvalEvaluator(prompts, test_a, renderer, max_tokens=2048)
    elif config.dataset == "toolalpaca":
        return ToolUseEvaluator(test_q, test_a, renderer, max_tokens=1024)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")


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


def _save_results(
    log_path: str,
    metrics: dict[str, float],
    config: BenchmarkConfig,
    model_path: str | None,
) -> None:
    Path(log_path).mkdir(parents=True, exist_ok=True)
    output_path = Path(log_path) / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "config": {
                    "model_name": config.model_name,
                    "model_path": model_path,
                    "dataset": config.dataset,
                },
            },
            f,
            indent=2,
        )
    logger.info(f"Saved results to {output_path}")


async def cli_main(config: BenchmarkConfig) -> None:
    """Run the configured benchmark phase(s)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    results: dict[str, dict[str, float]] = {}

    if config.phase in ("all", "base_eval"):
        results["base"] = await run_base_eval(config)

    if config.phase in ("all", "sft"):
        sft_path = await run_sft(config)
        if sft_path:
            sft_eval_config = BenchmarkConfig(
                **{**chz.asdict(config), "checkpoint_path": sft_path, "phase": "eval_checkpoint"}  # type: ignore[arg-type]
            )
            results["sft"] = await run_eval_checkpoint(sft_eval_config)

    if config.phase in ("all", "sdft"):
        sdft_path = await run_sdft(config)
        if sdft_path:
            sdft_eval_config = BenchmarkConfig(
                **{**chz.asdict(config), "checkpoint_path": sdft_path, "phase": "eval_checkpoint"}  # type: ignore[arg-type]
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
    log_root = config.log_root or "/tmp/tinker-examples/sdft-benchmark"
    summary_path = Path(log_root) / f"benchmark_summary_{config.dataset}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    config = chz.entrypoint(BenchmarkConfig)
    asyncio.run(cli_main(config))
