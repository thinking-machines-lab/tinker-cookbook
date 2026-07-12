"""
Self-Distillation Fine-Tuning (SDFT) recipe.

Implements SDFT from "Self-Distillation Enables Continual Learning"
(arxiv 2601.19897). A teacher model conditioned on golden demonstrations
provides per-token KL signals to train a student model.

Example usage:
    # SciKnowEval (paper's science benchmark)
    python -m tinker_cookbook.recipes.sdft.train \
        model_name=Qwen/Qwen3-8B \
        dataset=sciknoweval \
        groups_per_batch=32 \
        learning_rate=2e-5

    # ToolAlpaca (paper's tool-use benchmark)
    python -m tinker_cookbook.recipes.sdft.train \
        model_name=Qwen/Qwen3-8B \
        dataset=toolalpaca \
        groups_per_batch=32 \
        learning_rate=2e-5

    # Debug run (small batch)
    python -m tinker_cookbook.recipes.sdft.train \
        groups_per_batch=4 group_size=1 \
        max_tokens=256 max_steps=5
"""

import asyncio
import contextlib
import logging
from datetime import datetime
from pathlib import Path
from typing import cast

import chz

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.distillation import sdft

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for SDFT training."""

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 128
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Dataset
    dataset: str = "sciknoweval"  # sciknoweval | toolalpaca
    sciknoweval_domain: str = "Chemistry"
    toolalpaca_data_path: str | None = None  # Local Arrow path for ToolAlpaca

    # Training
    group_size: int = 1  # Paper uses num_generations=1
    groups_per_batch: int = 32  # Paper: gradient_accumulation=32
    learning_rate: float = 2e-5
    max_tokens: int = 2048
    temperature: float = 1.0

    # SDFT-specific
    topk: int = 20
    reverse: bool = False
    teacher_sync_every: int | None = None
    max_context_length: int = 32768

    # Optimizer
    num_substeps: int = 1

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Evaluation and checkpointing
    eval_every: int = 20
    save_every: int = 20

    # Service
    base_url: str | None = None
    max_steps: int | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Optional token DB capture of the student's on-policy samples (prompt,
    # sampled tokens, logprobs) for inspection with tinker_cookbook.tokendb.
    # Requires the tokendb extra: pip install 'tinker-cookbook[tokendb]'
    token_db_path: str | None = None


def _token_db_attrs(cli_config: CLIConfig) -> dict[str, str]:
    """Attrs stamped onto every captured sample row (token_db_path set).

    SDFT is self-distillation: the teacher is the same base model as the
    student (conditioned on golden demonstrations), so both attrs record
    ``model_name``.
    """
    return {
        "student_model": cli_config.model_name,
        "teacher_model": cli_config.model_name,
        "dataset": cli_config.dataset,
    }


async def cli_main(cli_config: CLIConfig) -> None:
    """Convert CLI config to full config and run SDFT training."""
    from tinker_cookbook.recipes.sdft.datasets import (
        SciKnowEvalSDFTBuilder,
        ToolAlpacaSDFTBuilder,
    )

    # Resolve renderer name
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    # Create log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_slug = cli_config.model_name.replace("/", "-")
        run_name = (
            f"sdft-{cli_config.dataset}-{model_slug}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = f"/tmp/tinker-examples/sdft/{run_name}"

    wandb_name = cli_config.wandb_name or Path(log_path).name

    # Build dataset
    if cli_config.dataset == "sciknoweval":
        builder = SciKnowEvalSDFTBuilder(
            groups_per_batch=cli_config.groups_per_batch,
            group_size=cli_config.group_size,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            domain=cli_config.sciknoweval_domain,
        )
    elif cli_config.dataset == "toolalpaca":
        builder = ToolAlpacaSDFTBuilder(
            groups_per_batch=cli_config.groups_per_batch,
            group_size=cli_config.group_size,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            data_path=cli_config.toolalpaca_data_path,
        )
    else:
        raise ValueError(f"Unknown dataset: {cli_config.dataset}. Options: sciknoweval, toolalpaca")

    train_dataset, test_dataset = await builder()

    # Build config
    config = sdft.Config(
        model_name=cli_config.model_name,
        recipe_name="recipe_sdft",
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        base_url=cli_config.base_url,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        topk=cli_config.topk,
        reverse=cli_config.reverse,
        teacher_sync_every=cli_config.teacher_sync_every,
        max_context_length=cli_config.max_context_length,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        log_path=log_path,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        max_steps=cli_config.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training, optionally capturing the student's samples to a token DB.
    # (Only the student's on-policy generations go through the completers;
    # the teacher's forced-decode logprob probes are not captured.)
    with contextlib.ExitStack() as stack:
        if cli_config.token_db_path is not None:
            from tinker_cookbook.tokendb import (
                ActiveCapture,
                TokenDbWriter,
                capture_samples,
                check_token_db_dependencies,
            )
            from tinker_cookbook.tokendb.capture import SupportsDecode
            from tinker_cookbook.tokenizer_utils import get_tokenizer

            check_token_db_dependencies()
            writer = stack.enter_context(
                TokenDbWriter(
                    cli_config.token_db_path,
                    context={
                        "recipe_name": "sdft/train",
                        "model_name": cli_config.model_name,
                    },
                )
            )
            # The HF tokenizer's decode(list[int]) returns str at runtime.
            tokenizer = cast(SupportsDecode, get_tokenizer(cli_config.model_name))
            stack.enter_context(
                capture_samples(
                    ActiveCapture(writer, tokenizer=tokenizer),
                    attrs=_token_db_attrs(cli_config),
                )
            )
        await sdft.main(config, train_dataset, test_dataset)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
