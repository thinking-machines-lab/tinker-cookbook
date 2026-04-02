"""GRPO baseline on SciKnowEval for comparison with SDPO.

Usage:
    python -m tinker_cookbook.recipes.sdpo.run_grpo_baseline \
        model_name=Qwen/Qwen3-8B \
        sciknoweval_domain=chemistry \
        groups_per_batch=32 \
        group_size=8 \
        learning_rate=1e-5 \
        max_tokens=8192
"""

import asyncio
from datetime import datetime

import chz

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.sdpo.sciknoweval_env import (
    SciKnowEvalDatasetBuilder,
    SciKnowEvalDomain,
)
from tinker_cookbook.rl.train import Config, main


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    sciknoweval_domain: SciKnowEvalDomain = "chemistry"
    seed: int = 0

    group_size: int = 8
    groups_per_batch: int = 32
    learning_rate: float = 1e-5
    max_tokens: int = 8192
    temperature: float = 1.0

    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    eval_every: int = 5
    save_every: int = 10

    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig) -> None:
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    model_slug = cli_config.model_name.replace("/", "-")
    run_name = (
        f"grpo-sciknoweval-{cli_config.sciknoweval_domain}-{model_slug}-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-seed{cli_config.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cli_config.log_path or f"/tmp/tinker-examples/sdpo/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    dataset_builder = SciKnowEvalDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        domain=cli_config.sciknoweval_domain,
        seed=cli_config.seed,
    )

    config = Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        dataset_builder=dataset_builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        lora_rank=cli_config.lora_rank,
        renderer_name=renderer_name,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
    )

    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
