"""Lightweight local CLI for the Search-R1 recipe.

A laptop-friendly alternative to ``train.py``: no Chroma service, no Gemini
embeddings, no 160 GB RAM. Uses an in-memory BM25 index over a small Wikipedia
subset. Retrieval quality is lower than the full setup, so expect smaller
accuracy gains, but the training loop is identical and runs end-to-end on
standard hardware.
"""

import asyncio
from datetime import datetime
from pathlib import Path

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.search_tool.search_env_local import LocalSearchR1DatasetBuilder
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    # Model parameters
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None

    # Training parameters
    learning_rate: float = 4e-5
    batch_size: int = 64
    seed: int = 2
    max_tokens: int = 1024
    eval_every: int = 0

    # Dataset parameters
    group_size: int = 8
    max_turns: int = 5
    format_coef: float = 0.1
    max_trajectory_tokens: int = 16 * 1024

    # Local search configuration
    n_results: int = 3
    max_train_examples: int | None = 2048

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    max_steps: int | None = None


async def cli_main(cli_config: CLIConfig) -> None:
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    builder = LocalSearchR1DatasetBuilder(
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=cli_config.model_name,
        n_results=cli_config.n_results,
        seed=cli_config.seed,
        max_turns=cli_config.max_turns,
        format_coef=cli_config.format_coef,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_train_examples=cli_config.max_train_examples,
    )

    model_name_short = cli_config.model_name.lower().replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"search_r1_local_{model_name_short}_bs{cli_config.batch_size}_"
        f"gs{cli_config.group_size}_seed{cli_config.seed}_"
        f"lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_{date_and_time}"
    )

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/rl_search_local/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    config = train.Config(
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        max_steps=cli_config.max_steps,
    )

    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
