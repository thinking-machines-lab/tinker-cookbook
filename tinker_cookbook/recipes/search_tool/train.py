"""
CLI for Search-R1 replication
"""

import asyncio
from datetime import datetime
from pathlib import Path

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.search_tool.search_env import SearchR1DatasetBuilder
from tinker_cookbook.recipes.search_tool.tools import (
    ChromaToolClientConfig,
    EmbeddingConfig,
    RetrievalConfig,
)
from tinker_cookbook.rl import train


@chz.chz
class CLIConfig:
    # Model parameters
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None

    # Training parameters
    learning_rate: float = 4e-5
    batch_size: int = 512
    seed: int = 2
    max_tokens: int = 1024
    eval_every: int = 0

    # Dataset parameters
    group_size: int = 8
    max_trajectory_tokens: int = 8 * 1024

    # Chroma configuration
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection_name: str = "wiki_embeddings"
    n_results: int = 3
    embedding_model_name: str = "gemini-embedding-001"
    embedding_dim: int = 768

    # Streaming configuration
    stream_minibatch: bool = False
    num_minibatches: int = 4

    # Logging parameters
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    # Build chroma tool config
    chroma_tool_config = ChromaToolClientConfig(
        chroma_host=cli_config.chroma_host,
        chroma_port=cli_config.chroma_port,
        chroma_collection_name=cli_config.chroma_collection_name,
        retrieval_config=RetrievalConfig(
            n_results=cli_config.n_results,
            embedding_config=EmbeddingConfig(
                model_name=cli_config.embedding_model_name,
                embedding_dim=cli_config.embedding_dim,
            ),
        ),
    )

    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Build dataset builder
    builder = SearchR1DatasetBuilder(
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        renderer_name=renderer_name,
        model_name_for_tokenizer=cli_config.model_name,
        chroma_tool_config=chroma_tool_config,
        seed=cli_config.seed,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
    )

    # Configure streaming minibatch
    if cli_config.stream_minibatch:
        stream_minibatch_config = train.StreamMinibatchConfig(
            groups_per_batch=cli_config.batch_size,
            num_minibatches=cli_config.num_minibatches,
        )
        bs_str = f"bs{cli_config.batch_size}_stream"
    else:
        stream_minibatch_config = None
        bs_str = f"bs{cli_config.batch_size}"

    # Build run name
    model_name_short = cli_config.model_name.lower().replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"search_r1_{model_name_short}_{bs_str}_gs{cli_config.group_size}_seed{cli_config.seed}_tracj{cli_config.max_trajectory_tokens // 1024}k_lr{cli_config.learning_rate}_rank{cli_config.lora_rank}_{date_and_time}"

    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/rl_search/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    # Validate /tmp exists
    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")

    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Build training config
    config = train.Config(
        model_name=cli_config.model_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        eval_every=cli_config.eval_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        lora_rank=cli_config.lora_rank,
        stream_minibatch_config=stream_minibatch_config,
    )

    # Run training
    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
