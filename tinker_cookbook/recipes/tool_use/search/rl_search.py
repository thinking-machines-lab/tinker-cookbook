import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.tool_use.search.search_env import SearchR1DatasetBuilder
from tinker_cookbook.recipes.tool_use.search.tools import (
    ChromaToolClientConfig,
    EmbeddingConfig,
    RetrievalConfig,
)
from tinker_cookbook.rl import train


def build_config(
    model_name: str,
    seed: int,
    learning_rate: float,
    batch_size: int,
    lora_rank: int,
    chroma_port: int,
    wandb_project: str,
    stream_minibatch: bool,
    max_trajectory_tokens: int,
    log_path: str | None,
) -> train.Config:
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    chroma_tool_config = ChromaToolClientConfig(
        chroma_host="localhost",
        chroma_port=chroma_port,
        chroma_collection_name="wiki_embeddings",
        retrieval_config=RetrievalConfig(
            n_results=3,
            embedding_config=EmbeddingConfig(
                model_name="gemini-embedding-001",
                embedding_dim=768,
            ),
        ),
    )

    builder = SearchR1DatasetBuilder(
        batch_size=batch_size,
        group_size=8,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
        chroma_tool_config=chroma_tool_config,
        seed=seed,
        max_trajectory_tokens=max_trajectory_tokens,
    )
    if stream_minibatch:
        stream_minibatch_config = train.StreamMinibatchConfig(
            groups_per_batch=batch_size,
            num_minibatches=4,
        )
        bs_str = f"bs{batch_size}_stream"
    else:
        stream_minibatch_config = None
        bs_str = f"bs{batch_size}"

    run_name = f"search_r1_{model_name.lower()}_{bs_str}_gs8_seed{seed}_tracj{max_trajectory_tokens // 1024}k_lr{learning_rate}_rank{lora_rank}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    if log_path is None:
        log_path = f"/tmp/tinker-examples/rl_search/{run_name}"

    if not Path("/tmp").exists():
        raise ValueError("/tmp does not exist")

    return train.Config(
        model_name=model_name,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=learning_rate,
        max_tokens=1024,
        eval_every=0,
        wandb_project=wandb_project,
        wandb_name=run_name,
        lora_rank=lora_rank,
        stream_minibatch_config=stream_minibatch_config if stream_minibatch else None,
    )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train RL model for HotpotQA search")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model name (default: Qwen/Qwen3-4B-Instruct-2507)",
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Random seed for reproducibility (default: 2)"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=32, help="Lora rank for training (default: 32)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for training (default: 512)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-5,
        help="Learning rate for training (default: 4e-5)",
    )
    parser.add_argument("--chroma_port", type=int, default=8000, help="Chroma port (default: 8000)")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tinker_cookbook_retrieval",
        help="Wandb project (default: tinker_cookbook_retrieval)",
    )
    parser.add_argument(
        "--stream_minibatch", action="store_true", help="Stream minibatch (default: False)"
    )
    parser.add_argument(
        "--max_trajectory_tokens",
        type=int,
        default=8 * 1024,
        help="Max trajectory tokens (default: 8 * 1024)",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Load checkpoint path (default: None)",
    )
    args = parser.parse_args()

    config = build_config(
        args.model_name,
        args.seed,
        args.learning_rate,
        args.batch_size,
        args.lora_rank,
        args.chroma_port,
        args.wandb_project,
        args.stream_minibatch,
        args.max_trajectory_tokens,
        args.log_path,
    )
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
