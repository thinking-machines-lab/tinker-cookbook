import argparse
import asyncio
import random
from collections import defaultdict
from typing import TypedDict

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.tool_use.search.search_env import (
    SearchEnv,
    SearchR1Datum,
    download_search_r1_dataset,
)
from tinker_cookbook.recipes.tool_use.search.tools import (
    ChromaToolClient,
    ChromaToolClientConfig,
    EmbeddingConfig,
    RetrievalConfig,
)
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer

ROLLOUT_CONCURRENCY = 1024
rollout_semaphore = asyncio.Semaphore(ROLLOUT_CONCURRENCY)


class EvaluationResult(TypedDict):
    question: str
    correct_score: float
    trajectory: object


def split_data_by_source(data: list[SearchR1Datum]) -> dict[str, list[SearchR1Datum]]:
    """Split data by data source."""
    data_by_source = defaultdict(list)
    for item in data:
        data_by_source[item["data_source"]].append(item)
    return dict(data_by_source)


def sample_k_from_each_source(
    data_by_source: dict[str, list[SearchR1Datum]], k: int, seed: int = 42
) -> dict[str, list[SearchR1Datum]]:
    """Sample K items from each data source."""
    random.seed(seed)
    sampled_data = {}
    total_samples = 0

    for source, items in data_by_source.items():
        if len(items) <= k:
            sampled_data[source] = items
        else:
            sampled_data[source] = random.sample(items, k)
        total_samples += len(sampled_data[source])
        print(f"{source}: {len(items)} -> {len(sampled_data[source])} samples")

    print(f"Total samples: {total_samples}")
    return sampled_data


async def evaluate_single_item(
    item: SearchR1Datum,
    args: argparse.Namespace,
    chroma_tool_client: ChromaToolClient,
    policy: TinkerTokenCompleter,
    renderer: renderers.Renderer,
) -> EvaluationResult:
    env = SearchEnv(
        item["question"],
        item["answer"],
        chroma_tool_client,
        renderer,
        convo_prefix=SearchEnv.standard_fewshot_prefix(
            renderer, chroma_tool_client.get_tool_schemas()
        ),
    )
    async with rollout_semaphore:
        trajectory = await do_single_rollout(policy, env)

    # Extract correct metric from the last transition
    correct_score = 0.0
    if trajectory.transitions:
        correct_score = trajectory.transitions[-1].metrics.get("correct", 0.0)

    return {"question": item["question"], "correct_score": correct_score, "trajectory": trajectory}


async def evaluate_one_dataset(data: list[SearchR1Datum], args: argparse.Namespace):
    # load model and renderer
    tokenizer = get_tokenizer(args.base_model)
    renderer_name = model_info.get_recommended_renderer_name(args.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=args.tinker_checkpoint_url)
    policy = TinkerTokenCompleter(sampling_client, max_tokens=args.max_tokens)

    chroma_config = ChromaToolClientConfig(
        chroma_host="localhost",
        chroma_port=8000,
        chroma_collection_name="wiki_embeddings",
        retrieval_config=RetrievalConfig(
            n_results=3,
            embedding_config=EmbeddingConfig(
                model_name="gemini-embedding-001",
                embedding_dim=768,
            ),
        ),
    )
    chroma_tool_client = await ChromaToolClient.create(chroma_config)

    # Run evaluations in parallel using asyncio.gather
    tasks = [
        evaluate_single_item(item, args, chroma_tool_client, policy, renderer) for item in data
    ]

    print(f"Evaluating {len(tasks)} items")
    results = await asyncio.gather(*tasks)

    # Aggregate results
    correct_scores = [result["correct_score"] for result in results]

    if correct_scores:
        total_correct = sum(correct_scores)
        accuracy = total_correct / len(correct_scores)
        return {
            "total_samples": len(correct_scores),
            "total_correct": total_correct,
            "accuracy": accuracy,
        }

    return {"total_samples": 0, "total_correct": 0, "accuracy": 0.0}


async def main():
    parser = argparse.ArgumentParser(description="Offline evaluation with data source sampling")
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate per data source",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--split", choices=["train", "test"], default="test", help="Dataset split to use"
    )
    parser.add_argument(
        "--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Base model to use"
    )
    parser.add_argument(
        "--tinker-checkpoint-url", type=str, required=True, help="Tinker checkpoint URL"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate"
    )

    args = parser.parse_args()

    # Download the data
    print(f"Downloading {args.split} split...")
    data = download_search_r1_dataset(args.split)
    print(f"Total data points: {len(data)}")

    # Split by data source
    data_by_source = split_data_by_source(data)
    print(f"\nData sources found: {list(data_by_source.keys())}")
    print("Original distribution:")
    for source, items in data_by_source.items():
        print(f"  {source}: {len(items)}")

    # Sample K from each source
    print(f"\nSampling up to {args.max_eval_samples} samples from each source...")
    sampled_data_by_source = sample_k_from_each_source(
        data_by_source, args.max_eval_samples, args.seed
    )

    # Collect results from all datasets
    dataset_results = {}
    for source, data in sampled_data_by_source.items():
        print(f"Evaluating {source}...")
        result = await evaluate_one_dataset(data, args)
        dataset_results[source] = result

    # Print results table
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Accuracy':<10} {'Correct':<10} {'Total':<10}")
    print("-" * 80)

    total_all_correct = 0
    total_all_samples = 0

    for dataset, result in dataset_results.items():
        accuracy = result["accuracy"]
        correct = result["total_correct"]
        total = result["total_samples"]
        total_all_correct += correct
        total_all_samples += total
        print(f"{dataset:<15} {accuracy:<10.3f} {correct:<10.0f} {total:<10}")

    if total_all_samples > 0:
        overall_accuracy = total_all_correct / total_all_samples
        print("-" * 80)
        print(
            f"{'OVERALL':<15} {overall_accuracy:<10.3f} {total_all_correct:<10.0f} {total_all_samples:<10}"
        )
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
