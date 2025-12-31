import asyncio
import random
from collections import defaultdict
from typing import Literal, TypedDict

import chz
import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.search_tool.search_env import (
    SearchEnv,
    SearchR1Datum,
    download_search_r1_dataset,
)
from tinker_cookbook.recipes.search_tool.tools import (
    ChromaToolClient,
    ChromaToolClientConfig,
    EmbeddingConfig,
    RetrievalConfig,
)
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer

ROLLOUT_CONCURRENCY = 1024
rollout_semaphore = asyncio.Semaphore(ROLLOUT_CONCURRENCY)


@chz.chz
class CLIConfig:
    # Evaluation parameters
    max_eval_samples: int = chz.field(
        default=100, doc="Maximum number of samples to evaluate per data source"
    )
    seed: int = chz.field(default=42, doc="Random seed for sampling")
    split: Literal["train", "test"] = chz.field(default="test", doc="Dataset split to use")

    # Model parameters
    base_model: str = chz.field(default="Qwen/Qwen3-4B-Instruct-2507", doc="Base model to use")
    tinker_checkpoint_url: str = chz.field(doc="Tinker checkpoint URL (required)")
    max_tokens: int = chz.field(default=1024, doc="Maximum number of tokens to generate")


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


async def evaluate_one_dataset(data: list[SearchR1Datum], config: CLIConfig):
    # load model and renderer
    tokenizer = get_tokenizer(config.base_model)
    renderer_name = model_info.get_recommended_renderer_name(config.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=config.tinker_checkpoint_url)
    policy = TinkerTokenCompleter(sampling_client, max_tokens=config.max_tokens)

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
    tasks = [evaluate_single_item(item, chroma_tool_client, policy, renderer) for item in data]

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


async def cli_main(config: CLIConfig):
    # Download the data
    print(f"Downloading {config.split} split...")
    data = download_search_r1_dataset(config.split)
    print(f"Total data points: {len(data)}")

    # Split by data source
    data_by_source = split_data_by_source(data)
    print(f"\nData sources found: {list(data_by_source.keys())}")
    print("Original distribution:")
    for source, items in data_by_source.items():
        print(f"  {source}: {len(items)}")

    # Sample K from each source
    print(f"\nSampling up to {config.max_eval_samples} samples from each source...")
    sampled_data_by_source = sample_k_from_each_source(
        data_by_source, config.max_eval_samples, config.seed
    )

    # Collect results from all datasets
    dataset_results = {}
    for source, data in sampled_data_by_source.items():
        print(f"Evaluating {source}...")
        result = await evaluate_one_dataset(data, config)
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
    config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(config))
