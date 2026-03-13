"""

## VLM Image Classifier

Launcher for evaluating trained image classifiers.

```bash
python -m tinker_cookbook.recipes.vlm_classifier.eval_sweep \
    experiment_dir=$HOME/tinker-experiments output_file=results.json
```

With early stopping (use best checkpoint per run based on validation accuracy):

```bash
python -m tinker_cookbook.recipes.vlm_classifier.eval_sweep \
    experiment_dir=$HOME/tinker-experiments output_file=results.json
```

"""

import asyncio
import json
import logging
import os
import re
from typing import Any

import chz
import tinker

from tinker_cookbook.checkpoint_utils import get_last_checkpoint, load_checkpoints_file
from tinker_cookbook.recipes.vlm_classifier.eval import get_evaluator_builder


# Set up logger
logger = logging.getLogger(__name__)


def get_checkpoint_at_step(
    log_dir: str,
    step: int,
    required_key: str = "sampler_path",
) -> dict[str, Any] | None:
    """
    Get the checkpoint at a specific step from the checkpoints.jsonl file.

    Args:
        log_dir: The directory containing checkpoints.jsonl.
        step: The step number to find.
        required_key: The key to check for in the checkpoint.

    Returns:
        The checkpoint at the specified step, or None if not found.
    """
    checkpoints = load_checkpoints_file(log_dir)
    for checkpoint in checkpoints:
        if checkpoint.get("batch") == step and required_key in checkpoint:
            logger.info(f"Found checkpoint at step {step}: {checkpoint}")
            return checkpoint
    logger.warning(f"No checkpoint found at step {step} with key '{required_key}' in {log_dir}")
    return None


def parse_hyperparams_from_experiment_name(experiment_name: str) -> dict[str, Any]:
    """
    Parse hyperparameters from the experiment directory name.

    Experiment names follow the format from sweep.py:
    {dataset}-{model_name}-{lora_rank}rank-{learning_rate}lr-{batch_size}batch-{examples_per_class}shot-seed{subset_seed}-{date}

    Example: caltech101-Qwen-Qwen3-VL-235B-A22B-Instruct-32rank-0.0005lr-32batch-4shot-seed0-2025-11-26
    """

    hyperparams: dict[str, Any] = {}

    # Parse dataset: first segment before the first dash
    hyperparams["dataset"] = experiment_name.split("-")[0]

    # Parse lora_rank: look for pattern like "32rank"
    if match := re.search(r"-(\d+)rank-", experiment_name):
        hyperparams["lora_rank"] = int(match.group(1))

    # Parse learning_rate: look for pattern like "0.0005lr" or "5e-4lr"
    if match := re.search(r"-([\d.e+-]+)lr-", experiment_name):
        hyperparams["learning_rate"] = float(match.group(1))

    # Parse batch_size: look for pattern like "32batch"
    if match := re.search(r"-(\d+)batch", experiment_name):
        hyperparams["batch_size"] = int(match.group(1))

    # Parse examples_per_class (shot): look for pattern like "4shot"
    if match := re.search(r"-(\d+)shot-", experiment_name):
        hyperparams["examples_per_class"] = int(match.group(1))

    # Parse subset_seed: look for pattern like "seed0"
    if match := re.search(r"-seed(\d+)-", experiment_name):
        hyperparams["subset_seed"] = int(match.group(1))

    # Parse date: look for pattern like "2025-11-26" at the end
    if match := re.search(r"-(\d{4}-\d{2}-\d{2})$", experiment_name):
        hyperparams["date"] = match.group(1)

    return hyperparams


@chz.chz
class EvalConfig:
    """
    Config for evaluating all experiments in a sweep directory.
    """

    experiment_dir: str
    output_file: str

    renderer_name: str = "qwen3_vl"
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"

    # Infrastructure parameters
    base_url: str | None = None

    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1

    n_eval: int | None = None
    max_parallel_tasks: int = 1024
    max_parallel_evals: int = 5

    max_image_size: int = 480

    # Early stopping: map experiment name to the step of the best checkpoint
    # If not provided or experiment not in dict, uses the last checkpoint
    early_stopping_checkpoints: dict[str, int] | None = None


async def evaluate_experiment(
    experiment_name: str,
    eval_config: EvalConfig,
    service_client: tinker.ServiceClient,
) -> dict[str, Any]:
    """
    Evaluate a single few-shot image classifier experiment.
    """

    experiment_path = os.path.join(eval_config.experiment_dir, experiment_name)
    assert os.path.isdir(experiment_path), f"Experiment directory does not exist: {experiment_path}"

    # Load checkpoint: use early stopping step if provided, otherwise use last checkpoint
    early_stop_step = (
        eval_config.early_stopping_checkpoints.get(experiment_name)
        if eval_config.early_stopping_checkpoints
        else None
    )

    if early_stop_step is not None:
        checkpoint = get_checkpoint_at_step(
            experiment_path, early_stop_step, required_key="sampler_path"
        )
        assert checkpoint is not None, (
            f"No checkpoint at step {early_stop_step} with sampler_path found in {experiment_path}"
        )
        logger.info(
            f"Using early stopping checkpoint at step {early_stop_step} for {experiment_name}"
        )
    else:
        checkpoint = get_last_checkpoint(experiment_path, required_key="sampler_path")
        assert checkpoint is not None, f"No checkpoint with sampler_path found in {experiment_path}"

    # Parse hyperparameters (including dataset) from directory name
    hyperparams = parse_hyperparams_from_experiment_name(experiment_name)
    assert "dataset" in hyperparams, f"Unable to parse the dataset name from {experiment_path}"

    # Create evaluator for this dataset
    evaluator_builder = get_evaluator_builder(
        dataset=hyperparams["dataset"],
        model_name_for_tokenizer=eval_config.model_name,
        renderer_name=eval_config.renderer_name,
        temperature=eval_config.temperature,
        max_tokens=eval_config.max_tokens,
        top_p=eval_config.top_p,
        top_k=eval_config.top_k,
        n_eval=eval_config.n_eval,
        max_parallel_tasks=eval_config.max_parallel_tasks,
        max_image_size=eval_config.max_image_size,
    )

    sampling_client = service_client.create_sampling_client(model_path=checkpoint["sampler_path"])
    metrics = await evaluator_builder()(sampling_client)  # type: ignore[arg-type]
    return {
        "experiment_name": experiment_name,
        "checkpoint_step": checkpoint.get("step"),
        **metrics,
        **hyperparams,
    }


async def evaluate_sweep(
    eval_config: EvalConfig,
    experiment_names: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Evaluate all few-shot image classifier experiments in a sweep directory.
    """

    service_client = tinker.ServiceClient(base_url=eval_config.base_url)

    # Limit concurrent evaluation tasks
    semaphore = asyncio.Semaphore(eval_config.max_parallel_evals)

    async def bounded_evaluate_experiment(experiment_name: str) -> dict[str, Any]:
        async with semaphore:
            return await evaluate_experiment(
                experiment_name=experiment_name,
                eval_config=eval_config,
                service_client=service_client,
            )

    # Evaluate all experiments in parallel (bounded by semaphore)
    logger.info(
        f"Submitting {len(experiment_names)} eval tasks (max {eval_config.max_parallel_evals} parallel)"
    )
    async_tasks = [
        asyncio.create_task(bounded_evaluate_experiment(name)) for name in experiment_names
    ]

    results = await asyncio.gather(*async_tasks)
    return {metrics["experiment_name"]: metrics for metrics in results}


def run_eval_sweep(eval_config: EvalConfig):
    """
    Evaluate all few-shot image classifier experiments in a sweep directory.
    """

    logging.basicConfig(level=logging.INFO)

    if not os.path.isdir(eval_config.experiment_dir):
        raise ValueError(f"Experiment directory does not exist: {eval_config.experiment_dir}")

    # Find all experiment subdirectories
    experiment_names = sorted(
        [
            d
            for d in os.listdir(eval_config.experiment_dir)
            if os.path.isdir(os.path.join(eval_config.experiment_dir, d))
        ]
    )

    logger.info(
        f"Found {len(experiment_names)} experiment directories in {eval_config.experiment_dir}"
    )
    classifier_results_json = asyncio.run(
        evaluate_sweep(
            eval_config=eval_config,
            experiment_names=experiment_names,
        )
    )

    # Save results to output file
    os.makedirs(os.path.dirname(os.path.abspath(eval_config.output_file)), exist_ok=True)
    with open(eval_config.output_file, "w") as f:
        json.dump(classifier_results_json, f, indent=2)

    logger.info(f"Saved classifier results to {eval_config.output_file}")
    print(json.dumps(classifier_results_json, indent=2))


if __name__ == "__main__":
    chz.nested_entrypoint(run_eval_sweep)
