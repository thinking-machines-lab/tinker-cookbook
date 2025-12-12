"""

## VLM Image Classifier

Launcher for training image classifiers based on VLMs.

```bash
python -m tinker_cookbook.recipes.vlm_classifier.train experiment_dir=./vlm_classifier
```

"""

import os

import asyncio
from datetime import datetime
from typing import Literal

import chz
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.utils.lr_scheduling import LRSchedule
from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.vlm_classifier.eval import get_evaluator_builder
from tinker_cookbook.recipes.vlm_classifier.data import get_dataset_builder
from tinker_cookbook.supervised import train


@chz.chz
class ExperimentConfig:
    """
    Experiments for few-shot image classification with VLMs.
    """

    experiment_dir: str
    load_checkpoint_path: str | None = None

    dataset: str = "caltech101"

    renderer_name: str = "qwen3_vl"
    model_name: str = "Qwen/Qwen3-VL-235B-A22B-Instruct"

    # Infrastructure parameters
    base_url: str | None = None
    behavior_if_log_dir_exists: Literal["delete", "resume", "ask", "raise"] = "ask"

    # Training parameters
    learning_rate: float = 5e-4
    num_epochs: int = 3
    lr_schedule: LRSchedule = "cosine"

    # Model parameters
    lora_rank: int = 32

    # Checkpointing and evaluation
    save_every: int = 20
    eval_every: int = 20
    infrequent_eval_every: int = 100

    # Logging parameters
    wandb_project: str | None = None
    wandb_name: str | None = None

    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE

    num_repeats: float = 1
    batch_size: int = 32
    max_length: int = 8192

    examples_per_class: int | None = None
    subset_seed: int = 0

    run_nll_evaluator: bool = True
    run_sampling_evaluator: bool = True

    temperature: float = 0.0
    max_tokens: int = 128
    top_p: float = 1.0
    top_k: int = -1

    n_eval: int = 128


def run_experiment(experiment_config: ExperimentConfig):
    """
    Launcher for training an image classifier based on a VLM on a custom vision dataset.
    """

    # build full config
    model_name = experiment_config.model_name.replace("/", "-")
    date_and_time = datetime.now().strftime("%Y-%m-%d")

    # Include examples_per_class and subset_seed in run name if doing few-shot
    shot_suffix = (
        f"-{experiment_config.examples_per_class}shot-seed{experiment_config.subset_seed}"
        if experiment_config.examples_per_class
        else ""
    )
    experiment_name = f"{experiment_config.dataset}-{model_name}-{experiment_config.lora_rank}rank-{experiment_config.learning_rate}lr-{experiment_config.batch_size}batch{shot_suffix}-{date_and_time}"

    experiment_path = os.path.join(experiment_config.experiment_dir, experiment_name)
    cli_utils.check_log_dir(
        experiment_path, behavior_if_exists=experiment_config.behavior_if_log_dir_exists
    )

    dataset_builder = get_dataset_builder(
        dataset=experiment_config.dataset,
        model_name_for_tokenizer=experiment_config.model_name,
        renderer_name=experiment_config.renderer_name,
        num_repeats=experiment_config.num_repeats,
        batch_size=experiment_config.batch_size,
        max_length=experiment_config.max_length,
        train_on_what=experiment_config.train_on_what,
        examples_per_class=experiment_config.examples_per_class,
        subset_seed=experiment_config.subset_seed,
        run_nll_evaluator=experiment_config.run_nll_evaluator,
    )

    evaluator_builders = []
    if experiment_config.run_sampling_evaluator:
        evaluator_builders = [
            get_evaluator_builder(
                dataset=experiment_config.dataset,
                model_name_for_tokenizer=experiment_config.model_name,
                renderer_name=experiment_config.renderer_name,
                temperature=experiment_config.temperature,
                max_tokens=experiment_config.max_tokens,
                top_p=experiment_config.top_p,
                top_k=experiment_config.top_k,
                n_eval=experiment_config.n_eval,
            )
        ]

    config = train.Config(
        log_path=experiment_path,
        model_name=experiment_config.model_name,
        load_checkpoint_path=experiment_config.load_checkpoint_path,
        dataset_builder=dataset_builder,
        evaluator_builders=evaluator_builders,
        infrequent_evaluator_builders=[],
        learning_rate=experiment_config.learning_rate,
        lr_schedule=experiment_config.lr_schedule,
        num_epochs=experiment_config.num_epochs,
        base_url=experiment_config.base_url,
        wandb_project=experiment_config.wandb_project,
        wandb_name=experiment_config.wandb_name or experiment_name,
        lora_rank=experiment_config.lora_rank,
        save_every=experiment_config.save_every,
        eval_every=experiment_config.eval_every,
        infrequent_eval_every=experiment_config.infrequent_eval_every,
    )

    asyncio.run(train.main(config))


if __name__ == "__main__":
    chz.nested_entrypoint(run_experiment)
