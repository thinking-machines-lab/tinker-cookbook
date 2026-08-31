"""Train Qwen3.8-27B with Tinker on binary Prophet Arena forecasts.

The default run uses the Brier reward for 100 steps.

    python -m tinker_cookbook.recipes.forecasting.train
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import chz
import tinker
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.rl import train
from tinker_cookbook.stores.storage import storage_from_uri
from tinker_cookbook.stores.training_store import TrainingRunStore

from .data import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET_REVISION,
    DEFAULT_MAX_TRAIN_QUESTIONS,
    DEFAULT_MAX_VALIDATION_QUESTIONS,
    DEFAULT_SPLIT_DATE,
)
from .env import ProphetArenaRLDatasetBuilder


@chz.chz
class Config:
    # Model
    model_name: str = "Qwen/Qwen3.8-27B"
    renderer_name: str | None = "qwen3_8_low_reasoning"
    load_checkpoint_path: str | None = None
    lora_rank: int = 32

    # Data
    data_path: str | None = None
    data_cache_dir: str = DEFAULT_CACHE_DIR
    dataset_revision: str = DEFAULT_DATASET_REVISION
    split_date: str = DEFAULT_SPLIT_DATE
    max_train_questions: int | None = DEFAULT_MAX_TRAIN_QUESTIONS
    max_validation_questions: int | None = DEFAULT_MAX_VALIDATION_QUESTIONS
    train_epochs: int = 3
    seed: int = 0

    # RL
    group_size: int = 32
    groups_per_batch: int = 16
    validation_group_size: int = 8
    learning_rate: float = 8e-5
    max_tokens: int = 24_576
    temperature: float = 1.0
    loss_fn: LossFnType = "importance_sampling"
    max_steps: int | None = 100

    # Evaluation and logging
    eval_every: int = 20
    save_every: int = 20
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    base_url: str | None = None


async def _evaluate_final_checkpoint(
    train_config: train.Config,
    dataset_builder: ProphetArenaRLDatasetBuilder,
) -> None:
    """Evaluate the final weights when periodic evaluation stops before them."""
    if train_config.eval_every <= 0:
        return

    checkpoint = checkpoint_utils.get_last_checkpoint(
        train_config.log_path, required_key="sampler_path"
    )
    if checkpoint is None or checkpoint.batch is None or checkpoint.sampler_path is None:
        raise RuntimeError("final sampler checkpoint was not saved")

    store = TrainingRunStore(storage_from_uri(train_config.log_path))
    if any(
        metrics.get("step") == checkpoint.batch and "test/env/all/brier_reward" in metrics
        for metrics in store.read_metrics()
    ):
        return

    _, validation_dataset = await dataset_builder()
    evaluator = train.RLTestSetEvaluator(
        validation_dataset,
        max_tokens=train_config.max_tokens,
        strategy=train_config.effective_rollout_strategy(),
    )
    sampling_client = tinker.ServiceClient(base_url=train_config.base_url).create_sampling_client(
        base_model=train_config.model_name,
        model_path=checkpoint.sampler_path,
    )
    metrics = await train.run_evaluations_parallel(
        [evaluator],
        sampling_client,
        train_config,
        checkpoint.batch,
        store=store,
    )
    store.write_metrics(metrics, step=checkpoint.batch)


async def cli_main(cfg: Config) -> None:
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cfg.model_name,
        explicit_renderer_name=cfg.renderer_name,
        load_checkpoint_path=cfg.load_checkpoint_path,
        base_url=cfg.base_url,
    )
    model_name = cfg.model_name.lower().replace("/", "-")
    run_name = (
        f"prophet-arena-{model_name}-bs{cfg.groups_per_batch}-"
        f"gs{cfg.group_size}-lr{cfg.learning_rate}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cfg.log_path or f"/tmp/tinker-examples/prophet_arena_qwen_rl/{run_name}"

    dataset_builder = ProphetArenaRLDatasetBuilder(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        groups_per_batch=cfg.groups_per_batch,
        group_size=cfg.group_size,
        validation_group_size=cfg.validation_group_size,
        train_epochs=cfg.train_epochs,
        data_path=cfg.data_path,
        data_cache_dir=cfg.data_cache_dir,
        dataset_revision=cfg.dataset_revision,
        split_date=cfg.split_date,
        max_train_questions=cfg.max_train_questions,
        max_validation_questions=cfg.max_validation_questions,
        seed=cfg.seed,
    )
    train_config = train.Config(
        model_name=cfg.model_name,
        recipe_name="recipe_prophet_arena_qwen_rl",
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        log_path=log_path,
        load_checkpoint_path=cfg.load_checkpoint_path,
        lora_rank=cfg.lora_rank,
        learning_rate=cfg.learning_rate,
        loss_fn=cfg.loss_fn,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        max_steps=cfg.max_steps,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name or run_name,
        base_url=cfg.base_url,
        kl_penalty_coef=0.0,
        compute_post_kl=False,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    await train.main(train_config)
    await _evaluate_final_checkpoint(train_config, dataset_builder)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(Config)))
