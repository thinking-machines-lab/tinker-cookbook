"""
Supervised fine-tuning (SFT)
"""

import asyncio
import logging
import os
import time

import chz
import tinker
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import (
    Evaluator,
    EvaluatorBuilder,
    SamplingClientEvaluator,
    TrainingClientEvaluator,
)
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import timed

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    """Configuration for supervised fine-tuning."""

    # Required parameters
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    model_name: str
    load_checkpoint_path: str | None = None
    dataset_builder: SupervisedDatasetBuilder

    # Training parameters
    learning_rate: float = 1e-4
    lr_schedule: str = "linear"
    num_epochs: int = 1

    # Model parameters
    lora_rank: int = 32

    # Infrastructure parameters
    base_url: str | None = None

    # Checkpointing and evaluation
    evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    infrequent_evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    save_every: int = 20
    eval_every: int = 10
    infrequent_eval_every: int = 100

    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Logging parameters
    wandb_project: str | None = None
    wandb_name: str | None = None


async def run_evals(
    evaluators: list[Evaluator],
    training_client: tinker.TrainingClient,
    step: int,
) -> dict[str, float]:
    """Run all evaluators and return metrics with test/ prefix."""
    metrics = {}
    sampling_client = None

    for evaluator in evaluators:
        if isinstance(evaluator, TrainingClientEvaluator):
            eval_metrics = await evaluator(training_client)
        elif isinstance(evaluator, SamplingClientEvaluator):
            # Create sampling client lazily, only when needed
            if sampling_client is None:
                sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                    f"evals_step_{step}"
                )
            eval_metrics = await evaluator(sampling_client)
        else:
            raise ValueError(f"Unknown evaluator type: {type(evaluator)}")

        # Add test/ prefix to all metrics
        metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

    return metrics


def do_update(
    epoch_idx: int,
    batch_idx: int,
    n_batches: int,
    total_steps: int,
    config: Config,
    training_client: tinker.TrainingClient,
    evaluators: list[Evaluator],
    infrequent_evaluators: list[Evaluator],
    dataset: SupervisedDataset,
    ml_logger: ml_log.Logger,
    log_path: str,
):
    start_time = time.time()
    step = epoch_idx * n_batches + batch_idx
    metrics: dict[str, int | float | str] = {"epoch": epoch_idx}

    # Save checkpoint if needed
    if step % config.save_every == 0 and step > 0:
        with timed("save_checkpoint", metrics):
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=log_path,
                kind="both",
                loop_state={"epoch": epoch_idx, "batch": batch_idx},
            )

    learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
        lr_schedule=config.lr_schedule, step=step, total_steps=total_steps
    )
    adam_params = tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        eps=config.adam_eps,
    )

    # Evaluation
    if config.eval_every > 0 and step % config.eval_every == 0:
        with timed("evals", metrics):
            eval_metrics = asyncio.run(run_evals(evaluators, training_client, step))
        metrics.update(eval_metrics)

    if config.infrequent_eval_every > 0 and step % config.infrequent_eval_every == 0:
        with timed("infrequent_evals", metrics):
            eval_metrics = asyncio.run(run_evals(infrequent_evaluators, training_client, step))
        metrics.update(eval_metrics)

    # Prepare batch
    with timed("get_batch", metrics):
        data = dataset.get_batch(batch_idx)
    logger.info(colorize_example(data[0], get_tokenizer(config.model_name)))

    with timed("step", metrics):
        # Queue up the forward-backward pass and optimizer step before requesting either
        fwd_bwd_future = training_client.forward_backward(data, loss_fn="cross_entropy")
        # Optimizer step
        optim_step_future = training_client.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()
        _optim_step_result = optim_step_future.result()

    # Compute training metrics
    logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
    weights = [datum.loss_fn_inputs["weights"] for datum in data]
    train_nll = compute_mean_nll(logprobs, weights)

    # Prepare metrics
    metrics.update(
        num_sequences=len(data),
        num_tokens=sum(datum.model_input.length for datum in data),
        num_loss_tokens=sum(sum(datum.loss_fn_inputs["weights"].data) for datum in data),
        learning_rate=learning_rate,
        train_mean_nll=train_nll,
        progress=step / total_steps,
    )

    # Log metrics
    metrics["time/total"] = time.time() - start_time
    ml_logger.log_metrics(metrics=metrics, step=step)


def main(config: Config):
    """Main training function that runs the complete training process."""
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_epoch = resume_info["epoch"]
        start_batch = resume_info["batch"]
    else:
        start_epoch = 0
        start_batch = 0

    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )
    service_client = tinker.ServiceClient(base_url=config.base_url)
    load_state_path: str | None = (
        resume_info["state_path"] if resume_info else config.load_checkpoint_path
    )

    user_metadata = {}
    if wandbd_link := ml_logger.get_logger_url():
        user_metadata["wandbd_link"] = wandbd_link

    if load_state_path:
        training_client = service_client.create_training_client_from_state(
            load_state_path, user_metadata
        )
        logger.info(f"Loaded weights from {load_state_path}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank, user_metadata=user_metadata
        )

    # Training setup
    dataset, maybe_test_dataset = config.dataset_builder()
    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs

    evaluators = [evaluator() for evaluator in config.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(NLLEvaluator.from_dataset(maybe_test_dataset))

    infrequent_evaluators = [evaluator() for evaluator in config.infrequent_evaluator_builders]
    logger.info(
        f"Training for {n_batches} batches x {config.num_epochs} epochs = {n_batches * config.num_epochs} steps"
    )

    # Training loop
    for epoch_idx in range(start_epoch, config.num_epochs):
        # Shuffle the dataset
        logger.info(msg=f"Starting epoch {epoch_idx}")
        dataset.set_epoch(seed=epoch_idx)

        for batch_idx in range(start_batch if epoch_idx == start_epoch else 0, n_batches):
            do_update(
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                n_batches=n_batches,
                total_steps=total_steps,
                config=config,
                training_client=training_client,
                evaluators=evaluators,
                infrequent_evaluators=infrequent_evaluators,
                dataset=dataset,
                ml_logger=ml_logger,
                log_path=config.log_path,
            )

    # Save final checkpoint if training actually happened
    if start_epoch < config.num_epochs:
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"epoch": config.num_epochs, "batch": n_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
