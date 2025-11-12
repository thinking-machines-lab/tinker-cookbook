"""
Supervised fine-tuning (SFT)

This module implements a pipelined supervised learning training loop. For background on
why we pipeline requests, see https://tinker-docs.thinkingmachines.ai/under-the-hood.
For a minimal, pedagogical example of SL training without these optimizations,
refer to `tinker_cookbook/recipes/sl_loop.py`.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import chz
import tinker
from tinker.lib.public_interfaces import APIFuture
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
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder
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


@dataclass
class SubmittedBatch:
    fwd_bwd_future: APIFuture[tinker.ForwardBackwardOutput]
    optim_step_future: APIFuture[tinker.OptimStepResponse]
    metrics: dict[str, int | float | str]
    data: list
    step: int
    epoch_idx: int
    batch_idx: int
    batch_start_time: float


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


async def main(config: Config):
    """Main training function that runs the complete training process."""
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_epoch = resume_info["epoch"]
        start_batch = resume_info["batch"]
    else:
        start_epoch = 0
        start_batch = 0

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

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link

    if load_state_path:
        training_client = await service_client.create_training_client_from_state_async(
            load_state_path, user_metadata
        )
        logger.info(f"Loaded weights from {load_state_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
            user_metadata=user_metadata,
        )

    dataset, maybe_test_dataset = config.dataset_builder()
    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs
    progress_denominator = total_steps if total_steps > 0 else 1
    tokenizer = get_tokenizer(config.model_name)

    evaluators = [evaluator() for evaluator in config.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(NLLEvaluator.from_dataset(maybe_test_dataset))

    infrequent_evaluators = [evaluator() for evaluator in config.infrequent_evaluator_builders]
    logger.info(
        f"Training for {n_batches} batches x {config.num_epochs} epochs = {n_batches * config.num_epochs} steps"
    )

    async def submit_batch(epoch_idx: int, batch_idx: int) -> SubmittedBatch:
        step = epoch_idx * n_batches + batch_idx
        batch_start_time = time.time()
        metrics: dict[str, int | float | str] = {"epoch": epoch_idx}
        metrics["progress"] = step / progress_denominator

        learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
            lr_schedule=config.lr_schedule,
            step=step,
            total_steps=total_steps,
        )
        metrics["learning_rate"] = learning_rate

        adam_params = tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            eps=config.adam_eps,
        )

        with timed("get_batch", metrics):
            data = dataset.get_batch(batch_idx)
        if data:
            logger.info(colorize_example(data[0], tokenizer))

        fwd_bwd_future = await training_client.forward_backward_async(data, loss_fn="cross_entropy")
        optim_step_future = await training_client.optim_step_async(adam_params)

        return SubmittedBatch(
            fwd_bwd_future=fwd_bwd_future,
            optim_step_future=optim_step_future,
            metrics=metrics,
            data=data,
            step=step,
            epoch_idx=epoch_idx,
            batch_idx=batch_idx,
            batch_start_time=batch_start_time,
        )

    async def finish_batch(submitted: SubmittedBatch):
        metrics = submitted.metrics
        metrics["progress"] = min((submitted.step + 1) / progress_denominator, 1.0)

        if submitted.step % config.save_every == 0 and submitted.step > 0:
            with timed("save_checkpoint", metrics):
                await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=f"{submitted.step:06d}",
                    log_path=config.log_path,
                    loop_state={"epoch": submitted.epoch_idx, "batch": submitted.batch_idx},
                    kind="both",
                )

        with timed("step", metrics):
            fwd_bwd_result = await submitted.fwd_bwd_future.result_async()
            await submitted.optim_step_future.result_async()

        logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in submitted.data]
        train_nll = compute_mean_nll(logprobs, weights)

        metrics.update(
            num_sequences=len(submitted.data),
            num_tokens=sum(datum.model_input.length for datum in submitted.data),
            num_loss_tokens=sum(
                sum(datum.loss_fn_inputs["weights"].data) for datum in submitted.data
            ),
            train_mean_nll=train_nll,
        )
        metrics["time/total"] = time.time() - submitted.batch_start_time

        if evaluators and config.eval_every > 0 and submitted.step % config.eval_every == 0:
            with timed("evals", metrics):
                eval_metrics = await run_evals(evaluators, training_client, submitted.step)
            metrics.update(eval_metrics)

        if (
            infrequent_evaluators
            and config.infrequent_eval_every > 0
            and submitted.step % config.infrequent_eval_every == 0
        ):
            with timed("infrequent_evals", metrics):
                eval_metrics = await run_evals(
                    infrequent_evaluators, training_client, submitted.step
                )
            metrics.update(eval_metrics)

        ml_logger.log_metrics(metrics=metrics, step=submitted.step)

    pending_batch: SubmittedBatch | None = None

    for epoch_idx in range(start_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch_idx}")
        dataset.set_epoch(seed=epoch_idx)

        start_batch_idx = start_batch if epoch_idx == start_epoch else 0
        for batch_idx in range(start_batch_idx, n_batches):
            submitted_batch = await submit_batch(epoch_idx, batch_idx)
            if pending_batch is not None:
                await finish_batch(pending_batch)
            pending_batch = submitted_batch

    if pending_batch is not None:
        await finish_batch(pending_batch)

    if start_epoch < config.num_epochs:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"epoch": config.num_epochs, "batch": n_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    chz.nested_entrypoint(lambda config: asyncio.run(main(config)), allow_hyphens=True)
