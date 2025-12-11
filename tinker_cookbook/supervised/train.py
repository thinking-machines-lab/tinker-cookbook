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
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier, LRSchedule
from tinker_cookbook.utils.misc_utils import timed
from tinker_cookbook.utils.trace import scope, update_scope_context, trace_init

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
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1

    # Model parameters
    lora_rank: int = 32

    # Infrastructure parameters
    base_url: str | None = None

    # Checkpointing and evaluation (0 = disabled for *_every fields)
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

    enable_trace: bool = False


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
    eval_metrics: dict[str, float] | None = None
    infrequent_eval_metrics: dict[str, float] | None = None


@scope
async def run_evals(
    evaluators: list[Evaluator],
    training_client: tinker.TrainingClient,
    step: int,
) -> dict[str, float]:
    """Evaluate the current model weights and prefix results with ``test/``.

    The helper is called immediately before optimizer step `step` is submitted, so it
    measures the weights produced after step `step-1` (or the initial weights for step 0).
    Training-client evaluators run against the mutable training client, while sampling
    evaluators request a fresh `SamplingClient` snapshot via
    `save_weights_and_get_sampling_client_async` to ensure their work uses a fixed
    checkpoint. Returned metrics are prefixed with ``test/`` so they can be logged next
    to the same-step training metrics.
    """
    update_scope_context({"step": step})

    metrics = {}
    sampling_client = None

    @scope
    async def run_evaluator(evaluator: Evaluator) -> dict[str, float]:
        update_scope_context(
            {
                "step": step,
                "evaluator_name": type(evaluator).__name__,
            }
        )
        if isinstance(evaluator, TrainingClientEvaluator):
            update_scope_context({"evaluator_type": "TrainingClientEvaluator"})
            return await evaluator(training_client)
        elif isinstance(evaluator, SamplingClientEvaluator):
            update_scope_context({"evaluator_type": "SamplingClientEvaluator"})
            # Create sampling client lazily, only when needed
            nonlocal sampling_client
            if sampling_client is None:
                # Snapshot the current pre-step weights and create a new sampling client.
                sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                    f"evals_step_{step}"
                )
            return await evaluator(sampling_client)
        else:
            raise ValueError(f"Unknown evaluator type: {type(evaluator)}")

    for evaluator in evaluators:
        eval_metrics = await run_evaluator(evaluator)
        # Add test/ prefix to all metrics
        metrics.update(eval_metrics)

    return metrics


@scope
async def main(config: Config):
    """Run the standard supervised learning loop used by the supervised recipes.

    Responsibilities:
    1. Initialize logging, build the dataset/evaluator objects, construct (or resume) the
       training client, and determine the ``epoch``/``batch`` indices to start from.
    2. Iterate over batches: fetch data, optionally run evaluations before submitting the
       optimizer step (so they observe pre-step weights), issue `forward_backward` and
       `optim_step` requests, and log metrics once the futures resolve.
    3. Save checkpoints at the configured cadence so runs can resume or export weights,
       then emit a final checkpoint when training completes.

    Training and evaluation metrics share the same ``step`` index to keep dashboards easy
    to read.
    """
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_epoch = resume_info["epoch"]
        start_batch = resume_info["batch"]
    else:
        start_epoch = 0
        start_batch = 0
    # (start_epoch, start_batch) now represent the next batch to execute if resuming.

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )
    if config.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(config.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=os.path.join(config.log_path, "trace_events.jsonl"))

    service_client = tinker.ServiceClient(base_url=config.base_url)

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link

    if resume_info:
        # Resuming interrupted training - load optimizer state for proper continuation
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"], user_metadata
            )
        )
        logger.info(f"Resumed training from {resume_info['state_path']}")
    elif config.load_checkpoint_path:
        # Starting fresh from a checkpoint - load weights only (fresh optimizer)
        training_client = await service_client.create_training_client_from_state_async(
            config.load_checkpoint_path, user_metadata
        )
        logger.info(f"Loaded weights from {config.load_checkpoint_path}")
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

    @scope
    async def submit_batch(epoch_idx: int, batch_idx: int) -> SubmittedBatch:
        step = epoch_idx * n_batches + batch_idx
        update_scope_context({"step": step})

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

        # Trigger evaluations BEFORE submitting training operations so they snapshot pre-step weights
        eval_metrics = None
        if evaluators and config.eval_every > 0 and step % config.eval_every == 0:
            with timed("evals", metrics):
                eval_metrics = await run_evals(evaluators, training_client, step)

        infrequent_eval_metrics = None
        if (
            infrequent_evaluators
            and config.infrequent_eval_every > 0
            and step % config.infrequent_eval_every == 0
        ):
            with timed("infrequent_evals", metrics):
                infrequent_eval_metrics = await run_evals(
                    infrequent_evaluators, training_client, step
                )

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
            eval_metrics=eval_metrics,
            infrequent_eval_metrics=infrequent_eval_metrics,
        )

    @scope
    async def finish_batch(submitted: SubmittedBatch):
        update_scope_context({"step": submitted.step})

        metrics = submitted.metrics
        metrics["progress"] = min((submitted.step + 1) / progress_denominator, 1.0)

        if config.save_every > 0 and submitted.step % config.save_every == 0 and submitted.step > 0:
            with timed("save_checkpoint", metrics):
                # Enqueue a checkpoint save after the forward/backward and optimizer
                # requests for this step; the snapshot will reflect post-step weights.
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

        # Merge evaluation metrics gathered before the training step was submitted
        if submitted.eval_metrics is not None:
            metrics.update(submitted.eval_metrics)

        if submitted.infrequent_eval_metrics is not None:
            metrics.update(submitted.infrequent_eval_metrics)

        # Emit all metrics for this step (train and eval) on the `submitted.step` row.
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
