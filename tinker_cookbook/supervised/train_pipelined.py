"""
A faster version of supervised learning that submits the next batch while the current batch is being processed.
This avoids missing a clock cycle of the training process while you load the next batch.
"""

import logging
import time
from dataclasses import dataclass

import chz
import tinker
from tinker.lib.public_interfaces import APIFuture
from tinker_cookbook.display import colorize_example
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator
from tinker_cookbook.supervised.train import Config, run_evals
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

logger = logging.getLogger(__name__)


@dataclass
class SubmitBatchResult:
    fwd_bwd_future: APIFuture[tinker.ForwardBackwardOutput]
    optim_step_future: APIFuture[tinker.OptimStepResponse]
    metrics: dict[str, float | str]
    data: list
    batch_idx: int


async def save_checkpoint_async(
    training_client: tinker.TrainingClient, name: str
) -> dict[str, str]:
    """Save model checkpoint asynchronously.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
    Returns:
        Dictionary with 'weights_path' and 'state_path' keys
    """
    # XXX currently saving both sampler and state
    save_weights_future = await training_client.save_weights_for_sampler_async(name)
    save_state_future = await training_client.save_state_async(name)
    save_weights_result = await save_weights_future.result_async()
    save_state_result = await save_state_future.result_async()
    logger.info(f"Saved weights for sampler to: {save_weights_result.path}")
    logger.info(f"Saved state to: {save_state_result.path}")
    return {"weights_path": save_weights_result.path, "state_path": save_state_result.path}


async def main(config: Config):
    """Main training function that runs the complete training process."""
    # Setup
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=config.wandb_name,
        do_configure_logging_module=True,
    )

    service_client = tinker.ServiceClient(base_url=config.base_url)
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model_name, rank=config.lora_rank
    )

    # Training setup
    dataset, maybe_test_dataset = config.dataset_builder()
    n_batches = len(dataset)

    evaluators = [evaluator() for evaluator in config.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(NLLEvaluator.from_dataset(maybe_test_dataset))
    logger.info(f"Training for {n_batches} batches")

    async def submit_batch(batch_idx: int) -> SubmitBatchResult:
        learning_rate = (
            compute_schedule_lr_multiplier(
                lr_schedule=config.lr_schedule, step=batch_idx, total_steps=n_batches
            )
            * config.learning_rate
        )
        adam_params = tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            eps=config.adam_eps,
        )
        data = dataset.get_batch(batch_idx)
        logger.info(colorize_example(data[0], get_tokenizer(config.model_name)))

        fwd_bwd_future = await training_client.forward_backward_async(data, loss_fn="cross_entropy")
        optim_step_future = await training_client.optim_step_async(adam_params)
        metrics = {
            "learning_rate": learning_rate,
            "num_sequences": len(data),
            "num_tokens": sum(datum.model_input.length for datum in data),
            "batch_start_time": time.time(),
            "progress": (batch_idx + 1) / n_batches,
        }
        return SubmitBatchResult(
            fwd_bwd_future=fwd_bwd_future,
            optim_step_future=optim_step_future,
            metrics=metrics,
            data=data,
            batch_idx=batch_idx,
        )

    async def finish_batch(submit_result: SubmitBatchResult):
        metrics = submit_result.metrics
        batch_idx = submit_result.batch_idx

        # Save checkpoint if needed (before waiting for results)
        if batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_paths = await save_checkpoint_async(
                training_client=training_client, name=f"{batch_idx:06d}"
            )
            metrics.update(checkpoint_paths)

        # Wait for results
        fwd_bwd_result = await submit_result.fwd_bwd_future.result_async()
        _optim_step_result = await submit_result.optim_step_future.result_async()

        # Compute training metrics
        logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        weights = [datum.loss_fn_inputs["weights"] for datum in submit_result.data]
        train_nll = compute_mean_nll(logprobs, weights)

        batch_start_time = submit_result.metrics.pop("batch_start_time")
        assert isinstance(batch_start_time, float)
        metrics = {
            "train_mean_nll": train_nll,
            "batch_time": time.time() - batch_start_time,
            **submit_result.metrics,
        }
        # Evaluation
        if config.eval_every > 0 and batch_idx % config.eval_every == 0:
            eval_metrics = await run_evals(evaluators, training_client, batch_idx)
            metrics.update(eval_metrics)

        # Log metrics
        ml_logger.log_metrics(metrics=metrics, step=batch_idx)

    # Pipelined training loop
    pending_batch = None

    for batch_idx in range(n_batches):
        # Submit the current batch
        current_batch = await submit_batch(batch_idx)

        # If there's a pending batch from the previous iteration, finish it
        if pending_batch is not None:
            await finish_batch(pending_batch)

        # The current batch becomes the pending batch for the next iteration
        pending_batch = current_batch

    # Finish the last batch
    if pending_batch is not None:
        await finish_batch(pending_batch)

    # Save final checkpoint
    await save_checkpoint_async(training_client=training_client, name="final")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    import asyncio

    chz.nested_entrypoint(lambda config: asyncio.run(main(config)), allow_hyphens=True)
