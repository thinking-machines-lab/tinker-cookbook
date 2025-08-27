"""
Supervised fine-tuning (SFT)
"""

import asyncio
import logging
import os
import time

import chz
import tinker
from tinker import types
from tinker_cookbook.display import colorize_example
from tinker_cookbook.evaluators import (
    EvaluatorBuilder,
    SamplingClientEvaluator,
    TrainingClientEvaluator,
)
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed
from tinker_cookbook.utils.training_utils import compute_schedule_lr_multiplier, save_checkpoint

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    """Configuration for supervised fine-tuning."""

    # Required parameters
    log_relpath: str
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

    @property
    def log_base_dir(self) -> str:
        return os.path.expanduser("~/experiments")


async def run_evals(
    evaluators: list[TrainingClientEvaluator | SamplingClientEvaluator],
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


def main(config: Config):
    """Main training function that runs the complete training process."""
    logging.basicConfig(level=logging.INFO)

    # Setup
    ml_logger = ml_log.setup_logging(
        log_dir=os.path.join(config.log_base_dir, config.log_relpath),
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=config.wandb_name,
    )
    service_client = tinker.ServiceClient(base_url=config.base_url)
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )

    if config.load_checkpoint_path:
        training_client.load_state(config.load_checkpoint_path)
        logger.info(f"Loaded weights from {config.load_checkpoint_path}")

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
    for epoch_idx in range(config.num_epochs):
        # Shuffle the dataset
        logger.info(f"Shuffling dataset for epoch {epoch_idx}")
        dataset.shuffle(seed=epoch_idx)

        for batch_idx in range(n_batches):
            step = epoch_idx * n_batches + batch_idx
            metrics: dict[str, int | float | str] = {"epoch": epoch_idx}
            start_time = time.time()
            learning_rate = (
                compute_schedule_lr_multiplier(
                    lr_schedule=config.lr_schedule,
                    step=step,
                    total_steps=total_steps,
                )
                * config.learning_rate
            )
            adam_params = types.AdamParams(
                learning_rate=learning_rate,
                beta1=config.adam_beta1,
                beta2=config.adam_beta2,
                eps=config.adam_eps,
            )

            # Save checkpoint if needed
            if step % config.save_every == 0 and step > 0:
                with timed("save_checkpoint", metrics):
                    state_path = save_checkpoint(
                        training_client=training_client, name=f"{step:06d}"
                    )
                    metrics["state_path"] = state_path
            # Evaluation
            if config.eval_every > 0 and step % config.eval_every == 0:
                with timed("evals", metrics):
                    eval_metrics = asyncio.run(run_evals(evaluators, training_client, step))
                metrics.update(eval_metrics)

            if config.infrequent_eval_every > 0 and step % config.infrequent_eval_every == 0:
                with timed("infrequent_evals", metrics):
                    eval_metrics = asyncio.run(
                        run_evals(infrequent_evaluators, training_client, step)
                    )
                metrics.update(eval_metrics)

            # Prepare batch
            with timed("get_batch", metrics):
                data = dataset.get_batch(batch_idx)
            print(colorize_example(data[0], get_tokenizer(config.model_name)))

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
                learning_rate=learning_rate,
                train_mean_nll=train_nll,
                progress=step / total_steps,
            )

            # Log metrics
            metrics["time/total"] = time.time() - start_time
            ml_logger.log_metrics(metrics=metrics, step=step)

    # Save final checkpoint
    state_future = training_client.save_state("final")
    save_weights_future = training_client.save_weights_for_sampler("final")
    state_path = state_future.result().path
    weights_path = save_weights_future.result().path
    logger.info(f"Saved state to {state_path} and weights to {weights_path}")
    ml_logger.log_metrics(
        metrics={"state_path": state_path, "weights_path": weights_path}, step=total_steps
    )

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    chz.nested_entrypoint(main, allow_hyphens=True)
