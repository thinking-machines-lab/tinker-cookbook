"""
Supervised fine-tuning (SFT)
"""

import logging
import os
import time

import chz
import tinker_public
from tinker_cookbook.display import colorize_example
from tinker_cookbook.evaluators import EvaluatorBuilder
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.ml_log import setup_logging
from tinker_cookbook.utils.training_utils import compute_schedule_lr_multiplier, save_checkpoint
from tinker_public import types

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    """Configuration for supervised fine-tuning."""

    # Required parameters
    log_relpath: str
    model_name: str
    dataset_builder: SupervisedDatasetBuilder
    # dataset_builder optionally returns an evaluator (test set)

    # Training parameters
    learning_rate: float = 1e-5
    lr_schedule: str = "linear"

    # Model parameters
    lora_rank: int = 32

    # Infrastructure parameters
    num_replicas: int = 8
    base_url: str | None = None

    # Checkpointing and evaluation
    evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    save_every: int = 20
    test_interval: int = 10

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


def main(config: Config):
    """Main training function that runs the complete training process."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting supervised fine-tuning")

    # Validate config
    # Warn if we got an unexpected renderer

    # Setup
    ml_logger = setup_logging(
        log_dir=os.path.join(config.log_base_dir, config.log_relpath),
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=config.wandb_name,
    )
    service_client = tinker_public.ServiceClient(base_url=config.base_url)
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )

    # Training setup
    dataset, maybe_evaluator = config.dataset_builder()
    n_batches = len(dataset)

    evaluators = [evaluator() for evaluator in config.evaluator_builders]
    if maybe_evaluator is not None:
        evaluators.append(maybe_evaluator)
    logger.info(f"Training for {n_batches} batches")

    # Training loop
    for batch_idx in range(n_batches):
        metrics = {}
        batch_start_time = time.time()
        learning_rate = (
            compute_schedule_lr_multiplier(
                lr_schedule=config.lr_schedule, step=batch_idx, total_steps=n_batches
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
        if batch_idx % config.save_every == 0 and batch_idx > 0:
            metrics["save_path"] = save_checkpoint(
                training_client=training_client, name=f"{batch_idx:06d}"
            )

        # Prepare batch
        data = dataset.get_batch(batch_idx)
        print(colorize_example(data[0], get_tokenizer(config.model_name)))
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
            progress=(batch_idx + 1) / n_batches,
            batch_time=time.time() - batch_start_time,
        )

        # Evaluation
        if config.test_interval > 0 and batch_idx % config.test_interval == 0:
            for evaluator in evaluators:
                eval_metrics = evaluator(training_client)
                metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})
                # TODO make sure evaluators have different names
        # Log metrics
        ml_logger.log_metrics(metrics=metrics, step=batch_idx)
    # Save final checkpoint
    save_checkpoint(training_client=training_client, name="final")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    chz.nested_entrypoint(main, allow_hyphens=True)
