"""
Direct Preference Optimization (DPO) training
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Tuple, cast

import chz
import tinker
import torch
from tinker import types
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.evaluators import EvaluatorBuilder
from tinker_cookbook.supervised.nll_evaluator import NLLEvaluator
from tinker_cookbook.supervised.train import run_evals
from tinker_cookbook.supervised.types import ChatDatasetBuilder
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    """Configuration for Direct Preference Optimization (DPO) training."""

    # Required parameters
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    model_name: str
    dataset_builder: ChatDatasetBuilder
    load_checkpoint_path: str | None = None
    # dataset_builder optionally returns an evaluator (test set)

    # Training parameters
    learning_rate: float = 1e-5
    lr_schedule: str = "linear"
    dpo_beta: float = 0.1

    # Model parameters
    lora_rank: int = 32

    # Infrastructure parameters
    num_replicas: int = 8
    base_url: str | None = None

    # Checkpointing and evaluation
    evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    save_every: int = 20
    eval_every: int = 10

    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Logging parameters
    wandb_project: str | None = None
    wandb_name: str | None = None

    # DPO-specific parameters
    reference_model_name: str | None = None


def create_dpo_clients(
    config: Config,
    resume_info: dict[str, Any] | None = None,
) -> Tuple[tinker.TrainingClient, tinker.TrainingClient]:
    """Create and configure the training clients for DPO.

    Creates both the main training client and the reference client.
    The reference client is used to compute the reference model's log probabilities
    for the DPO loss computation.

    Args:
        config: DPO configuration object

    Returns:
        Tuple of (main training client, reference client)
    """
    # Create shared service client for both training and reference clients
    service_client = tinker.ServiceClient(base_url=config.base_url)
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )
    reference_client = service_client.create_lora_training_client(
        base_model=config.reference_model_name or config.model_name, rank=config.lora_rank
    )

    load_state_path = (resume_info and resume_info["state_path"]) or config.load_checkpoint_path
    if load_state_path:
        assert isinstance(load_state_path, str)
        rc_future = reference_client.load_state(load_state_path)
        tc_future = training_client.load_state(load_state_path)
        _ = rc_future.result()
        _ = tc_future.result()
        logger.info(f"Loaded weights from {load_state_path}")
    return training_client, reference_client


def compute_dpo_loss(
    chosen_logprobs: List[torch.Tensor],
    rejected_logprobs: List[torch.Tensor],
    chosen_ref_logprobs: List[torch.Tensor],
    rejected_ref_logprobs: List[torch.Tensor],
    dpo_beta: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute DPO loss and metrics.

    Args:
        chosen_logprobs: Log probabilities for chosen responses
        rejected_logprobs: Log probabilities for rejected responses
        chosen_ref_logprobs: Reference log probabilities for chosen responses
        rejected_ref_logprobs: Reference log probabilities for rejected responses
        dpo_beta: DPO beta parameter

    Returns:
        Tuple of (loss tensor, metrics dictionary)
    """
    # Compute log ratios
    chosen_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
    )
    rejected_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
    )

    # Compute DPO loss
    losses = -torch.log(torch.sigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio)))
    loss = losses.mean()

    # Compute metrics
    accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
    chosen_rewards = dpo_beta * chosen_log_ratio
    rejected_rewards = dpo_beta * rejected_log_ratio
    margin = dpo_beta * (chosen_rewards - rejected_rewards).mean().item()

    metrics = {
        "dpo_loss": loss.item(),
        "accuracy": accuracy,
        "margin": margin,
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }

    return loss, metrics


def main(config: Config):
    """Main training function that runs the complete DPO training process."""
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=config.wandb_name,
        do_configure_logging_module=True,
    )

    # Check for existing checkpoints
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_batch = resume_info["batch"] + 1  # Start from next batch after checkpoint
        logger.info(f"Resuming from batch {start_batch}")
    else:
        start_batch = 0
        logger.info("Starting training from scratch")

    # Setup
    training_client, reference_client = create_dpo_clients(config, resume_info)
    tokenizer = get_tokenizer(config.model_name)

    # Training setup
    dataset, maybe_test_dataset = config.dataset_builder()
    n_batches = len(dataset)

    evaluators = [evaluator() for evaluator in config.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(NLLEvaluator.from_dataset(maybe_test_dataset))
        # XXX I don't think we want this NLLEvaluator
    logger.info(f"Training for {n_batches} batches")

    # Training loop
    for batch_idx in range(start_batch, n_batches):
        metrics = {}
        batch_start_time = time.time()
        learning_rate = (
            compute_schedule_lr_multiplier(
                lr_schedule=config.lr_schedule,
                step=batch_idx,
                total_steps=n_batches,
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
            save_result = checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=config.log_path,
                kind="both",
                loop_state={"batch": batch_idx},
            )
            if "state_path" in save_result:
                metrics["state_path"] = save_result["state_path"]

        # Prepare batch
        data = dataset.get_batch(batch_idx)

        # Split data into chosen and rejected pairs
        chosen_data = [datum for i, datum in enumerate(data) if i % 2 == 0]
        rejected_data = [datum for i, datum in enumerate(data) if i % 2 == 1]

        # Print example
        if batch_idx == 0:
            for i in range(10):
                print_example(chosen_data[i], tokenizer, "Chosen")
                print_example(rejected_data[i], tokenizer, "Rejected")

        # Get reference log probabilities using single forward pass on original data
        all_ref_result = reference_client.forward(data, "cross_entropy").result()
        all_ref_logprob_seqs = [
            torch.tensor(out["logprobs"].data) for out in all_ref_result.loss_fn_outputs
        ]

        # Split reference results into chosen and rejected
        chosen_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(0, len(data), 2)]
        rejected_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(1, len(data), 2)]

        # Create DPO loss function
        def dpo_loss_fn(
            data: List[types.Datum], logprobs_list: List[torch.Tensor]
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            # Split logprobs into chosen and rejected
            chosen_logprob_seqs = [logprobs_list[i] for i in range(0, len(data), 2)]
            rejected_logprob_seqs = [logprobs_list[i] for i in range(1, len(data), 2)]

            # Extract log probabilities
            chosen_logprobs = []
            chosen_ref_logprobs = []
            rejected_logprobs = []
            rejected_ref_logprobs = []

            for i in range(len(chosen_data)):
                # Compute weighted logprobs for chosen responses
                chosen_logprob_seq = chosen_logprob_seqs[i]
                chosen_ref_logprob_seq = chosen_ref_logprob_seqs[i]
                chosen_weights = torch.tensor(chosen_data[i].loss_fn_inputs["weights"].data)
                chosen_logprob = torch.dot(chosen_logprob_seq.float(), chosen_weights.float())
                chosen_ref_logprob = torch.dot(
                    chosen_ref_logprob_seq.float(), chosen_weights.float()
                )
                chosen_logprobs.append(chosen_logprob)
                chosen_ref_logprobs.append(chosen_ref_logprob)

                # Compute weighted logprobs for rejected responses
                rejected_logprob_seq = rejected_logprob_seqs[i]
                rejected_ref_logprob_seq = rejected_ref_logprob_seqs[i]
                rejected_weights = torch.tensor(rejected_data[i].loss_fn_inputs["weights"].data)
                rejected_logprob = torch.dot(rejected_logprob_seq.float(), rejected_weights.float())
                rejected_ref_logprob = torch.dot(
                    rejected_ref_logprob_seq.float(), rejected_weights.float()
                )
                rejected_logprobs.append(rejected_logprob)
                rejected_ref_logprobs.append(rejected_ref_logprob)

            # Compute DPO loss
            return compute_dpo_loss(
                chosen_logprobs=chosen_logprobs,
                rejected_logprobs=rejected_logprobs,
                chosen_ref_logprobs=chosen_ref_logprobs,
                rejected_ref_logprobs=rejected_ref_logprobs,
                dpo_beta=config.dpo_beta,
            )

        # Do forward-backward with custom DPO loss
        backward_result = training_client.forward_backward_custom(data, dpo_loss_fn).result()
        dpo_metrics = backward_result.metrics

        # Optimizer step
        training_client.optim_step(adam_params).result()

        # Prepare metrics
        metrics.update(
            num_pairs=len(chosen_data),
            num_tokens=sum(datum.model_input.length for datum in data),
            learning_rate=learning_rate,
            progress=(batch_idx + 1) / n_batches,
            batch_time=time.time() - batch_start_time,
            **dpo_metrics,
        )

        # Evaluation
        if config.eval_every > 0 and batch_idx % config.eval_every == 0:
            eval_metrics = asyncio.run(run_evals(evaluators, training_client, batch_idx))
            metrics.update(eval_metrics)

        # Log metrics
        ml_logger.log_metrics(metrics=metrics, step=batch_idx)

    # Save final checkpoint if training actually happened
    if start_batch < n_batches:
        _ = checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"batch": n_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("DPO training completed successfully")


def print_example(datum: types.Datum, tokenizer: Tokenizer, label: str = ""):
    """Print a formatted example from the dataset."""
    int_tokens = list(datum.model_input.to_ints())
    weights = datum.loss_fn_inputs["weights"].data
    logger.info(f"\n{label} Example:")
    logger.info(format_colorized(int_tokens, cast(list[float], weights), tokenizer))
