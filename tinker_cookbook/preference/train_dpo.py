"""
Direct Preference Optimization (DPO) training
"""

import asyncio
import logging
import os
import time
from typing import Any, cast

import chz
import tinker
import torch
import torch.nn.functional as F
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.eval.evaluators import Evaluator, EvaluatorBuilder
from tinker_cookbook.supervised.train import run_evals
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier, LRSchedule
from tinker_cookbook.utils.misc_utils import timed

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
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1
    dpo_beta: float = 0.1

    # Model parameters
    lora_rank: int = 32

    # Infrastructure parameters
    num_replicas: int = 8
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

    # DPO-specific parameters
    reference_model_name: str | None = None


def create_dpo_clients(
    config: Config,
    resume_info: dict[str, Any] | None = None,
) -> tuple[tinker.TrainingClient, tinker.SamplingClient]:
    """Create and configure the training client and reference sampling client for DPO.

    Creates the main training client and a reference sampling client.
    The reference sampling client is used to compute the reference model's log probabilities
    for the DPO loss computation more efficiently than a separate training client.

    Args:
        config: DPO configuration object
        resume_info: Resume information from checkpoint

    Returns:
        Tuple of (main training client, reference sampling client)
    """
    # Create shared service client for both training and reference clients
    service_client = tinker.ServiceClient(base_url=config.base_url)
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )

    # Load state - differentiate between resuming DPO training vs starting fresh from SFT
    if resume_info:
        # Resuming interrupted DPO training - load optimizer state for proper continuation
        training_client.load_state_with_optimizer(resume_info["state_path"]).result()
        logger.info(f"Resumed DPO training from {resume_info['state_path']}")
    elif config.load_checkpoint_path:
        # Starting fresh DPO from SFT checkpoint - load weights only (fresh optimizer)
        training_client.load_state(config.load_checkpoint_path).result()
        logger.info(f"Loaded weights from {config.load_checkpoint_path}")
    # Create a sampling client for the reference model from the training client
    reference_client = training_client.save_weights_and_get_sampling_client("reference")
    return training_client, reference_client


def compute_dpo_loss(
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    dpo_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
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
    losses = -F.logsigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio))
    loss = losses.mean()

    # Compute metrics
    accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
    chosen_rewards = dpo_beta * chosen_log_ratio
    rejected_rewards = dpo_beta * rejected_log_ratio
    margin = (chosen_rewards - rejected_rewards).mean().item()

    metrics = {
        "dpo_loss": loss.item(),
        "accuracy": accuracy,
        "margin": margin,
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }

    return loss, metrics


def do_update(
    epoch_idx: int,
    batch_idx: int,
    n_batches: int,
    total_steps: int,
    config: Config,
    training_client: tinker.TrainingClient,
    reference_client: tinker.SamplingClient,
    evaluators: list[Evaluator],
    infrequent_evaluators: list[Evaluator],
    dataset: SupervisedDataset,
    ml_logger: ml_log.Logger,
    log_path: str,
    tokenizer: Tokenizer,
):
    """Perform a single DPO training update step."""
    start_time = time.time()
    step = epoch_idx * n_batches + batch_idx
    metrics: dict[str, int | float | str] = {"epoch": epoch_idx}

    # Save checkpoint if needed
    if config.save_every > 0 and step % config.save_every == 0 and step > 0:
        with timed("save_checkpoint", metrics):
            save_result = checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=log_path,
                kind="both",
                loop_state={"epoch": epoch_idx, "batch": batch_idx},
            )
        if "state_path" in save_result:
            metrics["state_path"] = save_result["state_path"]

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

    # Split data into chosen and rejected pairs
    chosen_data = [datum for i, datum in enumerate(data) if i % 2 == 0]
    rejected_data = [datum for i, datum in enumerate(data) if i % 2 == 1]

    # Print example for first batch
    if step == 0:
        for i in range(min(10, len(chosen_data))):
            print_example(chosen_data[i], tokenizer, "Chosen")
            print_example(rejected_data[i], tokenizer, "Rejected")

    with timed("get_ref_logprobs", metrics):
        # Get reference log probabilities using synchronous compute_logprobs
        # Need to reconstruct full sequences for the sampling client
        full_sequences = []
        for datum in data:
            # Reconstruct the full sequence by appending the last target token
            target_tokens = datum.loss_fn_inputs["target_tokens"].data
            if target_tokens:
                full_sequence = datum.model_input.append_int(int(target_tokens[-1]))
                full_sequences.append(full_sequence)
            else:
                # If no target tokens, just use the model input as is
                full_sequences.append(datum.model_input)

        # Compute reference log probabilities in parallel
        async def compute_all_ref_logprobs():
            return await asyncio.gather(
                *[reference_client.compute_logprobs_async(seq) for seq in full_sequences]
            )

        all_ref_logprobs = asyncio.run(compute_all_ref_logprobs())

        # Extract the relevant logprobs (skip the first token which is the prompt)
        all_ref_logprob_seqs = [torch.tensor(logprobs[1:]) for logprobs in all_ref_logprobs]

        # Split reference results into chosen and rejected
        chosen_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(0, len(data), 2)]
        rejected_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(1, len(data), 2)]

    # Create DPO loss function
    def dpo_loss_fn(
        data: list[tinker.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
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
            chosen_ref_logprob = torch.dot(chosen_ref_logprob_seq.float(), chosen_weights.float())
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

    with timed("step", metrics):
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
        progress=step / total_steps,
        **dpo_metrics,
    )

    # Log metrics
    metrics["time/total"] = time.time() - start_time
    ml_logger.log_metrics(metrics=metrics, step=step)


def main(config: Config):
    """Main training function that runs the complete DPO training process."""
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_epoch = resume_info["epoch"]
        start_batch = resume_info["batch"]
    else:
        start_epoch = 0
        start_batch = 0

    # Setup
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )
    training_client, reference_client = create_dpo_clients(config, resume_info)
    tokenizer = get_tokenizer(config.model_name)

    # Training setup
    dataset, maybe_test_dataset = config.dataset_builder()
    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs

    evaluators = [evaluator() for evaluator in config.evaluator_builders]
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
                reference_client=reference_client,
                evaluators=evaluators,
                infrequent_evaluators=infrequent_evaluators,
                dataset=dataset,
                ml_logger=ml_logger,
                log_path=config.log_path,
                tokenizer=tokenizer,
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
    logger.info("DPO training completed successfully")


def print_example(datum: tinker.Datum, tokenizer: Tokenizer, label: str = ""):
    """Print a formatted example from the dataset."""
    int_tokens = list(datum.model_input.to_ints())
    weights = datum.loss_fn_inputs["weights"].data
    logger.info(f"\n{label} Example:")
    logger.info(format_colorized(int_tokens, cast(list[float], weights), tokenizer))
