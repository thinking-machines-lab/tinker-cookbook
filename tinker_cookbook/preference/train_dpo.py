"""
Direct Preference Optimization (DPO) training
"""

import asyncio
import logging
from pathlib import Path
from typing import cast

import chz
import tinker
import torch
import torch.nn.functional as F

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.eval.evaluators import Evaluator, EvaluatorBuilder
from tinker_cookbook.supervised.train import run_evals
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import iteration_dir

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    """Configuration for Direct Preference Optimization (DPO) training."""

    # Required parameters
    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    model_name: str
    dataset_builder: ChatDatasetBuilder
    load_checkpoint_path: str | None = None
    renderer_name: str | None = None
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
    ttl_seconds: int | None = 604800  # 7 days
    # Rolling checkpoint cadence (0 = disabled). Saves training state for resume
    # but skips the sampler-weight export, making it cheaper than periodic checkpoints.
    rolling_save_every: int = 0
    # TTL for rolling checkpoints; short to auto-clean if explicit deletion fails.
    rolling_ttl_seconds: int = 7200  # 2 hours

    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Logging parameters
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Profiling
    enable_trace: bool = False
    span_chart_every: int = 0

    # DPO-specific parameters
    reference_model_name: str | None = None

    # Maximum number of training steps. If None, train for num_epochs * n_batches.
    max_steps: int | None = None


def create_dpo_clients(
    config: Config,
    resume_info: checkpoint_utils.CheckpointRecord | None = None,
    user_metadata: dict[str, str] | None = None,
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

    if resume_info:
        # Resuming interrupted DPO training - load weights + optimizer state
        assert resume_info.state_path is not None
        checkpoint_utils.check_renderer_name_for_checkpoint(
            service_client, resume_info.state_path, config.renderer_name
        )
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info.state_path, user_metadata=user_metadata
        )
        logger.info(f"Resumed DPO training from {resume_info.state_path}")
    elif config.load_checkpoint_path:
        # Starting fresh DPO from checkpoint - load weights only (fresh optimizer)
        checkpoint_utils.check_renderer_name_for_checkpoint(
            service_client, config.load_checkpoint_path, config.renderer_name
        )
        training_client = service_client.create_training_client_from_state(
            config.load_checkpoint_path, user_metadata=user_metadata
        )
        logger.info(f"Loaded weights from {config.load_checkpoint_path}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank, user_metadata=user_metadata
        )
    # Create a sampling client for the reference model from the training client
    reference_client = training_client.save_weights_and_get_sampling_client()
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
    rolling_mgr: checkpoint_utils.RollingCheckpointManager | None = None,
):
    """Perform a single DPO training update step."""
    step = epoch_idx * n_batches + batch_idx
    metrics: dict[str, int | float | str] = {"epoch": epoch_idx}

    with trace.trace_iteration(step=step) as window:
        # Save checkpoint if needed
        if config.save_every > 0 and step % config.save_every == 0 and step > 0:
            with trace.scope_span_sync("save_checkpoint"):
                save_result = checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{step:06d}",
                    log_path=log_path,
                    kind="both",
                    loop_state={"epoch": epoch_idx, "batch": batch_idx},
                    ttl_seconds=config.ttl_seconds,
                )
            if "state_path" in save_result:
                metrics["state_path"] = save_result["state_path"]

        if rolling_mgr is not None:
            rolling_mgr.maybe_save(step=step, loop_state={"epoch": epoch_idx, "batch": batch_idx})

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
            with trace.scope_span_sync("evals"):
                eval_metrics = asyncio.run(run_evals(evaluators, training_client, step))
            metrics.update(eval_metrics)

        if config.infrequent_eval_every > 0 and step % config.infrequent_eval_every == 0:
            with trace.scope_span_sync("infrequent_evals"):
                eval_metrics = asyncio.run(run_evals(infrequent_evaluators, training_client, step))
            metrics.update(eval_metrics)

        # Prepare batch
        with trace.scope_span_sync("get_batch"):
            data = dataset.get_batch(batch_idx)

        # Split data into chosen and rejected pairs
        chosen_data = [datum for i, datum in enumerate(data) if i % 2 == 0]
        rejected_data = [datum for i, datum in enumerate(data) if i % 2 == 1]

        # Print example for first batch
        if step == 0:
            for i in range(min(10, len(chosen_data))):
                print_example(chosen_data[i], tokenizer, "Chosen")
                print_example(rejected_data[i], tokenizer, "Rejected")

        with trace.scope_span_sync("get_ref_logprobs"):
            # Get reference log probabilities
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

        with trace.scope_span_sync("step"):
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

    # Log timing metrics from trace_iteration window
    metrics.update(window.get_timing_metrics())
    window.write_spans_jsonl(Path(log_path) / "timing_spans.jsonl", step=step)
    if config.span_chart_every > 0 and step % config.span_chart_every == 0:
        iter_dir = iteration_dir(log_path, step)
        if iter_dir is not None:
            iter_dir.mkdir(parents=True, exist_ok=True)
            trace.save_gantt_chart_html(window, step, iter_dir / "timing_gantt.html")
    ml_logger.log_metrics(metrics=metrics, step=step)


def main(config: Config):
    """Main training function that runs the complete DPO training process."""
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_epoch = resume_info.epoch or 0
        start_batch = resume_info.batch
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
    if config.enable_trace:
        trace_events_path = str(Path(config.log_path) / "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace.trace_init(output_file=trace_events_path)

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)
    model_info.warn_if_renderer_not_recommended(config.model_name, config.renderer_name)
    training_client, reference_client = create_dpo_clients(config, resume_info, user_metadata)
    service_client = tinker.ServiceClient(base_url=config.base_url)
    rolling_mgr = checkpoint_utils.RollingCheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=config.log_path,
        rolling_save_every=config.rolling_save_every,
        save_every=config.save_every,
        rolling_ttl_seconds=config.rolling_ttl_seconds,
    )
    tokenizer = get_tokenizer(config.model_name)

    # Training setup
    dataset, maybe_test_dataset = config.dataset_builder()
    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    evaluators = [evaluator() for evaluator in config.evaluator_builders]
    infrequent_evaluators = [evaluator() for evaluator in config.infrequent_evaluator_builders]
    logger.info(
        f"Training for {n_batches} batches x {config.num_epochs} epochs = {n_batches * config.num_epochs} steps"
    )

    # Training loop
    reached_max_steps = False
    for epoch_idx in range(start_epoch, config.num_epochs):
        # Shuffle the dataset
        logger.info(msg=f"Starting epoch {epoch_idx}")
        dataset.set_epoch(seed=epoch_idx)

        for batch_idx in range(start_batch if epoch_idx == start_epoch else 0, n_batches):
            step = epoch_idx * n_batches + batch_idx
            if config.max_steps is not None and step >= config.max_steps:
                reached_max_steps = True
                break
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
                rolling_mgr=rolling_mgr,
            )
        if reached_max_steps:
            break

    # Save final checkpoint if training actually happened
    did_train = start_epoch < config.num_epochs and (
        config.max_steps is None or start_epoch * n_batches + start_batch < config.max_steps
    )
    if did_train:
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"epoch": config.num_epochs, "batch": 0},
            ttl_seconds=None,
        )
    else:
        logger.info("Training was already complete; nothing to do")
    # Clean up rolling checkpoints after the final save so that the last
    # entry in checkpoints.jsonl always points to valid server-side data.
    rolling_mgr.finalize()

    # Cleanup
    ml_logger.close()
    logger.info("DPO training completed successfully")


def print_example(datum: tinker.Datum, tokenizer: Tokenizer, label: str = ""):
    """Print a formatted example from the dataset."""
    int_tokens = list(datum.model_input.to_ints())
    weights = datum.loss_fn_inputs["weights"].data
    logger.info(f"\n{label} Example:")
    logger.info(format_colorized(int_tokens, cast(list[float], weights), tokenizer))
