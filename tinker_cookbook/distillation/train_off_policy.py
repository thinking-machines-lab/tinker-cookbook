"""Off-policy distillation with soft targets from teacher model(s).

Two distillation approaches are available in ``tinker_cookbook.distillation``:

**On-policy** (``train_on_policy.py``):
  The student generates its own rollouts. Teacher logprobs are computed
  on the student's sampled tokens and used as a KL penalty on the RL
  advantages. The student learns from its own distribution.

**Off-policy** (this module):
  The student trains on fixed data (e.g., an SFT data mix). At each token
  position, the teacher's highest-probability tokens are used as soft
  targets for cross-entropy. The student learns to match the teacher's
  distribution on ground-truth data. No rollouts, no sampling from the
  student.

Off-policy supports both single-teacher and multi-teacher configurations:

  Single teacher::

      config = Config(
          dataset_configs=[
              DatasetWithTeacher(dataset_builder=sft_builder, teacher_config=teacher),
          ],
      )

  Multi-teacher (MOPD — each domain has its own teacher)::

      config = Config(
          dataset_configs=[
              DatasetWithTeacher(dataset_builder=code_sft, teacher_config=code_teacher),
              DatasetWithTeacher(dataset_builder=math_sft, teacher_config=math_teacher),
              DatasetWithTeacher(dataset_builder=chat_sft, teacher_config=chat_teacher),
          ],
      )

  The training loop routes each example to its domain-specific teacher.
  All teacher forward passes run concurrently for speed.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import chz
import tinker
import torch

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.distillation.datasets import TeacherConfig
from tinker_cookbook.eval.evaluators import SamplingClientEvaluatorBuilder
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.misc_utils import safezip

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@chz.chz
class DatasetWithTeacher:
    """Pairs a supervised dataset with its teacher model.

    Each domain in multi-teacher distillation gets one of these. The
    teacher's highest-probability tokens are used as soft targets for training.

    Args:
        dataset_builder: Builds the supervised dataset for this domain.
        teacher_config: Teacher model to distill from.
        weight: Relative sampling weight when blending multiple domains.
    """

    dataset_builder: SupervisedDatasetBuilder
    teacher_config: TeacherConfig
    weight: float = 1.0


@chz.chz
class Config:
    """Configuration for off-policy distillation with soft teacher targets.

    Args:
        learning_rate: Optimizer learning rate.
        dataset_configs: One per domain. Each pairs a dataset with a teacher.
        model_name: Student model name.
        n_teacher_targets: Number of teacher tokens per position used as soft targets.
        teacher_concurrency: Max concurrent teacher forward passes.
        batch_size: Number of examples per training step.
    """

    learning_rate: float
    dataset_configs: list[DatasetWithTeacher]
    model_name: str
    renderer_name: str | None = None
    lora_rank: int = 32

    # Distillation parameters
    n_teacher_targets: int = 20
    """Number of highest-probability teacher tokens per position used as soft targets."""
    teacher_concurrency: int = 64
    """Max concurrent teacher forward passes per batch."""

    # Training parameters
    batch_size: int = 64
    """Number of examples per training step."""

    # Checkpointing and logging
    save_every: int = 10
    eval_every: int = 20
    max_steps: int | None = None
    load_checkpoint_path: str | None = None
    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    wandb_project: str | None = None
    wandb_name: str | None = None
    base_url: str | None = None
    ttl_seconds: int | None = 604800
    """Server-side checkpoint retention (seconds). None = keep indefinitely."""

    enable_trace: bool = False

    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)


# ---------------------------------------------------------------------------
# Teacher soft-target collection
# ---------------------------------------------------------------------------


async def _collect_topk_for_datum(
    teacher_client: tinker.SamplingClient,
    datum: tinker.Datum,
    n_teacher_targets: int,
) -> tinker.Datum:
    """Replace a datum's (N,) targets with (N, n_teacher_targets) soft targets from the teacher.

    Teacher-forces the full sequence through the teacher model and reads
    the highest-probability (token_id, logprob) pairs at each position.
    The teacher probabilities are renormalized over the selected set and
    used as soft target weights.

    Args:
        teacher_client: Sampling client for the teacher model.
        datum: Original SFT datum with (N,) target_tokens and weights.
        n_teacher_targets: Number of top teacher tokens per position.

    Returns:
        New datum with (N, n_teacher_targets) shaped target_tokens and weights.
    """
    # The datum's model_input is the shifted input (tokens 0..N-1).
    # target_tokens contains the next-token targets (tokens 1..N).
    # To teacher-force, we need the full unshifted sequence.
    original_targets = datum.loss_fn_inputs["target_tokens"].to_torch()
    original_weights = datum.loss_fn_inputs["weights"].to_torch()
    seq_len = len(original_targets)

    # Build full sequence: input tokens + last target token
    last_target = int(original_targets[-1].item())
    full_sequence = datum.model_input.append_int(last_target)

    response = await teacher_client.sample_async(
        prompt=full_sequence,
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=n_teacher_targets,
    )

    topk_prompt = response.topk_prompt_logprobs
    if topk_prompt is None:
        logger.warning("Teacher returned no prompt logprobs, falling back to hard targets")
        return datum

    # topk_prompt has one entry per input token. The target at position t
    # corresponds to predicting the token at position t+1, so we need the
    # last seq_len entries to align with the target sequence.
    offset = len(topk_prompt) - seq_len
    relevant_topk = topk_prompt[offset:]

    # Unpack teacher response into tensors in one shot.
    # The offset calculation above skips prompt position 0 (which has no
    # logprobs), so all entries in relevant_topk should be non-None.
    token_lists: list[list[int]] = []
    logprob_lists: list[list[float]] = []
    for position_topk in relevant_topk:
        assert position_topk is not None, "Unexpected None in teacher logprobs after offset"
        entries = position_topk[:n_teacher_targets]
        toks: list[int] = []
        lps: list[float] = []
        for tok, lp in entries:
            toks.append(tok)
            lps.append(lp)
        # Pad if fewer entries returned
        toks += [toks[0]] * (n_teacher_targets - len(toks))
        lps += [float("-inf")] * (n_teacher_targets - len(lps))
        token_lists.append(toks)
        logprob_lists.append(lps)

    topk_tokens = torch.tensor(token_lists, dtype=torch.long)
    topk_logprobs = torch.tensor(logprob_lists)

    # Renormalize over selected teacher tokens via logsumexp
    topk_logprobs -= torch.logsumexp(topk_logprobs, dim=-1, keepdim=True)
    topk_weights = topk_logprobs.exp()

    # Zero out positions where original weight was 0 (e.g., prompt tokens)
    position_mask = (original_weights > 0).float().unsqueeze(-1)  # (N, 1)
    topk_weights = topk_weights * position_mask

    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(topk_tokens),
            "weights": tinker.TensorData.from_torch(topk_weights),
        },
    )


async def _collect_topk_batch(
    teacher_clients: list[tinker.SamplingClient],
    datums: list[tinker.Datum],
    n_teacher_targets: int,
    concurrency: int = 64,
) -> list[tinker.Datum]:
    """Collect soft targets for a batch of datums, concurrently.

    Each datum is paired with its domain-specific teacher client. All
    teacher forward passes run concurrently, bounded by a semaphore.

    Args:
        teacher_clients: One teacher client per datum (may repeat for
            datums from the same domain).
        datums: Original SFT datums with (N,) targets.
        n_teacher_targets: Number of top teacher tokens per position.
        concurrency: Max concurrent teacher requests.

    Returns:
        Datums with (N, n_teacher_targets) shaped target_tokens and weights.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _one(client: tinker.SamplingClient, datum: tinker.Datum) -> tinker.Datum:
        async with sem:
            return await _collect_topk_for_datum(client, datum, n_teacher_targets)

    return list(
        await asyncio.gather(
            *[_one(client, datum) for client, datum in safezip(teacher_clients, datums)]
        )
    )


# ---------------------------------------------------------------------------
# Composite supervised dataset with domain routing
# ---------------------------------------------------------------------------


class _CompositeSupervisedDataset:
    """Wraps multiple supervised datasets with domain-aware batching.

    Each ``get_batch`` returns datums from all domains according to their
    weights, along with a list of teacher indices for routing.
    """

    def __init__(
        self,
        datasets: list[SupervisedDataset],
        weights: list[float],
        batch_size: int,
    ):
        self._datasets = datasets
        self._batch_size = batch_size

        if batch_size < len(weights):
            raise ConfigurationError(
                f"batch_size ({batch_size}) must be >= number of domains ({len(weights)})"
            )

        total_weight = sum(weights)
        self._counts: list[int] = []
        remaining = batch_size
        for i, w in enumerate(weights):
            if i == len(weights) - 1:
                self._counts.append(remaining)
            else:
                count = max(1, round(batch_size * w / total_weight))
                self._counts.append(count)
                remaining -= count

        self._length = min(len(ds) for ds in datasets) if datasets else 0

    def get_batch(self, index: int) -> tuple[list[tinker.Datum], list[int]]:
        """Get a mixed batch with domain indices.

        Returns:
            datums: List of training datums from all domains.
            teacher_indices: Domain index for each datum (for teacher routing).
        """
        all_datums: list[tinker.Datum] = []
        all_indices: list[int] = []

        for domain_idx, (dataset, count) in enumerate(safezip(self._datasets, self._counts)):
            batch = dataset.get_batch(index)
            # Take up to `count` datums from this domain
            selected = batch[:count]
            all_datums.extend(selected)
            all_indices.extend([domain_idx] * len(selected))

        return all_datums, all_indices

    def __len__(self) -> int:
        return self._length


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


async def main(config: Config) -> None:
    """Run off-policy distillation with soft teacher targets.

    For each training step:
    1. Get a batch of SFT datums from the composite dataset
    2. Route each datum to its domain-specific teacher
    3. Collect teacher logprobs (concurrently)
    4. Replace hard targets with soft teacher targets
    5. Train student with cross_entropy on (N, n_teacher_targets) targets
    """
    if not config.dataset_configs:
        raise ConfigurationError("At least one dataset_config is required")

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=config.wandb_name,
    )
    if config.enable_trace:
        trace_events_path = str(Path(config.log_path) / "trace_events.jsonl")
        logger.info(f"Tracing enabled. Events saved to {trace_events_path}")
        trace.trace_init(output_file=trace_events_path)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    start_batch = resume_info.batch if resume_info else 0

    service_client = tinker.ServiceClient(base_url=config.base_url)
    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link

    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, renderer_name)

    if resume_info:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, resume_info.state_path, renderer_name
        )
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path, user_metadata=user_metadata
            )
        )
        logger.info(f"Resumed training from {resume_info.state_path}")
    elif config.load_checkpoint_path:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, config.load_checkpoint_path, renderer_name
        )
        training_client = await service_client.create_training_client_from_state_async(
            config.load_checkpoint_path, user_metadata=user_metadata
        )
        logger.info(f"Loaded weights from {config.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            config.model_name, rank=config.lora_rank, user_metadata=user_metadata
        )

    teacher_clients: list[tinker.SamplingClient] = []
    datasets: list[SupervisedDataset] = []
    weights: list[float] = []
    for dc in config.dataset_configs:
        tc = dc.teacher_config
        if tc.load_checkpoint_path:
            client = service_client.create_sampling_client(
                base_model=tc.base_model, model_path=tc.load_checkpoint_path
            )
        else:
            client = service_client.create_sampling_client(base_model=tc.base_model)
        teacher_clients.append(client)
        logger.info(f"Teacher: {tc.base_model} (checkpoint: {tc.load_checkpoint_path})")

        train_ds, _ = dc.dataset_builder()
        datasets.append(train_ds)
        weights.append(dc.weight)
        logger.info(f"Dataset: {len(train_ds)} batches, weight={dc.weight}")

    composite = _CompositeSupervisedDataset(datasets, weights, config.batch_size)
    total_batches = len(composite)
    if config.max_steps is not None:
        total_batches = min(config.max_steps, total_batches)
    logger.info(
        f"Will train for {total_batches} steps, n_teacher_targets={config.n_teacher_targets}"
    )

    evaluators = [eb() for eb in config.evaluator_builders]

    sampling_client: tinker.SamplingClient | None = None

    for i_batch in range(start_batch, total_batches):
        metrics: dict[str, Any] = {
            "step": i_batch,
            "progress/done_frac": (i_batch + 1) / total_batches,
            "optim/lr": config.learning_rate,
        }

        if config.eval_every > 0 and i_batch % config.eval_every == 0 and evaluators:
            if sampling_client is None:
                sampling_client = await training_client.save_weights_and_get_sampling_client_async()
            for evaluator in evaluators:
                eval_metrics = await evaluator(sampling_client)
                metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        datums, teacher_indices = composite.get_batch(i_batch)
        metrics["batch_size"] = len(datums)

        datum_teacher_clients = [teacher_clients[idx] for idx in teacher_indices]

        logger.info(
            f"Step {i_batch}: collecting {config.n_teacher_targets} teacher targets "
            f"from {len(datums)} examples ({len(set(teacher_indices))} teachers)"
        )
        topk_datums = await _collect_topk_batch(
            datum_teacher_clients, datums, config.n_teacher_targets, config.teacher_concurrency
        )

        fwd_bwd_future = await training_client.forward_backward_async(
            topk_datums, loss_fn="cross_entropy"
        )
        optim_future = await training_client.optim_step_async(
            tinker.AdamParams(learning_rate=config.learning_rate)
        )
        train_result = await fwd_bwd_future.result_async()
        await optim_future.result_async()
        if train_result.metrics:
            metrics.update({f"train/{k}": v for k, v in train_result.metrics.items()})

        if config.save_every > 0 and i_batch > start_batch and i_batch % config.save_every == 0:
            path_dict = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{i_batch:06d}",
                log_path=config.log_path,
                kind="both",
                loop_state={"batch": i_batch},
                ttl_seconds=config.ttl_seconds,
            )
            sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
        else:
            # Invalidate; will be lazily created if needed for eval
            sampling_client = None

        ml_logger.log_metrics(metrics, step=i_batch)
        logger.info(
            f"Step {i_batch}: loss={train_result.metrics.get('total_loss', float('nan')):.4f}"
        )

    # Final checkpoint
    if start_batch < total_batches:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"batch": total_batches},
            ttl_seconds=None,
        )

    ml_logger.close()
    logger.info("Off-policy distillation completed successfully")
