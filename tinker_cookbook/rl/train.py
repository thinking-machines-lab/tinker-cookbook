"""
Implements RL on general MDPs
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import time
from collections.abc import Callable, Coroutine, Iterable, Iterator, Sequence
from concurrent.futures import Executor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from tinker_cookbook.stores.training_store import TrainingRunStore

import chz
import numpy as np
import tinker
import torch
from fireworks.training.sdk import (
    DeploymentManager,
    DeploymentSampler,
    FiretitanServiceClient,
    FiretitanTrainingClient,
    WeightSyncer,
)
from tinker.types import LossFnType
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import (
    SamplingClientEvaluator,
    SamplingClientEvaluatorBuilder,
)
from tinker_cookbook.exceptions import ConfigurationError

# from tinker_cookbook.rl.custom import "ppo"
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.metric_util import (
    RLTestSetEvaluator,
    compute_trajectory_metrics,
)
from tinker_cookbook.rl.metrics import (
    compute_kl_sample_train,
    compute_post_kl,
    compute_sampling_client_metrics,
    incorporate_kl_penalty,
)
from tinker_cookbook.rl.rollout_logging import (
    RolloutSummaryExportConfig,
    RolloutSummaryGroup,
)
from tinker_cookbook.rl.rollout_strategy import (
    RolloutStrategy,
    rollout_strategy_from_config,
)
from tinker_cookbook.rl.rollouts import (  # noqa: F401 — re-exported for verifiers monkey-patching
    RolloutErrorCounter,
    do_group_rollout,
    do_group_rollout_and_filter_constant_reward,
    set_rollout_executor,
)
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import logtree, ml_log, trace
from tinker_cookbook.utils.misc_utils import iteration_dir, safezip, split_list

logger = logging.getLogger(__name__)

T = TypeVar("T")


@chz.chz
class KLReferenceConfig:
    """Configuration for the KL penalty reference model (frozen ``FiretitanTrainingClient``).

    If not specified in Config, the training model's base model is used.

    Attributes:
        base_model (str): Name of the base model to use as the KL reference.
        load_checkpoint_path (str | None): Optional checkpoint path to load
            reference model weights from. If None, the base model weights
            are used directly.
    """

    base_model: str
    load_checkpoint_path: str | None = None


async def gather_with_progress(
    coroutines: Iterable[Coroutine[Any, Any, T]],
    desc: str,
    max_concurrency: int = 4
) -> list[T]:
    """Run coroutines concurrently with a progress bar that updates as each completes.

    This preserves the order of results (like asyncio.gather) while providing
    real-time progress feedback as individual coroutines complete.

    Args:
        coroutines (Iterable[Coroutine[Any, Any, T]]): Coroutines to run concurrently.
        desc (str): Description label for the tqdm progress bar.

    Returns:
        list[T]: Results from each coroutine, in the same order as the input.
    """
    coroutine_list = list(coroutines)
    pbar = tqdm(total=len(coroutine_list), desc=desc)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def track(coro: Coroutine[Any, Any, T]) -> T:
        if semaphore:
            async with semaphore:
                result = await coro
        else:
            result = await coro
        pbar.update(1)
        return result

    try:
        results = await asyncio.gather(*[track(coro) for coro in coroutine_list])
    finally:
        pbar.close()

    return results


def _get_evaluator_name(evaluator: SamplingClientEvaluator) -> str:
    return (
        evaluator.name
        if isinstance(evaluator, RLTestSetEvaluator) and evaluator.name is not None
        else ""
    )


def _sanitize_filename_component(text: str) -> str:
    """Make a safe filename component."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return sanitized.strip("._") or "unnamed"


def _maybe_export_rollout_summary_jsonl(
    *,
    config: Config,
    base_name: str,
    split: str,
    iteration: int,
    groups_P: Sequence[RolloutSummaryGroup],
    store: TrainingRunStore | None,
) -> None:
    """Write per-trajectory rollout summaries via the store when enabled."""
    if not config.rollout_json_export or store is None:
        return
    from tinker_cookbook.rl.rollout_logging import (
        serialize_rollout_summaries_from_groups,
    )

    records = serialize_rollout_summaries_from_groups(
        split=split, iteration=iteration, groups_P=groups_P
    )
    store.write_rollouts(iteration, records, base_name=base_name)


_LOGTREE_EXPLANATION = (
    "This HTML log was generated by logtree during RL training. "
    "It shows rollouts and rewards for a subset of trajectory groups in this iteration. "
    "To customize what gets logged, modify the logtree calls in your Env implementation "
    "(see examples in tinker_cookbook/recipes/)."
)


@contextmanager
def _get_logtree_scope(
    output_dir: Path | None,
    num_groups_to_log: int,
    f_name: str,
    scope_name: str,
    iteration: int,
    store: TrainingRunStore | None,
) -> Iterator[None]:
    """Context manager that logs rollout data to HTML and JSON via logtree.

    Creates ``output_dir/f_name.html`` (direct I/O — visualization artifact)
    and writes the logtree JSON via ``store.write_logtree()`` when store is available.
    """
    if output_dir is None or num_groups_to_log <= 0:
        yield
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logtree_path = str(output_dir / f"{f_name}.html")
    logtree_trace = None
    try:
        with logtree.init_trace(scope_name, path=logtree_path) as logtree_trace:
            logtree.log_text(_LOGTREE_EXPLANATION)
            yield
    finally:
        if logtree_trace is not None and store is not None:
            store.write_logtree(iteration, logtree_trace.to_dict(), base_name=f_name)


def _select_representative_inds(scores: list[float], num_inds: int) -> list[int]:
    assert num_inds <= len(scores)
    sorted_inds = np.argsort(scores)
    uniform_inds = np.linspace(0, len(sorted_inds) - 1, num_inds).astype(int)
    return [int(sorted_inds[i]) for i in uniform_inds]


def print_group(traj_group: TrajectoryGroup, tokenizer: Tokenizer):
    """Print a subset of the trajectory group to the console.

    Selects a representative sample of up to 4 trajectories (spanning the
    reward distribution) and logs their tokens, rewards, advantages, and
    per-step metrics via the module logger.

    Args:
        traj_group (TrajectoryGroup): The trajectory group to display.
        tokenizer (Tokenizer): Tokenizer used to decode tokens for display.
    """
    # Cut down the number of trajectories to print
    max_trajs_to_print = 4
    if len(traj_group.trajectories_G) > max_trajs_to_print:
        inds = _select_representative_inds(traj_group.get_total_rewards(), max_trajs_to_print)
        traj_group = TrajectoryGroup(
            trajectories_G=[traj_group.trajectories_G[i] for i in inds],
            final_rewards_G=[traj_group.final_rewards_G[i] for i in inds],
            metrics_G=[traj_group.metrics_G[i] for i in inds],
        )

    rewards = traj_group.get_total_rewards()
    advantages_G = compute_advantages([traj_group])
    data_D, metadata_D = assemble_training_data([traj_group], advantages_G)

    buf = io.StringIO()

    def bprint(s: str):
        print(s, file=buf)

    bprint("\n====== Trajectory Group ======")
    last_metadata = None
    for datum, metadata in safezip(data_D, metadata_D):
        idx = metadata["traj_idx"]
        if metadata != last_metadata:
            bprint(f"****** trajectory idx={idx}, reward={rewards[idx]:.3g} ******")
            # Print trajectory-level metrics
            if traj_group.metrics_G[idx]:
                bprint("Trajectory metrics:")
                for key, value in traj_group.metrics_G[idx].items():
                    bprint(f"  {key}: {value}")
            # Print per-transition metrics
            transition_metrics = [
                transition.metrics
                for transition in traj_group.trajectories_G[idx].transitions
                if transition.metrics
            ]
            if transition_metrics:
                bprint("Per-step metrics:")
                for i, metrics in enumerate(transition_metrics):
                    bprint(f"  Step {i}:")
                    for key, value in metrics.items():
                        bprint(f"    {key}: {value}")
        bprint("---- datum ----")
        bprint(colorize_example(datum, tokenizer, key="advantages"))
        last_metadata = metadata
    bprint("====== End Trajectory Group ======")
    logger.info(buf.getvalue().rstrip())


def _remove_mask(datum: tinker.Datum) -> tinker.Datum:
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


def _cross_entropy_forward_datum(datum: tinker.Datum) -> tinker.Datum:
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={"target_tokens": datum.loss_fn_inputs["target_tokens"]},
    )


def _training_logprobs_from_forward(
    forward_result: tinker.ForwardBackwardOutput,
) -> list[torch.Tensor]:
    return [output["logprobs"].to_torch() for output in forward_result.loss_fn_outputs]


@trace.scope
async def train_step(
    data_D: list[tinker.Datum],
    training_client: FiretitanTrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: LossFnType,
    loss_fn_config: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> list[torch.Tensor]:
    """Train the model on collected trajectories.

    Pipelines ``forward_backward`` and ``optim_step`` so they land on the same
    clock cycle, maximizing GPU utilization. The data is split into
    ``num_substeps`` batches; each batch is enqueued before consuming the
    previous result to keep the pipeline full.

    Args:
        data_D (list[tinker.Datum]): Training data assembled from trajectory
            rollouts, including advantages and log-probabilities.
        training_client (tinker.TrainingClient): Client connected to the
            Tinker training service.
        learning_rate (float): Learning rate for the Adam optimizer.
        num_substeps (int): Number of sub-batches to split data_D into.
            Each sub-batch triggers one forward_backward + optim_step pair.
        loss_fn (LossFnType): Loss function identifier (e.g.
            ``"importance_sampling"``, ``"ppo"``).
        loss_fn_config (dict[str, Any] | None): Extra configuration passed
            to the loss function. Defaults to None.
        metrics (dict[str, Any] | None): If provided, optimizer metrics from
            the final optim_step are merged into this dict in-place.

    Returns:
        list[torch.Tensor]: Per-datum training log-probabilities returned by
        the forward pass, one tensor per datum in data_D.

    Example::

        logprobs = await train_step(
            data_D=data,
            training_client=client,
            learning_rate=1e-5,
            num_substeps=2,
            loss_fn="importance_sampling",
        )
    """
    batches = split_list(data_D, min(num_substeps, len(data_D)))
    if not batches:
        return []

    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    training_logprobs_D: list[torch.Tensor] = []
    optim_result: tinker.OptimStepResponse | None = None

    # Enqueue first batch
    fwd_bwd_future = await training_client.forward_backward_async(
        [_remove_mask(d) for d in batches[0]], loss_fn, loss_fn_config,
    )
    forward_future = await training_client.forward_async(
        [_cross_entropy_forward_datum(d) for d in batches[0]], loss_fn="cross_entropy"
    )
    optim_future = await training_client.optim_step_async(adam_params)

    for i in range(len(batches)):
        # Enqueue next batch before consuming current results (to stay on same clock cycle)
        if i + 1 < len(batches):
            next_fwd_bwd_future = await training_client.forward_backward_async(
                [_remove_mask(d) for d in batches[i + 1]],
                loss_fn,
                loss_fn_config,
            )
            next_forward_future = await training_client.forward_async(
                [_cross_entropy_forward_datum(d) for d in batches[i + 1]],
                loss_fn="cross_entropy",
            )
            next_optim_future = await training_client.optim_step_async(adam_params)
        else:
            next_fwd_bwd_future = None
            next_forward_future = None
            next_optim_future = None
        # Consume current results
        await fwd_bwd_future.result_async()
        forward_result = await forward_future.result_async()
        training_logprobs_D.extend(_training_logprobs_from_forward(forward_result))
        optim_result = await optim_future.result_async()
        # Move to next iteration
        if (
            next_fwd_bwd_future is not None
            and next_forward_future is not None
            and next_optim_future is not None
        ):
            fwd_bwd_future = next_fwd_bwd_future
            forward_future = next_forward_future
            optim_future = next_optim_future

    if metrics is not None and optim_result is not None and optim_result.metrics:
        metrics.update(optim_result.metrics)

    return training_logprobs_D


@chz.chz
class StreamMinibatchConfig:
    """Configuration for training with minibatch streaming.

    Once enough trajectories for a minibatch have been accumulated, training
    begins immediately rather than waiting for the full batch. This overlaps
    sampling and training within a single iteration.

    Attributes:
        groups_per_batch (int): Total number of trajectory groups across all
            minibatches and substeps in one training iteration.
        num_minibatches (int): Number of minibatches per optimizer substep.
            Each minibatch triggers one ``forward_backward()`` call, and one
            ``optim_step()`` is issued per substep.
    """

    # Total number of trajectory groups across all minibatches and substeps
    groups_per_batch: int
    # For each substep, we will divide up the number of trajectory groups
    # into this many minibatches.
    # We will do num_minibatches forward_backward() passes and one optim_step()
    # per substep.
    num_minibatches: int


@chz.chz
class AsyncConfig:
    """Configuration for async RL training.

    In async mode, sampling and training run concurrently. Trajectory groups
    generated from a sampler that is too many steps behind the current
    training step are discarded (or requeued) to limit off-policy staleness.

    Attributes:
        max_steps_off_policy (int): Maximum number of training steps a sample
            can lag behind the current step before being considered stale.
        groups_per_batch (int): Minimum number of trajectory groups required
            to form a training batch, even after discarding stale samples.
    """

    # If samples are generated from a sample more than this many steps ago,
    # we will skip training on them.
    max_steps_off_policy: int
    # We will ensure all batches have at least this many groups, even
    # as we discard stale samples
    groups_per_batch: int


@chz.chz
class Config:
    """Configuration for RL training.

    This is the main configuration object for :func:`main`. It controls the
    model, dataset, optimizer, loss function, KL penalty, evaluation cadence,
    checkpointing, logging, and execution mode (sync, async, or streaming
    minibatch).

    All fields use ``chz`` dataclass semantics and can be overridden via CLI
    or YAML configuration files.
    """

    # -------------------------------------------------------------------------
    # Core parameters (recommended to set for nearly all runs)
    # -------------------------------------------------------------------------
    # Base learning rate used by Adam.
    learning_rate: float
    # Builds the RL dataset; also determines number of groups per batch.
    dataset_builder: RLDatasetBuilder
    # Model name (base weights) to train.
    model_name: str
    # Maximum number of generated tokens per rollout trajectory.
    max_tokens: int
    # Directory for checkpoints, logs, and traces.
    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    # Evaluation cadence in training iterations (0 = disabled).
    eval_every: int = 20
    # Checkpoint cadence in training iterations (0 = disabled).
    save_every: int = 20
    # Optional evaluators run during training.
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    # Start training from weights at this checkpoint (fresh optimizer state).
    load_checkpoint_path: str | None = None
    # Renderer used by the training dataset/environment.
    renderer_name: str | None = None
    # Optional W&B project and run name.
    wandb_project: str | None = None
    wandb_name: str | None = None

    # -------------------------------------------------------------------------
    # KL penalty configuration (advanced)
    # -------------------------------------------------------------------------
    # KL penalty coefficient against reference policy (0 = disabled).
    kl_penalty_coef: float = 0.0
    # Optional position discount for KL penalty terms.
    kl_discount_factor: float = 0.0
    # Required when kl_penalty_coef > 0.
    kl_reference_config: KLReferenceConfig | None = None

    # -------------------------------------------------------------------------
    # Loss and optimizer behavior (advanced)
    # -------------------------------------------------------------------------
    # Loss function and configuration.
    # See https://tinker-docs.thinkingmachines.ai/losses
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Number of optimizer steps per training iteration.
    # Useful for very large batch sizes.
    num_substeps: int = 1
    # LoRA rank for the training adapter.
    lora_rank: int = 32

    # -------------------------------------------------------------------------
    # Sampling and diagnostics (advanced)
    # -------------------------------------------------------------------------
    # Changing sampling temperature is not generally recommended; T=1 is near-optimal
    # for most post-trained models, and non-1 temperatures currently do not play
    # well with KL penalty.
    temperature: float = 1.0
    # Compute extra post-update KL metrics (adds overhead).
    compute_post_kl: bool = False
    # Remove groups where all trajectories have identical reward.
    remove_constant_reward_groups: bool = False
    # Tolerance for errors during rollouts (container crashes, sandbox flakes, etc.).
    # False (default): crash on any error (FailFast).
    # True: retry failed trajectories with default budget (RetryOnFailure(max_retries=3)).
    # RolloutStrategy instance: custom strategy (e.g. RetryOnFailure(max_retries=5)).
    rollout_error_tolerance: bool | RolloutStrategy = False
    # Emit async trace events for debugging/profiling.
    enable_trace: bool = False
    # Save a Gantt chart HTML every N iterations (0 = disabled). Requires plotly.
    span_chart_every: int = 0

    # -------------------------------------------------------------------------
    # Execution mode knobs (advanced)
    # -------------------------------------------------------------------------
    # Enable async/off-policy training mode when set.
    async_config: AsyncConfig | None = None
    # Enable sync training with streaming minibatches when set.
    stream_minibatch_config: StreamMinibatchConfig | None = None
    # Optional service base URL override (primarily internal/dev use).
    base_url: str | None = None

    # -------------------------------------------------------------------------
    # Checkpoint retention and logging detail (advanced)
    # -------------------------------------------------------------------------
    # Periodic checkpoints use this TTL; the final checkpoint is kept indefinitely.
    # None disables expiry entirely.
    ttl_seconds: int | None = 604800  # 7 days
    # Rolling checkpoint cadence (0 = disabled). Saves training state for resume
    # but skips the sampler-weight export, making it cheaper than periodic checkpoints.
    rolling_save_every: int = 0
    # TTL for rolling checkpoints; short to auto-clean if explicit deletion fails.
    rolling_ttl_seconds: int = 7200  # 2 hours
    num_groups_to_log: int = 4  # Number of groups to log per iteration (0 = disable logging)
    rollout_json_export: bool = True

    # Maximum number of training iterations. If None, train on the full dataset.
    max_steps: int | None = None

    fireworks_base_model_name: str | None = None
    fireworks_deployment_id: str | None = None
    fireworks_hot_load_timeout: int = 600

@trace.scope
async def run_single_evaluation(
    evaluator: SamplingClientEvaluator,
    config: Config,
    i_batch: int,
    sampling_client: DeploymentSampler,
    evaluator_label: str,
    store: TrainingRunStore | None = None,
) -> dict[str, Any]:
    """Run a single evaluator and return its metrics.

    Sets up a logtree scope for the evaluation, exports rollout summary JSONL
    when applicable (for ``RLTestSetEvaluator``), and delegates to the
    evaluator callable.

    Args:
        evaluator (SamplingClientEvaluator): The evaluator to run.
        config (Config): RL training configuration (used for logging settings).
        i_batch (int): Current training iteration index.
        sampling_client (tinker.SamplingClient): Sampling client with the
            current model weights.
        evaluator_label (str): Filesystem-safe label used for log file naming.

    Returns:
        dict[str, Any]: Evaluation metrics produced by the evaluator.
    """
    ev_name = _get_evaluator_name(evaluator)
    eval_base_name = f"eval_{evaluator_label}"
    iter_dir = iteration_dir(config.log_path, i_batch)
    with _get_logtree_scope(
        output_dir=iter_dir,
        num_groups_to_log=config.num_groups_to_log,
        f_name=eval_base_name,
        scope_name=f"Running evaluation {ev_name} {i_batch}",
        iteration=i_batch,
        store=store,
    ):
        if isinstance(evaluator, RLTestSetEvaluator):
            rollout_summary_export = (
                RolloutSummaryExportConfig(
                    split=f"eval/{evaluator_label}",
                    iteration=i_batch,
                    base_name=eval_base_name,
                    sampling_client_step=i_batch,
                )
                if config.rollout_json_export and iter_dir is not None
                else None
            )
            eval_metrics = await evaluator(
                sampling_client,
                rollout_summary_export=rollout_summary_export,
                store=store,
            )
        else:
            eval_metrics = await evaluator(sampling_client)
        return eval_metrics


@trace.scope
async def run_evaluations_parallel(
    evaluators: list[SamplingClientEvaluator],
    sampling_client: DeploymentSampler,
    config: Config,
    i_batch: int,
    store: TrainingRunStore | None = None,
) -> dict[str, Any]:
    """Run all evaluators in parallel and return aggregated metrics.

    Each evaluator is launched as an independent ``asyncio.Task``. Results
    are gathered and merged into a single metrics dictionary.

    Args:
        evaluators (list[SamplingClientEvaluator]): Evaluators to execute.
        sampling_client (tinker.SamplingClient): Sampling client with the
            current model weights.
        config (Config): RL training configuration.
        i_batch (int): Current training iteration index.

    Returns:
        dict[str, Any]: Merged metrics from all evaluators.
    """

    # Create tasks for all evaluators with names for better traceability
    tasks = []
    for i, evaluator in enumerate(evaluators):
        ev_name = _get_evaluator_name(evaluator)
        evaluator_label = _sanitize_filename_component(ev_name or str(i))
        task = asyncio.create_task(
            run_single_evaluation(
                evaluator, config, i_batch, sampling_client, evaluator_label, store=store
            ),
            name=f"eval_{evaluator_label}_iteration_{i_batch:06d}",
        )
        tasks.append(task)

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Merge all metrics
    metrics = {}
    for result in results:
        metrics.update(result)

    return metrics


@trace.scope
async def do_sync_training_with_stream_minibatch(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    config: Config,
    training_client: FiretitanTrainingClient,
    kl_reference_client: FiretitanTrainingClient | None,
    weight_syncer: WeightSyncer,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    error_counter: RolloutErrorCounter | None = None,
    strategy: RolloutStrategy | None = None,
    checkpoint_mgr: checkpoint_utils.CheckpointManager | None = None,
):
    """Implement fully synchronous on-policy training with minibatch streaming.

    Once enough trajectories for a minibatch have been accumulated, training
    begins immediately rather than waiting for the full batch. This overlaps
    sampling and training within a single iteration, reducing wall-clock time.

    Args:
        start_batch (int): First training iteration index (inclusive).
        end_batch (int): Last training iteration index (exclusive).
        num_batches (int): Total number of batches in the dataset, used for
            progress fraction calculation.
        config (Config): RL training configuration. Must have
            ``stream_minibatch_config`` set.
        training_client (tinker.TrainingClient): Client connected to the
            Tinker training service.
        kl_reference_client (tinker.SamplingClient | None): Sampling client
            for the KL reference model, or None if KL penalty is disabled.
        evaluators (list[SamplingClientEvaluator]): Evaluators to run
            periodically during training.
        dataset (RLDataset): The RL dataset providing batches of
            ``EnvGroupBuilder`` instances.
        ml_logger (ml_log.Logger): Logger for metrics and W&B integration.
        tokenizer (Tokenizer): Tokenizer for decoding rollout tokens.
        error_counter (RolloutErrorCounter | None): Optional counter for
            tracking rollout errors. Defaults to None.
        strategy (RolloutStrategy | None): Rollout error handling strategy.
            Defaults to None.
    """
    # Initial sampling client
    assert checkpoint_mgr is not None
    if weight_syncer is not None and weight_syncer.base_identity is not None:
        # Weights were already hotloaded during setup (e.g. Fireworks path),
        # skip redundant save+hotload which can crash the deployment.
        sampling_client = weight_syncer.get_deployment_sampler()
    else:
        sampling_client, _ = await save_checkpoint_and_get_sampling_client(
            training_client,
            checkpoint_mgr,
            weight_syncer,
            start_batch,
        )

    for i_batch in range(start_batch, end_batch):
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }

        with trace.trace_iteration(step=i_batch) as window:
            # Run evaluations
            if (
                config.eval_every > 0 and i_batch % config.eval_every == 0
            ) or i_batch == end_batch - 1:
                async with trace.scope_span("run_evals"):
                    eval_metrics = await run_evaluations_parallel(
                        evaluators, sampling_client, config, i_batch, store=ml_logger.store
                    )
                    metrics.update(eval_metrics)

            iter_dir = iteration_dir(config.log_path, i_batch)
            with _get_logtree_scope(
                iter_dir,
                config.num_groups_to_log,
                "train",
                f"RL Iteration {i_batch}",
                iteration=i_batch,
                store=ml_logger.store,
            ):
                # Samplers will produce trajectory groups asynchronously,
                # and the trainer will consume them as soon as they are ready
                trajectory_groups_queue: asyncio.Queue[
                    WrappedTrajectoryGroup | _Shutdown | None
                ] = asyncio.Queue()
                env_group_builders_P = dataset.get_batch(i_batch)

                @trace.scope
                async def trajectory_group_worker_task(
                    builder: EnvGroupBuilder, enable_logging: bool
                ) -> None:
                    worker_metrics: dict[str, Any] = {}
                    t_start = time.time()
                    async with trace.scope_span("trajectory_group_worker"):
                        trajectory_group = await do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            max_tokens=config.max_tokens,
                            temperature=config.temperature,
                            do_remove_constant_reward_groups=config.remove_constant_reward_groups,
                            enable_logging=enable_logging,
                            strategy=strategy,
                        )
                    worker_metrics["time/trajectory_group_worker_loop/total"] = (
                        time.time() - t_start
                    )
                    # Ingest error info (safe: same event loop thread)
                    if error_counter is not None:
                        error_counter.ingest(trajectory_group)
                    if trajectory_group is not None:
                        trajectory_groups_queue.put_nowait(
                            WrappedTrajectoryGroup(
                                trajectory_group=trajectory_group,
                                env_group_builder=builder,
                                sampling_client_step=i_batch,
                                metrics=worker_metrics,
                            )
                        )
                    else:
                        trajectory_groups_queue.put_nowait(None)

                # Sample all trajectories asynchronously. If we have multiple minibatches,
                # then sampling can overlap with training.
                for i, builder in enumerate(env_group_builders_P):
                    asyncio.create_task(
                        trajectory_group_worker_task(
                            builder, enable_logging=i < config.num_groups_to_log
                        ),
                        name=f"trajectory_group_worker_task_{i}",
                    )

                # Run multiple optimizer substeps per training iteration
                streaming_result = await do_train_step_streaming_and_get_sampling_client(
                    config,
                    i_batch,
                    trajectory_groups_queue,
                    training_client,
                    checkpoint_mgr,
                    kl_reference_client,
                    weight_syncer,
                    tokenizer,
                )
                # _Shutdown cannot appear in the sync path's local queue
                assert streaming_result is not None, "Unexpected shutdown in sync streaming path"
                (
                    sampling_client,
                    full_batch_metrics,
                    full_batch_wrapped_trajectory_groups,
                ) = streaming_result

            _maybe_export_rollout_summary_jsonl(
                config=config,
                base_name="train",
                split="train",
                iteration=i_batch,
                groups_P=[
                    RolloutSummaryGroup(
                        trajectory_group=group.trajectory_group,
                        tags=group.env_group_builder.logging_tags(),
                        sampling_client_step=group.sampling_client_step,
                    )
                    for group in full_batch_wrapped_trajectory_groups
                ],
                store=ml_logger.store,
            )

        # Rolling checkpoint (fire-and-forget, overlaps with next iteration)
        if checkpoint_mgr is not None:
            await checkpoint_mgr.maybe_save_rolling_async(
                step=i_batch + 1, loop_state={"batch": i_batch + 1}
            )

        # Log metrics
        metrics.update(full_batch_metrics)
        if error_counter is not None:
            metrics.update(error_counter.get_metrics())
        metrics.update(window.get_timing_metrics())
        window.save_timing(i_batch, store=ml_logger.store)
        if (
            config.span_chart_every > 0
            and i_batch % config.span_chart_every == 0
            and iter_dir is not None
        ):
            iter_dir.mkdir(parents=True, exist_ok=True)
            trace.save_gantt_chart_html(window, i_batch, iter_dir / "timing_gantt.html")
        ml_logger.log_metrics(metrics, step=i_batch)


@chz.chz
class WrappedTrajectoryGroup:
    """A wrapper around a trajectory group that includes generation metadata.

    Used when sampling and training are overlapped (streaming minibatch or
    async modes) so that staleness can be checked and stale groups requeued.

    Attributes:
        trajectory_group (TrajectoryGroup): The collected trajectory group.
        env_group_builder (EnvGroupBuilder): The builder that produced this
            group. Retained so that stale groups can be requeued.
        sampling_client_step (int): The training step at which the sampling
            client was created for this rollout.
        metrics (dict[str, Any]): Timing and worker-level metrics collected
            during rollout generation.
    """

    trajectory_group: TrajectoryGroup
    # The env group builder that produced the trajectory group.
    # Pass this along in case the sampler is too stale, and we need to
    # requeue this group.
    env_group_builder: EnvGroupBuilder
    # The step that produced this trajectory group.
    sampling_client_step: int
    metrics: dict[str, Any] = chz.field(default_factory=dict)


@dataclass
class _Shutdown:
    """Sentinel value to signal graceful shutdown through async queues.

    Used in the cascading shutdown protocol for async RL training:
    dataloader -> workers -> training loop -> evaluation loop.
    """

    pass


class _AsyncCounter:
    """Async-safe counter for tracking the number of alive worker tasks."""

    def __init__(self, start: int):
        self._value = start
        self._lock = asyncio.Lock()

    async def decrement_and_get(self) -> int:
        async with self._lock:
            self._value -= 1
            return self._value


@trace.scope
async def do_async_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    config: Config,
    training_client: FiretitanTrainingClient,
    kl_reference_client: FiretitanTrainingClient | None,
    weight_syncer: WeightSyncer,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    error_counter: RolloutErrorCounter | None = None,
    strategy: RolloutStrategy | None = None,
    checkpoint_mgr: checkpoint_utils.CheckpointManager | None = None,
):
    """Implement async off-policy training, capped at K steps off policy.

    Launches four concurrent coroutine groups that communicate via async
    queues:

    1. **Dataloader loop** -- feeds ``EnvGroupBuilder`` items into a queue.
    2. **Trajectory worker loops** (one per ``groups_per_batch``) -- consume
       builders, run rollouts, and push ``WrappedTrajectoryGroup`` results.
    3. **Training loop** -- accumulates groups, discards stale samples, and
       performs forward_backward + optim_step.
    4. **Evaluation loop** -- runs evaluators whenever the sampling client is
       updated.

    Shutdown cascades from the dataloader through workers, training, and
    finally evaluation via ``_Shutdown`` sentinels and ``asyncio.Event`` flags.

    Args:
        start_batch (int): First training iteration index (inclusive).
        end_batch (int): Last training iteration index (exclusive).
        num_batches (int): Total number of batches, used for progress
            fraction calculation.
        config (Config): RL training configuration. Must have
            ``async_config`` set.
        training_client (tinker.TrainingClient): Client connected to the
            Tinker training service.
        kl_reference_client (tinker.SamplingClient | None): Sampling client
            for the KL reference model, or None if KL penalty is disabled.
        evaluators (list[SamplingClientEvaluator]): Evaluators to run
            periodically during training.
        dataset (RLDataset): The RL dataset providing batches of
            ``EnvGroupBuilder`` instances.
        ml_logger (ml_log.Logger): Logger for metrics and W&B integration.
        tokenizer (Tokenizer): Tokenizer for decoding rollout tokens.
        error_counter (RolloutErrorCounter | None): Optional counter for
            tracking rollout errors. Defaults to None.
        strategy (RolloutStrategy | None): Rollout error handling strategy.
            Defaults to None.
    """
    assert config.async_config is not None

    # We will have groups_per_batch workers generating rollouts, so cap the
    # queue size to be groups_per_batch.
    env_group_builders_queue = asyncio.Queue[EnvGroupBuilder | _Shutdown](
        maxsize=config.async_config.groups_per_batch
    )
    trajectory_groups_queue = asyncio.Queue[WrappedTrajectoryGroup | _Shutdown | None]()

    # Initial sampling client to use
    path_dict = await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name=f"{start_batch:06d}",
        log_path=config.log_path,
        loop_state={"batch": start_batch},
        kind="both",
        ttl_seconds=config.ttl_seconds,
        store=ml_logger.store,
    )

    # Shutdown coordination — cascading sequence:
    # 1. Dataloader exhausts data → sets dataloader_done_event (prevents requeuing stale
    #    samples) and enqueues one _Shutdown sentinel per worker into env_group_builders_queue.
    # 2. Each trajectory worker receives its _Shutdown sentinel → exits and decrements
    #    worker_alive_counter. The last worker enqueues a _Shutdown into trajectory_groups_queue.
    # 3. Training loop receives _Shutdown from trajectory_groups_queue → finishes current
    #    batch, sets evaluation_loop_should_shutdown_event, and exits.
    # 4. Eval loop sees evaluation_loop_should_shutdown_event → exits.
    dataloader_done_event = asyncio.Event()
    evaluation_loop_should_shutdown_event = asyncio.Event()
    worker_alive_counter = _AsyncCounter(config.async_config.groups_per_batch)

    # This will be updated by the training loop
    sampling_client = DeploymentSampler(
        config.model_name,
    )
    sampling_client_step = start_batch
    sampling_client_updated_event = asyncio.Event()
    sampling_client_updated_event.set()

    @trace.scope
    async def dataloader_loop():
        """Gets the next set of env builders to run"""
        i_batch = start_batch
        while i_batch < end_batch:
            env_group_builders_P = dataset.get_batch(i_batch)
            for env_group_builder in env_group_builders_P:
                await env_group_builders_queue.put(env_group_builder)
            i_batch += 1

        # Signal that no more data will be produced, so stale samples should not be requeued
        dataloader_done_event.set()
        # Enqueue shutdown sentinels — one per worker — to cascade the shutdown
        logger.info("[dataloader_loop] No more data, shutting down trajectory group workers")
        assert config.async_config is not None
        for _ in range(config.async_config.groups_per_batch):
            await env_group_builders_queue.put(_Shutdown())
        logger.info("[dataloader_loop] Terminated")

    @trace.scope
    async def trajectory_group_worker_loop():
        """Generates trajectories for a single env builder"""
        while True:
            env_group_builder = await env_group_builders_queue.get()
            if isinstance(env_group_builder, _Shutdown):
                logger.info("[trajectory_group_worker_loop] Received shutdown signal")
                break

            # Save a reference to the sampling client step in case it changes
            # while we're running the rollout
            sampling_client_step_copy = sampling_client_step
            worker_metrics: dict[str, Any] = {}
            t_start = time.time()
            async with trace.scope_span("trajectory_group_worker"):
                trajectory_group = await do_group_rollout_and_filter_constant_reward(
                    sampling_client,
                    env_group_builder,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    do_remove_constant_reward_groups=config.remove_constant_reward_groups,
                    strategy=strategy,
                )
            worker_metrics["time/trajectory_group_worker_loop/total"] = time.time() - t_start
            # Ingest error info (safe: same event loop thread)
            if error_counter is not None:
                error_counter.ingest(trajectory_group)
            if trajectory_group is None:
                trajectory_groups_queue.put_nowait(None)
            else:
                trajectory_groups_queue.put_nowait(
                    WrappedTrajectoryGroup(
                        trajectory_group=trajectory_group,
                        env_group_builder=env_group_builder,
                        sampling_client_step=sampling_client_step_copy,
                        metrics=worker_metrics,
                    )
                )

        # When this is the last worker to exit, signal the training loop to shut down
        num_alive = await worker_alive_counter.decrement_and_get()
        if num_alive == 0:
            logger.info(
                "[trajectory_group_worker_loop] Last worker exited, shutting down training loop"
            )
            trajectory_groups_queue.put_nowait(_Shutdown())
        logger.info("[trajectory_group_worker_loop] Terminated")

    @trace.scope
    async def training_loop():
        """
        Waits for a sufficient number of valid trajectories to be accumulated and trains on them.
        Will discard trajectories that are too stale.
        """
        assert config.async_config is not None

        i_batch = start_batch
        wrapped_trajectory_groups = []
        while i_batch < end_batch:

            def filter_stale_trajectory_group(
                wrapped_trajectory_group: WrappedTrajectoryGroup | None,
            ) -> bool:
                """Returns False if the trajectory group is too stale or not valid"""
                if wrapped_trajectory_group is None:
                    return False

                # If the samples are too stale, requeue the data so that it will be used eventually.
                # Skip requeuing during shutdown to avoid deadlocking on a full bounded queue.
                assert config.async_config is not None
                if (
                    i_batch - wrapped_trajectory_group.sampling_client_step
                    > config.async_config.max_steps_off_policy
                ):
                    if dataloader_done_event.is_set():
                        logger.info(
                            f"[training_loop] Step {i_batch}: Samples are too stale, "
                            "discarding (dataloader done)"
                        )
                    else:
                        logger.info(
                            f"[training_loop] Step {i_batch}: Samples are too stale, requeuing"
                        )
                        asyncio.create_task(
                            env_group_builders_queue.put(
                                wrapped_trajectory_group.env_group_builder
                            ),
                            name="requeue_stale_sample_task",
                        )
                    return False
                return True

            metrics: dict[str, Any] = {
                "training_client/step": i_batch,
                "optim/lr": config.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
            }

            nonlocal sampling_client
            nonlocal sampling_client_step
            if config.stream_minibatch_config is not None:
                # Streaming minibatch: delegate queue consumption to the streaming function.
                # We need to check for shutdown before entering the streaming function,
                # since it will block on queue.get() internally.
                wrapped_trajectory_group = await trajectory_groups_queue.get()
                if isinstance(wrapped_trajectory_group, _Shutdown):
                    logger.info("[training_loop] Received shutdown signal")
                    break
                if wrapped_trajectory_group is None:
                    continue
                await trajectory_groups_queue.put(wrapped_trajectory_group)

                with trace.trace_iteration(step=i_batch) as window:
                    streaming_result = await do_train_step_streaming_and_get_sampling_client(
                        config,
                        i_batch,
                        trajectory_groups_queue,
                        training_client,
                        checkpoint_mgr,
                        kl_reference_client,
                        weight_syncer,
                        tokenizer,
                        filter_stale_trajectory_group,
                    )
                if streaming_result is None:
                    logger.info("[training_loop] Received shutdown signal from streaming")
                    break
                (
                    sampling_client,
                    train_step_metrics,
                    full_batch_wrapped_trajectory_groups,
                ) = streaming_result
                iter_dir = iteration_dir(config.log_path, i_batch)
                _maybe_export_rollout_summary_jsonl(
                    config=config,
                    base_name="train",
                    split="train",
                    iteration=i_batch,
                    groups_P=[
                        RolloutSummaryGroup(
                            trajectory_group=group.trajectory_group,
                            tags=group.env_group_builder.logging_tags(),
                            sampling_client_step=group.sampling_client_step,
                        )
                        for group in full_batch_wrapped_trajectory_groups
                    ],
                    store=ml_logger.store,
                )
            else:
                wrapped_trajectory_group = await trajectory_groups_queue.get()
                if isinstance(wrapped_trajectory_group, _Shutdown):
                    logger.info("[training_loop] Received shutdown signal")
                    break
                if wrapped_trajectory_group is None:
                    continue

                if not filter_stale_trajectory_group(wrapped_trajectory_group):
                    continue

                # Dynamic sampling: Wait for enough trajectories to accumulate to
                # ensure all batch sizes are the same size. This avoids needing to adjust
                # the learning rate for different batch sizes.
                wrapped_trajectory_groups.append(wrapped_trajectory_group)
                if len(wrapped_trajectory_groups) < config.async_config.groups_per_batch:
                    continue
                logger.info(
                    f"[training_loop] Step {i_batch}: Will train on batch, num groups: {len(wrapped_trajectory_groups)}"
                )

                # Compute sampling client metrics, as samples may have been generated with
                # different sampler versions
                metrics.update(compute_sampling_client_metrics(wrapped_trajectory_groups))

                with trace.trace_iteration(step=i_batch) as window:
                    # TODO: For proper checkpointing, we also need to save dataloader state and
                    # all queued trajectory groups that haven't been trained on yet
                    (
                        sampling_client,
                        train_step_metrics,
                    ) = await do_train_step_and_get_sampling_client(
                        config,
                        i_batch,
                        training_client,
                        checkpoint_mgr,
                        kl_reference_client,
                        tokenizer,
                        [g.env_group_builder for g in wrapped_trajectory_groups],
                        [g.trajectory_group for g in wrapped_trajectory_groups],
                        weight_syncer=weight_syncer,
                    )
                iter_dir = iteration_dir(config.log_path, i_batch)
                _maybe_export_rollout_summary_jsonl(
                    config=config,
                    base_name="train",
                    split="train",
                    iteration=i_batch,
                    groups_P=[
                        RolloutSummaryGroup(
                            trajectory_group=group.trajectory_group,
                            tags=group.env_group_builder.logging_tags(),
                            sampling_client_step=group.sampling_client_step,
                        )
                        for group in wrapped_trajectory_groups
                    ],
                    store=ml_logger.store,
                )
            sampling_client_step = i_batch + 1
            sampling_client_updated_event.set()

            # Rolling checkpoint (fire-and-forget, overlaps with next iteration)
            if checkpoint_mgr is not None:
                await checkpoint_mgr.maybe_save_rolling_async(
                    step=i_batch + 1, loop_state={"batch": i_batch + 1}
                )

            # Log metrics
            metrics.update(train_step_metrics)
            if error_counter is not None:
                metrics.update(error_counter.get_metrics())
            metrics.update(window.get_timing_metrics())
            window.save_timing(i_batch, store=ml_logger.store)
            if config.span_chart_every > 0 and i_batch % config.span_chart_every == 0:
                iter_dir = iteration_dir(config.log_path, i_batch)
                if iter_dir is not None:
                    iter_dir.mkdir(parents=True, exist_ok=True)
                    trace.save_gantt_chart_html(window, i_batch, iter_dir / "timing_gantt.html")
            ml_logger.log_metrics(metrics, step=i_batch)
            i_batch += 1
            wrapped_trajectory_groups = []

        # Signal evaluation loop to shut down
        evaluation_loop_should_shutdown_event.set()
        sampling_client_updated_event.set()
        logger.info("[training_loop] Terminated")

    @trace.scope
    async def evaluation_loop():
        """Runs evals periodically, matching the sync training eval pipeline."""
        if len(evaluators) == 0 or config.eval_every == 0:
            return

        while not evaluation_loop_should_shutdown_event.is_set():
            await sampling_client_updated_event.wait()
            sampling_client_updated_event.clear()

            # Save a reference to the original values in case it changes
            # while we're running the evals
            sampling_client_eval_step = sampling_client_step
            sampling_client_eval = sampling_client
            if config.eval_every > 0 and sampling_client_eval_step % config.eval_every == 0:
                metrics: dict[str, Any] = {}
                with trace.trace_iteration(step=sampling_client_eval_step) as window:
                    eval_metrics = await run_evaluations_parallel(
                        evaluators,
                        sampling_client_eval,
                        config,
                        sampling_client_eval_step,
                        store=ml_logger.store,
                    )
                    metrics.update(eval_metrics)
                metrics.update(window.get_timing_metrics())
                ml_logger.log_metrics(metrics, step=sampling_client_eval_step)
        logger.info("[evaluation_loop] Terminated")

    await asyncio.gather(
        asyncio.create_task(dataloader_loop(), name="dataloader_loop"),
        *[
            asyncio.create_task(
                trajectory_group_worker_loop(), name=f"trajectory_group_worker_loop_{i}"
            )
            for i in range(config.async_config.groups_per_batch)
        ],
        asyncio.create_task(training_loop(), name="training_loop"),
        asyncio.create_task(evaluation_loop(), name="evaluation_loop"),
    )


@trace.scope
async def save_checkpoint_and_get_sampling_client(
    training_client: FiretitanTrainingClient,
    checkpoint_mgr: checkpoint_utils.CheckpointManager,
    weight_syncer: WeightSyncer,
    i_batch: int,
    start_batch: int = 0,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """Save a checkpoint (if due) and return a fresh sampling client.

    When ``i_batch`` falls on the periodic checkpoint cadence, a full
    checkpoint (weights + optimizer state) is persisted via
    *checkpoint_mgr*. Otherwise a lightweight sampler-only snapshot is
    created so that subsequent rollouts use the latest weights.

    Args:
        training_client (tinker.TrainingClient): Client connected to the
            Tinker training service.
        checkpoint_mgr (checkpoint_utils.CheckpointManager): Manager that
            handles periodic checkpoint saves.
        i_batch (int): Current training iteration index.
        start_batch (int): First iteration index of this run, used to avoid
            checkpointing on the very first step. Defaults to 0.

    Returns:
        tuple[tinker.SamplingClient, dict[str, Any]]: A sampling client
        loaded with the latest weights, and a (possibly empty) metrics dict.
    """
    metrics: dict[str, Any] = {}
    if i_batch > start_batch and checkpoint_mgr.should_save_periodic(i_batch):
        path_dict = await checkpoint_mgr.save_periodic_async(
            step=i_batch, loop_state={"batch": i_batch}
        )
    else:
        async with trace.scope_span("save_checkpoint"):
            await training_client.save_weights_and_get_sampling_client_async(), metrics

    checkpoint_type = "base" # if config.lora_rank == 0 else "delta"
    success = weight_syncer.hotload(path_dict["sampler_path"], checkpoint_type=checkpoint_type)
    if not success:
        raise ValueError(f"Failed to save and load checkpoint {path_dict["sampler_path"]}")
    return weight_syncer.get_deployment_sampler(), metrics

@trace.scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    kl_reference_client: FiretitanTrainingClient | None,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Convert trajectory groups into training data with computed advantages.

    Computes per-trajectory metrics, prints sample groups, assembles
    ``tinker.Datum`` objects with advantages, and optionally incorporates a
    KL penalty against a reference policy.

    Args:
        env_group_builders_P (Sequence[EnvGroupBuilder]): Builders that
            produced each trajectory group (used for logging tags).
        trajectory_groups_P (list[TrajectoryGroup]): Collected trajectory
            groups to convert.
        tokenizer (Tokenizer): Tokenizer for decoding tokens during logging.
        kl_reference_client (tinker.SamplingClient | None): Sampling client
            for the KL reference model, or None if KL penalty is disabled.
        kl_penalty_coef (float): Coefficient for the KL penalty term. Set
            to 0 to disable.
        kl_discount_factor (float): Position-based discount factor for KL
            penalty terms.

    Returns:
        tuple[list[tinker.Datum], dict[str, Any]]: A list of training datums
        and a dict of trajectory and KL penalty metrics.
    """

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Print up to two trajectory groups
    for traj_group in trajectory_groups_P[:2]:
        print_group(traj_group, tokenizer)

    # Assemble training data
    async with trace.scope_span("assemble_training_data"):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0 and kl_reference_client is not None:
        async with trace.scope_span("kl_vs_base"):
            kl_penalty_metrics = await incorporate_kl_penalty(
                data_D,
                kl_reference_client,
                kl_penalty_coef,
                kl_discount_factor,
            )
        metrics.update(kl_penalty_metrics)

    return data_D, metrics


@trace.scope
async def compute_full_batch_metrics_and_get_sampling_client(
    training_client: FiretitanTrainingClient,
    checkpoint_mgr: checkpoint_utils.CheckpointManager,
    weight_syncer: WeightSyncer,
    i_batch: int,
    data_D: list[tinker.Datum],
    training_logprobs_D: list[torch.Tensor],
    do_compute_post_kl: bool,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """Compute end-of-iteration metrics and return a fresh sampling client.

    Calculates KL divergence between the sampling and training log-probs,
    saves a checkpoint (if due), and optionally computes post-update KL
    metrics using the newly checkpointed weights.

    Args:
        training_client (tinker.TrainingClient): Client connected to the
            Tinker training service.
        checkpoint_mgr (checkpoint_utils.CheckpointManager): Manager that
            handles periodic checkpoint saves.
        i_batch (int): Current training iteration index (used for checkpoint
            naming).
        data_D (list[tinker.Datum]): Training data from the current iteration.
        training_logprobs_D (list[torch.Tensor]): Per-datum log-probabilities
            returned by the training forward pass.
        do_compute_post_kl (bool): Whether to compute post-update KL metrics
            against the new sampling client (adds an extra sampling call).

    Returns:
        tuple[tinker.SamplingClient, dict[str, Any]]: A sampling client
        loaded with post-update weights, and a dict of KL / checkpoint
        metrics.
    """
    metrics = {}
    print("training_logprobs_D: ", len(training_logprobs_D))
    # Compute KL metrics
    async with trace.scope_span("compute_kl_sample_train"):
        kl_sample_train_metrics = compute_kl_sample_train(data_D, training_logprobs_D)
        metrics.update(kl_sample_train_metrics)

    # Get a sampling client using the new weights
    sampling_client, checkpoint_metrics = await save_checkpoint_and_get_sampling_client(
        training_client, checkpoint_mgr, weight_syncer, i_batch,
    )
    metrics.update(checkpoint_metrics)

    # Compute post-KL metrics if configured
    if do_compute_post_kl:
        async with trace.scope_span("compute_post_kl"):
            post_kl_metrics = await compute_post_kl(data_D, sampling_client)
            metrics.update(post_kl_metrics)

    return sampling_client, metrics


@trace.scope
async def do_train_step_streaming_and_get_sampling_client(
    config: Config,
    i_batch: int,
    trajectory_groups_queue: asyncio.Queue[WrappedTrajectoryGroup | _Shutdown | None],
    training_client: FiretitanTrainingClient,
    checkpoint_mgr: checkpoint_utils.CheckpointManager,
    kl_reference_client: FiretitanTrainingClient | None,
    tokenizer: Tokenizer,
    weight_syncer: WeightSyncer,
    trajectory_group_filter: Callable[[WrappedTrajectoryGroup | None], bool] = lambda _: True,
) -> tuple[tinker.SamplingClient, dict[str, Any], list[WrappedTrajectoryGroup]] | None:
    """Consume trajectory groups from a queue and train as minibatches become ready.

    As soon as enough trajectories for a minibatch have accumulated, a
    ``forward_backward`` call is enqueued. After all minibatches in a substep
    are submitted, one ``optim_step`` is issued. This overlaps sampling I/O
    with GPU training.

    Args:
        config (Config): RL training configuration. Must have
            ``stream_minibatch_config`` set.
        i_batch (int): Current training iteration index.
        trajectory_groups_queue (asyncio.Queue): Queue yielding
            ``WrappedTrajectoryGroup``, ``None`` (filtered/failed group), or
            ``_Shutdown`` sentinel.
        training_client (tinker.TrainingClient): Client connected to the
            Tinker training service.
        kl_reference_client (tinker.SamplingClient | None): Sampling client
            for the KL reference model, or None if KL penalty is disabled.
        tokenizer (Tokenizer): Tokenizer for decoding tokens during logging.
        trajectory_group_filter (Callable): Predicate applied to each
            dequeued group. Groups for which the filter returns False are
            skipped. Defaults to accepting all groups.

    Returns:
        tuple[tinker.SamplingClient, dict[str, Any], list[WrappedTrajectoryGroup]] | None:
        A 3-tuple of (sampling client with updated weights, aggregated
        metrics, list of all trajectory groups used for training), or None
        if a ``_Shutdown`` sentinel was received.
    """
    assert config.stream_minibatch_config is not None
    assert config.stream_minibatch_config.groups_per_batch % config.num_substeps == 0, (
        f"{config.stream_minibatch_config.groups_per_batch=} must be divisible by {config.num_substeps=}"
    )
    # Number of groups across all minibatches in each optimizer substep
    groups_per_substep = config.stream_minibatch_config.groups_per_batch // config.num_substeps
    assert groups_per_substep % config.stream_minibatch_config.num_minibatches == 0, (
        f"{groups_per_substep} must be divisible by {config.stream_minibatch_config.num_minibatches=}"
    )
    # Number of groups per minibatch in each optimizer substep
    groups_per_minibatch = groups_per_substep // config.stream_minibatch_config.num_minibatches

    trace.update_scope_context({"step": i_batch})

    metrics = {}

    # Run multiple optimizer substeps per training iteration
    all_data_D = []
    all_training_logprobs_D = []
    all_wrapped_trajectory_groups = []
    for i_substep in range(config.num_substeps):
        # Run multiple minibatches per substep
        # Once we have enough trajectories for a minibatch, train on them
        wrapped_trajectory_groups = []
        forward_backward_futures: list[tinker.APIFuture[tinker.ForwardBackwardOutput]] = []
        forward_futures: list[tinker.APIFuture[tinker.ForwardBackwardOutput]] = []
        i_minibatch = 0
        while i_minibatch < config.stream_minibatch_config.num_minibatches:
            wrapped_trajectory_group = await trajectory_groups_queue.get()
            if isinstance(wrapped_trajectory_group, _Shutdown):
                logger.info("[do_train_step_streaming] Received shutdown signal")
                return None
            if not trajectory_group_filter(wrapped_trajectory_group):
                continue
            wrapped_trajectory_groups.append(wrapped_trajectory_group)

            if len(wrapped_trajectory_groups) < groups_per_minibatch:
                continue
            logger.info(
                f"[stream_minibatch] Step {i_batch}, Substep {i_substep}/{config.num_substeps}, Minibatch {i_minibatch}/{config.stream_minibatch_config.num_minibatches}: Will train on minibatch, num groups: {len(wrapped_trajectory_groups)}"
            )

            # Note: we may have removed trajectory groups that have the same reward.
            # To have the same results as the sync implementation, we will
            # remove these and train on a smaller batch.
            wrapped_trajectory_groups = [g for g in wrapped_trajectory_groups if g is not None]
            if len(wrapped_trajectory_groups) == 0:
                i_minibatch += 1
                continue

            data_D, prepare_minibatch_metrics = await prepare_minibatch(
                [g.env_group_builder for g in wrapped_trajectory_groups],
                [g.trajectory_group for g in wrapped_trajectory_groups],
                tokenizer,
                kl_reference_client,
                kl_penalty_coef=config.kl_penalty_coef,
                kl_discount_factor=config.kl_discount_factor,
            )
            metrics.update(prepare_minibatch_metrics)

            # Enqueue forward-backward (we'll await results after all minibatches are enqueued)
            async with trace.scope_span(
                f"train/fwd_bwd_substep_{i_substep}_mb_{i_minibatch}_enqueue"
            ):
                forward_backward_futures.append(
                    await training_client.forward_backward_async(
                        [_remove_mask(d) for d in data_D],
                        loss_fn=config.loss_fn,
                        loss_fn_config=config.loss_fn_config,
                    )
                )
            async with trace.scope_span(
                f"train/forward_substep_{i_substep}_mb_{i_minibatch}_enqueue"
            ):
                forward_futures.append(
                    await training_client.forward_async(
                        [_cross_entropy_forward_datum(d) for d in data_D],
                        loss_fn="cross_entropy",
                    )
                )
            all_data_D.extend(data_D)
            all_wrapped_trajectory_groups.extend(wrapped_trajectory_groups)
            i_minibatch += 1
            wrapped_trajectory_groups = []

        # Enqueue optim_step before awaiting results (so they land on same clock cycle)
        adam_params = tinker.AdamParams(
            learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
        )
        async with trace.scope_span(f"train/optim_substep_{i_substep}_enqueue"):
            optim_future = await training_client.optim_step_async(adam_params)

        # Now consume all forward-backward and forward results
        for i_mb, fwd_bwd_future in enumerate(forward_backward_futures):
            async with trace.scope_span(f"train/fwd_bwd_substep_{i_substep}_mb_{i_mb}_consume"):
                await fwd_bwd_future.result_async()
            async with trace.scope_span(f"train/forward_substep_{i_substep}_mb_{i_mb}_consume"):
                forward_result = await forward_futures[i_mb].result_async()
                all_training_logprobs_D.extend(_training_logprobs_from_forward(forward_result))

        async with trace.scope_span(f"train/optim_substep_{i_substep}_consume"):
            optim_result = await optim_future.result_async()

        if optim_result.metrics:
            metrics.update(optim_result.metrics)

    # Aggregate metrics across the entire batch
    metrics.update(compute_sampling_client_metrics(all_wrapped_trajectory_groups))
    metrics.update(
        compute_trajectory_metrics(
            [g.trajectory_group for g in all_wrapped_trajectory_groups],
            [g.env_group_builder.logging_tags() for g in all_wrapped_trajectory_groups],
        )
    )
    (
        sampling_client,
        full_batch_metrics,
    ) = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        checkpoint_mgr,
        weight_syncer,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        all_data_D,
        all_training_logprobs_D,
        config.compute_post_kl,
    )
    metrics.update(full_batch_metrics)
    return sampling_client, metrics, all_wrapped_trajectory_groups


@trace.scope
async def do_train_step_and_get_sampling_client(
    config: Config,
    i_batch: int,
    training_client: FiretitanTrainingClient,
    checkpoint_mgr: checkpoint_utils.CheckpointManager,
    kl_reference_client: FiretitanTrainingClient | None,
    weight_syncer: WeightSyncer,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    """Prepare a minibatch, run one training step, and return updated weights.

    This is the standard (non-streaming) single-iteration training step. It
    prepares data from trajectory groups, calls :func:`train_step`, computes
    full-batch metrics, and checkpoints.

    Args:
        config (Config): RL training configuration.
        i_batch (int): Current training iteration index.
        training_client (tinker.TrainingClient): Client connected to the
            Tinker training service.
        kl_reference_client (tinker.SamplingClient | None): Sampling client
            for the KL reference model, or None if KL penalty is disabled.
        tokenizer (Tokenizer): Tokenizer for decoding tokens during logging.
        env_group_builders_P (Sequence[EnvGroupBuilder]): Builders that
            produced each trajectory group.
        trajectory_groups_P (list[TrajectoryGroup]): Collected trajectory
            groups for this iteration.

    Returns:
        tuple[tinker.SamplingClient, dict[str, Any]]: A sampling client
        loaded with post-update weights, and a dict of all metrics from
        minibatch preparation, training, and checkpointing.
    """
    trace.update_scope_context({"step": i_batch})

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        kl_reference_client,
        kl_penalty_coef=config.kl_penalty_coef,
        kl_discount_factor=config.kl_discount_factor,
    )
    metrics.update(prepare_minibatch_metrics)

    training_logprobs_D = await train_step(
        data_D=data_D,
        training_client=training_client,
        learning_rate=config.learning_rate,
        num_substeps=config.num_substeps,
        loss_fn=config.loss_fn,
        loss_fn_config=config.loss_fn_config,
        metrics=metrics,
    )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        checkpoint_mgr,
        weight_syncer,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        data_D,
        training_logprobs_D,
        config.compute_post_kl,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@trace.scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    config: Config,
    training_client: FiretitanTrainingClient,
    kl_reference_client: FiretitanTrainingClient | None,
    weight_syncer: WeightSyncer,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    error_counter: RolloutErrorCounter | None = None,
    strategy: RolloutStrategy | None = None,
    checkpoint_mgr: checkpoint_utils.CheckpointManager | None = None,
):
    """Implement fully synchronous on-policy training.

    Each iteration samples all trajectory groups in parallel (via
    :func:`gather_with_progress`), trains on the full batch, checkpoints,
    and then runs evaluations. This is the simplest execution mode and
    guarantees that all training data is fully on-policy.

    Args:
        start_batch (int): First training iteration index (inclusive).
        end_batch (int): Last training iteration index (exclusive).
        num_batches (int): Total number of batches in the dataset, used for
            progress fraction calculation.
        config (Config): RL training configuration.
        training_client (tinker.TrainingClient): Client connected to the
            Tinker training service.
        kl_reference_client (tinker.SamplingClient | None): Sampling client
            for the KL reference model, or None if KL penalty is disabled.
        evaluators (list[SamplingClientEvaluator]): Evaluators to run
            periodically during training.
        dataset (RLDataset): The RL dataset providing batches of
            ``EnvGroupBuilder`` instances.
        ml_logger (ml_log.Logger): Logger for metrics and W&B integration.
        tokenizer (Tokenizer): Tokenizer for decoding rollout tokens.
        error_counter (RolloutErrorCounter | None): Optional counter for
            tracking rollout errors. Defaults to None.
        strategy (RolloutStrategy | None): Rollout error handling strategy.
            Defaults to None.
    """
    # Initial sampling client
    assert checkpoint_mgr is not None
    if weight_syncer is not None and weight_syncer.base_identity is not None:
        # Weights were already hotloaded during setup (e.g. Fireworks path),
        # skip redundant save+hotload which can crash the deployment.
        sampling_client = weight_syncer.get_deployment_sampler()
    else:
        sampling_client, _ = await save_checkpoint_and_get_sampling_client(
            training_client,
            checkpoint_mgr,
            weight_syncer,
            start_batch,
        )

    for i_batch in range(start_batch, end_batch):
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }

        with trace.trace_iteration(step=i_batch) as window:
            # Run evaluations
            if config.eval_every > 0 and i_batch % config.eval_every == 0:
                eval_metrics = await run_evaluations_parallel(
                    evaluators, sampling_client, config, i_batch, store=ml_logger.store
                )
                metrics.update(eval_metrics)

            # Get batch and sample trajectories
            env_group_builders_P = dataset.get_batch(i_batch)

            # Initialize logtree trace for this iteration if logging is enabled
            iter_dir = iteration_dir(config.log_path, i_batch)
            async with trace.scope_span("sampling"):
                with _get_logtree_scope(
                    output_dir=iter_dir,
                    num_groups_to_log=config.num_groups_to_log,
                    f_name="train",
                    scope_name=f"RL Iteration {i_batch}",
                    iteration=i_batch,
                    store=ml_logger.store,
                ):
                    # Note: do_remove_constant_reward_groups=False here because we remove
                    # constant reward groups after all rollouts are collected (below)
                    results_P = await gather_with_progress(
                        (
                            do_group_rollout_and_filter_constant_reward(
                                sampling_client,
                                builder,
                                max_tokens=config.max_tokens,
                                temperature=config.temperature,
                                do_remove_constant_reward_groups=False,
                                enable_logging=i < config.num_groups_to_log,
                                strategy=strategy,
                            )
                            for i, builder in enumerate(env_group_builders_P)
                        ),
                        desc=f"Sampling batch {i_batch}",
                    )

            # Ingest error info from results
            if error_counter is not None:
                for result in results_P:
                    error_counter.ingest(result)

            # Filter out None results (from errored or fully-failed groups)
            successful = [
                (builder, tg)
                for builder, tg in safezip(env_group_builders_P, results_P)
                if tg is not None
            ]
            batch_skipped = not successful
            if batch_skipped:
                logger.warning(f"Batch {i_batch}: all groups failed or filtered, skipping batch")
            else:
                env_group_builders_P = [s[0] for s in successful]
                trajectory_groups_P: list[TrajectoryGroup] = [s[1] for s in successful]

                _maybe_export_rollout_summary_jsonl(
                    config=config,
                    base_name="train",
                    split="train",
                    iteration=i_batch,
                    groups_P=[
                        RolloutSummaryGroup(
                            trajectory_group=trajectory_group,
                            tags=env_group_builder.logging_tags(),
                            sampling_client_step=i_batch,
                        )
                        for env_group_builder, trajectory_group in safezip(
                            env_group_builders_P, trajectory_groups_P
                        )
                    ],
                    store=ml_logger.store,
                )

                if config.remove_constant_reward_groups:
                    trajectory_groups_P = remove_constant_reward_groups(trajectory_groups_P)

                # Train step
                sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
                    config,
                    i_batch,
                    training_client,
                    checkpoint_mgr,
                    kl_reference_client,
                    weight_syncer,
                    tokenizer,
                    env_group_builders_P,
                    trajectory_groups_P,
                )

                metrics.update(train_step_metrics)

                # Rolling checkpoint (fire-and-forget, overlaps with next iteration)
                if checkpoint_mgr is not None:
                    await checkpoint_mgr.maybe_save_rolling_async(
                        step=i_batch + 1, loop_state={"batch": i_batch + 1}
                    )

        metrics.update(window.get_timing_metrics())
        if error_counter is not None:
            metrics.update(error_counter.get_metrics())
        window.save_timing(i_batch, store=ml_logger.store)
        if (
            config.span_chart_every > 0
            and i_batch % config.span_chart_every == 0
            and iter_dir is not None
        ):
            iter_dir.mkdir(parents=True, exist_ok=True)
            trace.save_gantt_chart_html(window, i_batch, iter_dir / "timing_gantt.html")
        ml_logger.log_metrics(metrics, step=i_batch)


@trace.scope
async def main(
    config: Config,
    rollout_executor: Executor | None = None,
):
    """Main training loop for MDP RL.

    Orchestrates the full RL training lifecycle: initializes the Tinker
    service and training clients, resumes from checkpoints if available,
    builds the dataset and evaluators, dispatches to the appropriate
    execution mode (sync, async, or streaming minibatch), and saves a
    final checkpoint upon completion.

    Args:
        config (Config): Training configuration.
        rollout_executor (Executor | None): Optional ``concurrent.futures.Executor``
            for offloading group rollouts to separate processes or remote
            workers. Pass ``ProcessPoolExecutor(max_workers=N)`` for
            multi-process execution, or any custom ``Executor`` (Ray,
            cluster dispatchers, etc.). Default ``None`` runs rollouts as
            asyncio coroutines in-process.

    Raises:
        ConfigurationError: If ``kl_penalty_coef > 0`` but
            ``kl_reference_config`` is not set.

    Example::

        import asyncio
        from tinker_cookbook.rl.train import Config, main

        config = Config(
            learning_rate=1e-5,
            dataset_builder=my_dataset_builder,
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            max_tokens=2048,
            log_path="./logs/my_rl_run",
        )
        asyncio.run(main(config=config))
    """

    if rollout_executor is not None:
        set_rollout_executor(rollout_executor)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=config.wandb_name,
    )
    store = ml_logger.store
    if config.enable_trace:
        # Get and rename the current (main) task
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = str(Path(config.log_path) / "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace.trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_batch = resume_info.batch
    else:
        start_batch = 0

    service_client = FiretitanServiceClient(
        base_url=config.base_url,
        api_key=os.environ["FIREWORKS_API_KEY"],
    )
    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)
    model_info.warn_if_renderer_not_recommended(config.model_name, config.renderer_name)

    training_client = service_client.create_training_client(
        base_model=config.model_name,
        lora_rank=config.lora_rank,
    )
    if resume_info:
        # Resuming interrupted training - load optimizer state for proper continuation
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, resume_info.state_path, config.renderer_name
        )
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path, user_metadata=user_metadata
            )
        )
        logger.info(f"Resumed training from {resume_info.state_path}")
    elif config.load_checkpoint_path:
        # Starting fresh from a checkpoint - load weights only (fresh optimizer)
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, config.load_checkpoint_path, config.renderer_name
        )
        training_client = await service_client.create_training_client_from_state_async(
            config.load_checkpoint_path, user_metadata=user_metadata
        )
        logger.info(f"Loaded weights from {config.load_checkpoint_path}")
        training_client.load_state(config.load_checkpoint_path)
    # else:
    #     training_client = await service_client.create_lora_training_client_async(
    #         config.model_name, rank=config.lora_rank, user_metadata=user_metadata
    #     )

    deploy_mgr = DeploymentManager(api_key=os.environ["FIREWORKS_API_KEY"])
    weight_syncer = WeightSyncer(
        policy_client=training_client,
        deploy_mgr=deploy_mgr,
        deployment_id=config.fireworks_deployment_id,
        base_model=config.fireworks_base_model_name,
        hotload_timeout=config.fireworks_hot_load_timeout,
    )
    if config.fireworks_deployment_id:
        name = f"resume-{start_batch}-base" if start_batch > 0 else "step-0-base"
        weight_syncer.save_and_hotload(name, checkpoint_type="base")

    # Load the local tokenizer by public model name. Some Fireworks-hosted model
    # metadata points at internal paths that are not valid on the client machine.
    tokenizer = get_tokenizer(config.model_name)

    # Create dataset from thunk
    dataset, maybe_test_dataset = await config.dataset_builder()
    # Build rollout strategy and error counter from config
    strategy = rollout_strategy_from_config(config.rollout_error_tolerance)
    error_counter = RolloutErrorCounter() if strategy.catches_group_errors else None

    evaluators = [evaluator() for evaluator in config.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(
            RLTestSetEvaluator(
                maybe_test_dataset,
                max_tokens=config.max_tokens,
                strategy=strategy,
            )
        )

    num_batches = len(dataset)
    end_batch = min(config.max_steps, num_batches) if config.max_steps is not None else num_batches
    logger.info(f"Will train on {end_batch} batches")

    # Create KL reference client once if KL penalty is enabled
    if config.kl_penalty_coef > 0:
        if config.kl_reference_config is None:
            raise ConfigurationError(
                "kl_reference_config must be specified when kl_penalty_coef > 0"
            )
        kl_reference_client = service_client.create_training_client(
            base_model=config.kl_reference_config.base_model,
            lora_rank=config.lora_rank,
        )
        if config.kl_reference_config.load_checkpoint_path:
            kl_reference_client.load_state(config.kl_reference_config.load_checkpoint_path)
    else:
        kl_reference_client = None

    checkpoint_mgr = checkpoint_utils.CheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=config.log_path,
        save_every=config.save_every,
        ttl_seconds=config.ttl_seconds,
        rolling_save_every=config.rolling_save_every,
        rolling_ttl_seconds=config.rolling_ttl_seconds,
        store=store,
    )

    # Training loop
    if config.async_config is not None:
        training_func = do_async_training
    elif config.stream_minibatch_config is not None:
        training_func = do_sync_training_with_stream_minibatch
    else:
        training_func = do_sync_training
    await training_func(
        start_batch=start_batch,
        end_batch=end_batch,
        num_batches=end_batch,
        config=config,
        training_client=training_client,
        kl_reference_client=kl_reference_client,
        weight_syncer=weight_syncer,
        evaluators=evaluators,
        dataset=dataset,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
        error_counter=error_counter,
        strategy=strategy,
        checkpoint_mgr=checkpoint_mgr,
    )

    # Save final checkpoint
    if start_batch < end_batch:
        await checkpoint_mgr.save_final_async(loop_state={"batch": end_batch})
    else:
        logger.info("Training was already complete; nothing to do")
        await checkpoint_mgr.finalize_async()

    # Cleanup
    if rollout_executor is not None:
        rollout_executor.shutdown(wait=True)
        set_rollout_executor(None)
    ml_logger.close()
    logger.info("Training completed successfully")