"""
Implements RL on general MDPs
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import logging
import re
import time
from collections.abc import Callable, Coroutine, Iterable, Iterator, Sequence
from concurrent.futures import Executor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from tinker_cookbook.stores.training_store import TrainingRunStore

import chz
import numpy as np
import tinker
import torch
from tinker.types import LossFnType
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.dataset_utils import EpochCycledRLDataset
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.metrics import (
    compute_kl_sample_train,
    compute_post_kl,
    compute_sampling_client_metrics,
    incorporate_kl_penalty,
)
from tinker_cookbook.rl.rollout_limits import TerminationRewardPolicy
from tinker_cookbook.rl.rollout_logging import (
    RolloutSummaryExportConfig,
    RolloutSummaryGroup,
)
from tinker_cookbook.rl.rollout_presets import (
    default_rollout_config_for_model,
    default_rollout_strategy_for_model,
)
from tinker_cookbook.rl.rollout_strategy import (
    RolloutStrategy,
    rollout_strategy_from_config,
)
from tinker_cookbook.rl.rollouts import (
    RolloutErrorCounter,
    do_group_rollout,  # noqa: F401 — re-exported for backwards compatibility; to
    # override rollout behavior, rebind tinker_cookbook.rl.rollouts.do_group_rollout
    # (rebinding this re-exported name does not affect the rollout pipeline)
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
from tinker_cookbook.utils.cost_estimation import CostTracker, TokenUsage, fetch_model_pricing
from tinker_cookbook.utils.git_rev import recipe_user_metadata
from tinker_cookbook.utils.misc_utils import all_same, iteration_dir, safezip, split_list
from tinker_cookbook.utils.session_naming import resolve_session_display_name

logger = logging.getLogger(__name__)

T = TypeVar("T")


@chz.chz
class KLReferenceConfig:
    """Configuration for the KL penalty reference model.

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

    async def track(coro: Coroutine[Any, Any, T]) -> T:
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
    from tinker_cookbook.rl.rollout_logging import serialize_rollout_summaries_from_groups

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
    # replace() keeps the datum's other fields, such as provenance spans.
    return dataclasses.replace(
        datum, loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"}
    )


def _check_provenance(data_D: list[tinker.Datum], loss_fn: LossFnType) -> None:
    """Raise if ``loss_fn`` needs provenance that some datum lacks.

    Called on a step's data before any of it is sent. With streaming
    minibatches it runs per minibatch, so a bad datum in a later minibatch
    raises after earlier ones were sent; the run fails either way.
    """
    if loss_fn != "stable_reinforce":
        return
    for datum in data_D:
        if datum.loss_fn_input_spans is None:
            raise ValueError(
                "loss_fn='stable_reinforce' needs provenance on every datum. A datum has none "
                "when one of its actions lacks a Tinker sequence_id "
                "(TokensWithLogprobs.sequence_id, which a custom TokenCompleter must set) or "
                "when none of its targets is a sampled token. Fix the data or train with a "
                "different loss_fn."
            )


def _training_logprobs_from_fwd_bwd(
    fwd_bwd_result: tinker.ForwardBackwardOutput,
) -> list[torch.Tensor]:
    return [output["logprobs"].to_torch() for output in fwd_bwd_result.loss_fn_outputs]


@trace.scope
async def train_step(
    data_D: list[tinker.Datum],
    training_client: tinker.TrainingClient,
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
    _check_provenance(data_D, loss_fn)
    batches = split_list(data_D, min(num_substeps, len(data_D)))
    if not batches:
        return []

    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    training_logprobs_D: list[torch.Tensor] = []
    optim_result: tinker.OptimStepResponse | None = None

    # Enqueue first batch
    fwd_bwd_future = await training_client.forward_backward_async(
        [_remove_mask(d) for d in batches[0]], loss_fn=loss_fn, loss_fn_config=loss_fn_config
    )
    optim_future = await training_client.optim_step_async(adam_params)

    for i in range(len(batches)):
        # Enqueue next batch before consuming current results (to stay on same clock cycle)
        if i + 1 < len(batches):
            next_fwd_bwd_future = await training_client.forward_backward_async(
                [_remove_mask(d) for d in batches[i + 1]],
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            )
            next_optim_future = await training_client.optim_step_async(adam_params)
        else:
            next_fwd_bwd_future = None
            next_optim_future = None
        # Consume current results
        fwd_bwd_result = await fwd_bwd_future.result_async()
        training_logprobs_D.extend(_training_logprobs_from_fwd_bwd(fwd_bwd_result))
        optim_result = await optim_future.result_async()
        # Move to next iteration
        if next_fwd_bwd_future is not None and next_optim_future is not None:
            fwd_bwd_future = next_fwd_bwd_future
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

    In async mode, sampling and training run concurrently, so training data
    can come from older weights ("stale" or off-policy data). At most
    ``groups_per_batch * (max_steps_off_policy + 1)`` trajectory groups may be
    started but not yet trained. Completed groups are never discarded for
    staleness: each batch takes the groups sampled from the oldest weights
    first, so a slow rollout is trained when it finishes, even if that makes
    it more than ``max_steps_off_policy`` steps stale. With
    ``max_steps_off_policy=0`` all data is on-policy. With streaming
    minibatches, the cap uses ``StreamMinibatchConfig.groups_per_batch`` and
    groups are trained in the order they finish.

    Training steps don't map one-to-one to dataset batches: each step trains
    ``groups_per_batch`` groups that survive filtering (such as
    ``remove_constant_reward_groups``) and rollout errors, wherever they came
    from. Training ends after ``max_steps`` steps or when the dataset runs out,
    whichever comes first; in the second case the remaining groups are trained
    as a final, smaller step and a warning reports how many steps ran.

    Async runs train with ``loss_fn``, not ``Config.loss_fn``. Its default,
    ``stable_reinforce``, stays stable on off-policy data. With another loss,
    prefer a clipped one such as ``ppo``: ``importance_sampling`` can become
    unstable on stale data.

    A rollout that never finishes keeps its slot: training stalls once more
    than ``groups_per_batch * max_steps_off_policy`` rollouts hang (with
    ``max_steps_off_policy=0``, as in sync training, a single one), and the
    run can't finish while a rollout whose group is still needed hangs. For
    environments that can hang, set
    ``rollout_error_tolerance=RetryOnFailure(per_rollout_timeout=...)``.
    Rollouts still running when training ends are cancelled (their env
    cleanup still finishes), except those in a rollout executor's threads or
    processes, which run to completion. Up to
    ``groups_per_batch * (max_steps_off_policy + 1)`` rollouts run at once, so
    size a rollout executor (and any sandbox pool) for that many.

    Attributes:
        max_steps_off_policy (int): How many training steps sampling may run
            ahead of training. Typical staleness when sampling outpaces
            training is about this many steps.
        groups_per_batch (int): Number of trajectory groups per training batch.
        loss_fn (LossFnType): Loss function for async runs, used in place of
            ``Config.loss_fn``. An async run whose ``Config.loss_fn`` is set to
            something other than its default and this value raises a
            ``ConfigurationError``.
    """

    max_steps_off_policy: int
    groups_per_batch: int
    loss_fn: LossFnType = "stable_reinforce"

    @chz.validate
    def _check(self) -> None:
        if self.max_steps_off_policy < 0 or self.groups_per_batch < 1:
            raise ConfigurationError(
                "AsyncConfig needs max_steps_off_policy >= 0 and groups_per_batch >= 1, got "
                f"{self.max_steps_off_policy} and {self.groups_per_batch}."
            )


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
    # Slug identifying the recipe driving this run (e.g. "recipe_math_rl").
    # Attached as user_metadata on the ServiceClient so all downstream
    # training/sampling calls inherit it.
    recipe_name: str
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
    # Display name for the Tinker session; defaults to wandb_name when set.
    session_name: str | None = None

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
    # Async runs use AsyncConfig.loss_fn (default "stable_reinforce") instead.
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
    # None (default): the model's default strategy (MinViableGroup for
    # thinkingmachines/Inkling models, FailFast for everything else; see
    # rollout_presets.default_rollout_strategy_for_model).
    # False: crash on any error (FailFast).
    # True: retry failed trajectories with default budget (RetryOnFailure(max_retries=3)).
    # RolloutStrategy instance: custom strategy (e.g. RetryOnFailure(max_retries=5)).
    rollout_error_tolerance: bool | RolloutStrategy | None = None
    # Stop-reason-keyed reward semantics applied after group grading (e.g.
    # agentic().termination for grade-then-clamp: limit-stopped trajectories get
    # min(reward, 0)). None (default) resolves to the model's default
    # (agentic().termination for thinkingmachines/Inkling models, no policy for
    # everything else; see rollout_presets.default_rollout_config_for_model).
    # To force no reward adjustment on a model whose default has one, pass
    # TerminationRewardPolicy(zero_reward_on_limit=False). This is the
    # trainer-side half of a RolloutConfig; the env-side half is applied where
    # envs are built (e.g. build_agent_tool_env(rollout_config=...)).
    termination: TerminationRewardPolicy | None = None
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

    # Number of passes over the training dataset. Step i trains on
    # dataset.get_batch(i % len(dataset)); steps, checkpoints and resume use the
    # global step across epochs.
    num_epochs: int = 1
    # Hard cap on training steps. Training runs min(num_epochs * len(dataset),
    # max_steps) steps; to train longer than one pass, raise num_epochs (a larger
    # max_steps alone stops early, with a warning at startup).
    max_steps: int | None = None

    def training_loss_fn(self) -> LossFnType:
        """The loss this run trains with: ``async_config.loss_fn`` for async runs,
        ``loss_fn`` otherwise."""
        return self.async_config.loss_fn if self.async_config is not None else self.loss_fn

    def records_stability_info(self) -> bool:
        """Whether sampling must record the information StableReinforce trains on."""
        return self.training_loss_fn() == "stable_reinforce"

    def effective_termination(self) -> TerminationRewardPolicy | None:
        """The termination policy this run trains with.

        ``termination=None`` (the default) resolves via
        :func:`~tinker_cookbook.rl.rollout_presets.default_rollout_config_for_model`,
        so ``thinkingmachines/Inkling`` models get the agentic grade-then-clamp
        policy and other models get no reward adjustment.  An explicit
        policy wins.
        """
        if self.termination is not None:
            return self.termination
        return default_rollout_config_for_model(self.model_name).termination

    def effective_rollout_strategy(self) -> RolloutStrategy:
        """The group rollout strategy this run trains with.

        ``rollout_error_tolerance=None`` (the default) resolves via
        :func:`~tinker_cookbook.rl.rollout_presets.default_rollout_strategy_for_model`
        (``MinViableGroup`` for ``thinkingmachines/Inkling`` models, ``FailFast``
        otherwise).  Explicit values — ``False``, ``True``, or a strategy
        instance — win over the model default.
        """
        if self.rollout_error_tolerance is None:
            return default_rollout_strategy_for_model(self.model_name)
        return rollout_strategy_from_config(self.rollout_error_tolerance)


def _resolve_end_batch(max_steps: int | None, num_batches: int, num_epochs: int = 1) -> int:
    """Total RL steps: ``num_epochs`` passes over ``num_batches``, capped by ``max_steps``."""
    if num_epochs < 1:
        raise ConfigurationError(f"num_epochs must be >= 1, got {num_epochs}")
    total = num_batches * num_epochs
    if max_steps is None:
        return total
    if max_steps > total:
        logger.warning(
            f"max_steps={max_steps} exceeds num_epochs * len(dataset) = "
            f"{num_epochs} * {num_batches} = {total}; training will stop after {total} "
            f"steps. Set num_epochs to train for more passes over the dataset."
        )
    return min(max_steps, total)


def _drop_metrics_logged_after_checkpoint(store: TrainingRunStore | None, step: int) -> None:
    """Remove metrics rows a previous process logged after the checkpoint we resume from."""
    if store is None:
        return
    dropped = store.truncate_metrics(step)
    if dropped:
        logger.info(f"Resuming at step {step}: dropped {dropped} metrics rows with step >= {step}")


def rollout_usage(trajectory_groups: Sequence[TrajectoryGroup]) -> TokenUsage:
    """The sampling the groups' rollouts were billed for.

    Every action token was sampled and every turn's observation was prefilled,
    split into cached and uncached by the prefix-cache hits the sampler
    reported for that turn (``TokensWithLogprobs.prompt_cache_hit_tokens``).
    See :mod:`tinker_cookbook.utils.cost_estimation` for the USD formula.
    """
    transitions = [
        transition
        for group in trajectory_groups
        for trajectory in group.trajectories_G
        for transition in trajectory.transitions
    ]
    prompt_tokens = sum(t.ob.length for t in transitions)
    cached_tokens = sum(t.ac.prompt_cache_hit_tokens for t in transitions)
    return TokenUsage(
        sampled=sum(len(t.ac.tokens) for t in transitions),
        prefill_cached=cached_tokens,
        prefill_uncached=prompt_tokens - cached_tokens,
    )


def training_usage(data_D: list[tinker.Datum]) -> TokenUsage:
    """The tokens in the datums sent to ``forward_backward`` (shared prefixes across
    turns appear once)."""
    return TokenUsage(trained=sum(d.model_input.length for d in data_D))


def count_discarded_rollout(
    cost_tracker: CostTracker | None, trajectory_group: TrajectoryGroup
) -> None:
    """Count the sampling of a completed group that is dropped before training.

    The loops drop constant-reward groups before training, so they never
    reach :func:`prepare_minibatch`; this records their :func:`rollout_usage`
    on the tracker as discarded sampling, in the cumulative totals at once
    and apart from any step's own metrics.
    """
    if cost_tracker is not None:
        cost_tracker.record_discarded_sampling(rollout_usage([trajectory_group]))


DISCARDED_WORK_LOG_INTERVAL_SECONDS = 60.0


class DiscardedWorkLog:
    """Keeps the cost of dropped groups visible while no training step is logging.

    A dropped group's tokens go into the tracker's cumulative and discarded
    totals right away, but the metrics rows are logged by training steps. If
    every group keeps being dropped (or rollouts run far ahead of training),
    there is no step to log them, so this publishes the tracker's totals
    (:meth:`CostTracker.cumulative_metrics`) through
    :meth:`ml_log.Logger.log_summary` instead, at most once per
    *interval_seconds* and only after a drop. The summary is outside the step
    history, so it reaches W&B at once rather than when the step advances.
    """

    def __init__(
        self,
        cost_tracker: CostTracker | None,
        ml_logger: ml_log.Logger,
        *,
        interval_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cost_tracker = cost_tracker
        self.ml_logger = ml_logger
        self.interval_seconds = (
            DISCARDED_WORK_LOG_INTERVAL_SECONDS if interval_seconds is None else interval_seconds
        )
        self.clock = clock
        self._last_logged = clock()

    def step_logged(self) -> None:
        """A training step just logged the metrics; restart the interval."""
        self._last_logged = self.clock()

    def discarded(self, step: int) -> None:
        """A group was just dropped and counted; log the totals if it has been a while."""
        if self.cost_tracker is None:
            return
        now = self.clock()
        if now - self._last_logged < self.interval_seconds:
            return
        self._last_logged = now
        self.ml_logger.log_summary(self.cost_tracker.cumulative_metrics())


def drop_constant_reward_group(
    trajectory_group: TrajectoryGroup | None,
    config: Config,
    cost_tracker: CostTracker | None,
    discarded_log: DiscardedWorkLog | None = None,
    step: int = 0,
) -> TrajectoryGroup | None:
    """Apply ``config.remove_constant_reward_groups`` to one completed group.

    Replaces the filter inside ``do_group_rollout_and_filter_constant_reward``
    for the loops that stream groups one at a time, so the group's sampling
    work is counted (see :func:`count_discarded_rollout`) before it is dropped,
    and *discarded_log* (if given) can report the totals for *step*.
    """
    if (
        trajectory_group is None
        or not config.remove_constant_reward_groups
        or not all_same(trajectory_group.get_total_rewards())
    ):
        return trajectory_group
    count_discarded_rollout(cost_tracker, trajectory_group)
    if discarded_log is not None:
        discarded_log.discarded(step)
    return None


def _loop_state(batch: int, cost_tracker: CostTracker | None) -> dict[str, Any]:
    """Checkpoint ``loop_state``: the next batch plus the running token totals, for resume."""
    state: dict[str, Any] = {"batch": batch}
    if cost_tracker is not None:
        state["cumulative_tokens"] = asdict(cost_tracker.cumulative)
        state["discarded_tokens"] = asdict(cost_tracker.discarded)
    return state


@trace.scope
async def run_single_evaluation(
    evaluator: SamplingClientEvaluator,
    config: Config,
    i_batch: int,
    sampling_client: tinker.SamplingClient,
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
    sampling_client: tinker.SamplingClient,
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


async def _maybe_run_evals(
    evaluators: list[SamplingClientEvaluator],
    sampling_client: tinker.SamplingClient,
    config: Config,
    step: int,
    ml_logger: ml_log.Logger,
    force: bool = False,
) -> dict[str, Any]:
    """Run evaluations at ``step`` if it is on the ``eval_every`` cadence, or if ``force``.

    Evaluation is disabled entirely when ``eval_every == 0`` or there are no evaluators.
    """
    if config.eval_every == 0 or len(evaluators) == 0:
        return {}
    if not force and step % config.eval_every != 0:
        return {}
    async with trace.scope_span("run_evals"):
        return await run_evaluations_parallel(
            evaluators, sampling_client, config, step, store=ml_logger.store
        )


@trace.scope
async def do_sync_training_with_stream_minibatch(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    config: Config,
    training_client: tinker.TrainingClient,
    kl_reference_client: tinker.SamplingClient | None,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    error_counter: RolloutErrorCounter | None = None,
    strategy: RolloutStrategy | None = None,
    checkpoint_mgr: checkpoint_utils.CheckpointManager | None = None,
    cost_tracker: CostTracker | None = None,
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
        checkpoint_mgr (checkpoint_utils.CheckpointManager | None): Manager
            for periodic and rolling checkpoint saves.
        cost_tracker (CostTracker | None): Accumulates per-step token counts
            into the ``tokens/`` and ``cost/`` metrics (see
            :meth:`CostTracker.record_step`); its totals are checkpointed for resume.
    """
    discarded_log = DiscardedWorkLog(cost_tracker, ml_logger)
    # Initial sampling client
    assert checkpoint_mgr is not None
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client,
        checkpoint_mgr,
        start_batch,
        start_batch,
        record_stability_info=config.records_stability_info(),
    )

    for i_batch in range(start_batch, end_batch):
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }

        with trace.trace_iteration(step=i_batch) as window:
            # Run evaluations
            metrics.update(
                await _maybe_run_evals(evaluators, sampling_client, config, i_batch, ml_logger)
            )

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
                            do_remove_constant_reward_groups=False,
                            enable_logging=enable_logging,
                            strategy=strategy,
                            termination=config.effective_termination(),
                        )
                    trajectory_group = drop_constant_reward_group(
                        trajectory_group, config, cost_tracker, discarded_log, i_batch
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
                    tokenizer,
                    cost_tracker=cost_tracker,
                )
                # _Shutdown cannot appear in the sync path's local queue
                assert streaming_result is not None, "Unexpected shutdown in sync streaming path"
                (
                    new_sampling_client,
                    full_batch_metrics,
                    full_batch_wrapped_trajectory_groups,
                ) = streaming_result
                if new_sampling_client is not None:
                    sampling_client = new_sampling_client

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
                step=i_batch + 1, loop_state=_loop_state(i_batch + 1, cost_tracker)
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
        discarded_log.step_logged()

    if start_batch < end_batch:
        # Evals run before each iteration's train step, so the final weights need one more.
        final_metrics = await _maybe_run_evals(
            evaluators, sampling_client, config, end_batch, ml_logger, force=True
        )
        if final_metrics:
            ml_logger.log_metrics(final_metrics, step=end_batch)


@chz.chz
class WrappedTrajectoryGroup:
    """A wrapper around a trajectory group that includes generation metadata.

    Used when sampling and training are overlapped (streaming minibatch or
    async modes) so that the training loop knows which weights produced each group.

    Attributes:
        trajectory_group (TrajectoryGroup): The collected trajectory group.
        env_group_builder (EnvGroupBuilder): The builder that produced this
            group (used for logging tags).
        sampling_client_step (int): The training step at which the sampling
            client was created for this rollout.
        metrics (dict[str, Any]): Timing and worker-level metrics collected
            during rollout generation.
    """

    trajectory_group: TrajectoryGroup
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
    training_client: tinker.TrainingClient,
    kl_reference_client: tinker.SamplingClient | None,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    error_counter: RolloutErrorCounter | None = None,
    strategy: RolloutStrategy | None = None,
    checkpoint_mgr: checkpoint_utils.CheckpointManager | None = None,
    cost_tracker: CostTracker | None = None,
):
    """Implement async off-policy training (see :class:`AsyncConfig`).

    Launches four concurrent coroutine groups that communicate via async
    queues:

    1. **Dataloader loop** -- feeds ``EnvGroupBuilder`` items into a queue.
    2. **Trajectory worker loops** (one per rollout slot) -- take a
       slot, run a rollout, and push the ``WrappedTrajectoryGroup`` result.
    3. **Training loop** -- trains on batches of completed groups, oldest
       sampler version first (with streaming minibatches, in the order they
       finish), and frees their slots after each step.
    4. **Evaluation loop** -- runs evaluators whenever the sampling client is
       updated.

    If the dataset runs out, shutdown cascades from the dataloader through the
    workers to training and evaluation via ``_Shutdown`` sentinels. When
    training finishes, the dataloader and workers are cancelled, since nothing
    they produce could be trained on anymore.

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
        checkpoint_mgr (checkpoint_utils.CheckpointManager | None): Manager
            for periodic and rolling checkpoint saves.
        cost_tracker (CostTracker | None): Accumulates per-step token counts
            into the ``tokens/`` and ``cost/`` metrics (see
            :meth:`CostTracker.record_step`); its totals are checkpointed for resume.
    """
    assert config.async_config is not None

    # Buffer up to groups_per_batch builders ahead of the rollout workers.
    env_group_builders_queue = asyncio.Queue[EnvGroupBuilder | _Shutdown](
        maxsize=config.async_config.groups_per_batch
    )
    trajectory_groups_queue = asyncio.Queue[WrappedTrajectoryGroup | _Shutdown | None]()

    discarded_log = DiscardedWorkLog(cost_tracker, ml_logger)
    # Initial sampling client to use
    path_dict = await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name=f"{start_batch:06d}",
        log_path=config.log_path,
        loop_state=_loop_state(start_batch, cost_tracker),
        kind="both",
        ttl_seconds=config.ttl_seconds,
        store=ml_logger.store,
    )

    # Shutdown coordination when the dataset runs out — cascading sequence:
    # 1. Dataloader exhausts data → enqueues one _Shutdown sentinel per worker into
    #    env_group_builders_queue.
    # 2. Each trajectory worker receives its _Shutdown sentinel → exits and decrements
    #    worker_alive_counter. The last worker enqueues a _Shutdown into trajectory_groups_queue.
    # 3. Training loop receives _Shutdown from trajectory_groups_queue → trains any remaining
    #    full batches, sets evaluation_loop_should_shutdown_event, and exits.
    # 4. Eval loop sees evaluation_loop_should_shutdown_event → evaluates the final
    #    weights if it hasn't already, then exits.
    evaluation_loop_should_shutdown_event = asyncio.Event()
    # A worker takes a slot before starting a rollout; the training loop frees it
    # once the group has been trained (a worker frees it right away if its group
    # is filtered out). This caps how far sampling can run ahead of training.
    groups_per_step = (
        config.stream_minibatch_config.groups_per_batch
        if config.stream_minibatch_config is not None
        else config.async_config.groups_per_batch
    )
    num_rollout_slots = groups_per_step * (config.async_config.max_steps_off_policy + 1)
    rollout_slots = asyncio.Semaphore(num_rollout_slots)
    # One worker per slot: with fewer workers than slots, the worker count rather
    # than max_steps_off_policy would cap how far sampling runs ahead.
    num_rollout_workers = num_rollout_slots
    worker_alive_counter = _AsyncCounter(num_rollout_workers)

    # This will be updated by the training loop
    sampling_client = training_client.create_sampling_client(
        path_dict["sampler_path"], record_stability_info=config.records_stability_info()
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

        # Enqueue shutdown sentinels — one per worker — to cascade the shutdown
        logger.info("[dataloader_loop] No more data, shutting down trajectory group workers")
        for _ in range(num_rollout_workers):
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

            await rollout_slots.acquire()
            # Read the sampling client and its step together (no await in
            # between), after getting a slot, so the rollout uses the newest
            # weights and records the step they came from.
            rollout_sampling_client = sampling_client
            sampling_client_step_copy = sampling_client_step
            worker_metrics: dict[str, Any] = {}
            t_start = time.time()
            async with trace.scope_span("trajectory_group_worker"):
                trajectory_group = await do_group_rollout_and_filter_constant_reward(
                    rollout_sampling_client,
                    env_group_builder,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    do_remove_constant_reward_groups=False,
                    strategy=strategy,
                    termination=config.effective_termination(),
                )
            trajectory_group = drop_constant_reward_group(
                trajectory_group, config, cost_tracker, discarded_log, sampling_client_step_copy
            )
            worker_metrics["time/trajectory_group_worker_loop/total"] = time.time() - t_start
            # Ingest error info (safe: same event loop thread)
            if error_counter is not None:
                error_counter.ingest(trajectory_group)
            if trajectory_group is None:
                # Filtered out (e.g. constant reward): it will never be trained.
                rollout_slots.release()
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
        Trains on batches of completed trajectory groups, oldest sampler version first.
        No group is discarded for being stale.
        """
        assert config.async_config is not None
        groups_per_batch = config.async_config.groups_per_batch

        # Completed groups not yet trained on; leftovers carry over to later steps.
        completed_groups: list[WrappedTrajectoryGroup] = []

        async def collect_batch() -> list[WrappedTrajectoryGroup] | None:
            """Return the ``groups_per_batch`` oldest completed groups.

            Once the workers shut down, returns the remaining groups as a final,
            smaller batch rather than dropping them, and None when none remain.
            """
            # Wait for a full batch, then also take everything else that has
            # already finished, so the oldest groups are trained first.
            while len(completed_groups) < groups_per_batch or not trajectory_groups_queue.empty():
                item = await trajectory_groups_queue.get()
                if isinstance(item, _Shutdown):
                    # Leave the sentinel for later calls.
                    trajectory_groups_queue.put_nowait(item)
                    break
                if item is not None:
                    completed_groups.append(item)
            if not completed_groups:
                return None
            completed_groups.sort(key=lambda g: g.sampling_client_step)
            batch = completed_groups[:groups_per_batch]
            del completed_groups[:groups_per_batch]
            return batch

        i_batch = start_batch
        while i_batch < end_batch:
            metrics: dict[str, Any] = {
                "training_client/step": i_batch,
                "optim/lr": config.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
            }

            nonlocal sampling_client
            nonlocal sampling_client_step
            if config.stream_minibatch_config is not None:
                # Streaming minibatch: delegate queue consumption to the streaming function,
                # which returns None once it reaches the shutdown sentinel (a step that
                # shutdown cuts short after an optimizer substep is still returned and
                # published). Don't peek and re-queue here: a re-queued group would land
                # behind an already-enqueued sentinel and never be trained.
                with trace.trace_iteration(step=i_batch) as window:
                    streaming_result = await do_train_step_streaming_and_get_sampling_client(
                        config,
                        i_batch,
                        trajectory_groups_queue,
                        training_client,
                        checkpoint_mgr,
                        kl_reference_client,
                        tokenizer,
                        # Skip filtered (None) groups so each step trains a full batch.
                        lambda group: group is not None,
                        cost_tracker=cost_tracker,
                    )
                if streaming_result is None:
                    logger.info("[training_loop] Received shutdown signal from streaming")
                    break
                (
                    new_sampling_client,
                    train_step_metrics,
                    full_batch_wrapped_trajectory_groups,
                ) = streaming_result
                # The filter above skips dropped groups, so the step always trained.
                assert new_sampling_client is not None
                sampling_client = new_sampling_client
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
                trained_groups = full_batch_wrapped_trajectory_groups
            else:
                # Train on full batches, except for whatever is left when the data
                # runs out, which is trained as a final smaller batch.
                wrapped_trajectory_groups = await collect_batch()
                if wrapped_trajectory_groups is None:
                    logger.info("[training_loop] Received shutdown signal")
                    break
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
                        cost_tracker=cost_tracker,
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
                trained_groups = wrapped_trajectory_groups
            staleness = [i_batch - g.sampling_client_step for g in trained_groups]
            metrics["async/staleness_max"] = max(staleness)
            metrics["async/staleness_min"] = min(staleness)
            metrics["async/staleness_mean"] = sum(staleness) / len(staleness)
            sampling_client_step = i_batch + 1
            sampling_client_updated_event.set()
            # Free the trained groups' slots only now that the new sampling client
            # is in place (no await since the train step returned), so rollouts
            # started on these slots use the new weights.
            for _ in trained_groups:
                rollout_slots.release()

            # Rolling checkpoint (fire-and-forget, overlaps with next iteration)
            if checkpoint_mgr is not None:
                await checkpoint_mgr.maybe_save_rolling_async(
                    step=i_batch + 1, loop_state=_loop_state(i_batch + 1, cost_tracker)
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
            discarded_log.step_logged()
            i_batch += 1

        if i_batch < end_batch:
            logger.warning(
                f"Async training ran out of data after {i_batch - start_batch} of "
                f"{end_batch - start_batch} steps. Each step trains {groups_per_step} "
                "groups that survive filtering and rollout errors, so steps run out "
                "before dataset batches do when groups are filtered or fail, or when "
                "dataset batches hold fewer groups than that. To train more steps, "
                "raise max_steps (and num_epochs, if it caps the run)."
            )

        # Signal evaluation loop to shut down
        evaluation_loop_should_shutdown_event.set()
        sampling_client_updated_event.set()
        logger.info("[training_loop] Terminated")

    @trace.scope
    async def evaluation_loop():
        """Runs evals periodically, matching the sync training eval pipeline."""
        if len(evaluators) == 0 or config.eval_every == 0:
            return

        last_eval_step: int | None = None
        while True:
            await sampling_client_updated_event.wait()
            sampling_client_updated_event.clear()
            # Read the flag before the step: the training loop sets it only after
            # publishing its final sampling client, so seeing it means the client
            # read below holds the final weights.
            shutting_down = evaluation_loop_should_shutdown_event.is_set()

            # Save a reference to the original values in case it changes
            # while we're running the evals
            sampling_client_eval_step = sampling_client_step
            sampling_client_eval = sampling_client
            # Always evaluate the final weights, even off the eval_every cadence
            # or if an earlier eval was still running when training finished.
            is_final = shutting_down and sampling_client_eval_step > start_batch
            on_cadence = (
                config.eval_every > 0 and sampling_client_eval_step % config.eval_every == 0
            )
            if sampling_client_eval_step != last_eval_step and (on_cadence or is_final):
                last_eval_step = sampling_client_eval_step
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
            if shutting_down:
                break
        logger.info("[evaluation_loop] Terminated")

    producer_tasks = [
        asyncio.create_task(dataloader_loop(), name="dataloader_loop"),
        *[
            asyncio.create_task(
                trajectory_group_worker_loop(), name=f"trajectory_group_worker_loop_{i}"
            )
            for i in range(num_rollout_workers)
        ],
    ]
    training_task = asyncio.create_task(training_loop(), name="training_loop")
    evaluation_task = asyncio.create_task(evaluation_loop(), name="evaluation_loop")
    # A task leaves `pending` when it finishes or when we cancel it, so each is
    # cancelled at most once and env cleanup is never interrupted by a second
    # cancellation.
    pending = {*producer_tasks, training_task, evaluation_task}
    try:
        # Run until every task is done, re-raising the first exception from any of them.
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if training_task in done:
                # Nothing the dataloader and workers produce can be trained
                # anymore; stop them while the last evaluation finishes.
                for task in producer_tasks:
                    task.cancel()
                pending.difference_update(producer_tasks)
    finally:
        # Cancel whatever is still running (rollouts still in flight can never be
        # trained on) and wait for everything, so nothing outlives this call.
        for task in pending:
            task.cancel()
        await asyncio.gather(
            *producer_tasks, training_task, evaluation_task, return_exceptions=True
        )


@trace.scope
async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    checkpoint_mgr: checkpoint_utils.CheckpointManager,
    i_batch: int,
    start_batch: int = 0,
    record_stability_info: bool = False,
    cost_tracker: CostTracker | None = None,
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
        record_stability_info (bool): Whether the sampling client records the
            per-sequence information that the ``stable_reinforce`` loss needs.
            Defaults to False.
        cost_tracker (CostTracker | None): Its running token totals are
            stored in the checkpoint's ``loop_state`` so a resumed run
            continues them.

    Returns:
        tuple[tinker.SamplingClient, dict[str, Any]]: A sampling client
        loaded with the latest weights, and a (possibly empty) metrics dict.
    """
    metrics: dict[str, Any] = {}
    if i_batch > start_batch and checkpoint_mgr.should_save_periodic(i_batch):
        path_dict = await checkpoint_mgr.save_periodic_async(
            step=i_batch, loop_state=_loop_state(i_batch, cost_tracker)
        )
        sampling_client = training_client.create_sampling_client(
            path_dict["sampler_path"], record_stability_info=record_stability_info
        )
        return sampling_client, metrics
    else:
        async with trace.scope_span("save_checkpoint"):
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                record_stability_info=record_stability_info
            )
            return sampling_client, metrics


@trace.scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    kl_reference_client: tinker.SamplingClient | None,
    kl_penalty_coef: float,
    kl_discount_factor: float,
    include_provenance: bool = True,
) -> tuple[list[tinker.Datum], dict[str, Any], TokenUsage]:
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
        include_provenance (bool): Attach provenance spans to the datums
            (see :func:`trajectory_to_data`). Defaults to True.

    Returns:
        tuple[list[tinker.Datum], dict[str, Any], TokenUsage]: A list of
        training datums, a dict of trajectory and KL penalty metrics, and the
        token usage so far for this data: the rollouts' sampling
        (:func:`rollout_usage`) and the reference-KL scoring calls (as their
        responses report it).
    """

    # Compute trajectory metrics
    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))
    usage = rollout_usage(trajectory_groups_P)

    # Print up to two trajectory groups
    for traj_group in trajectory_groups_P[:2]:
        print_group(traj_group, tokenizer)

    # Assemble training data
    async with trace.scope_span("assemble_training_data"):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, _metadata_D = assemble_training_data(
            trajectory_groups_P, advantages_P, include_provenance=include_provenance
        )

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0 and kl_reference_client is not None:
        async with trace.scope_span("kl_vs_base"):
            kl_penalty_metrics, kl_usage = await incorporate_kl_penalty(
                data_D,
                kl_reference_client,
                kl_penalty_coef,
                kl_discount_factor,
            )
        metrics.update(kl_penalty_metrics)
        usage += kl_usage

    return data_D, metrics, usage


@trace.scope
async def compute_full_batch_metrics_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    checkpoint_mgr: checkpoint_utils.CheckpointManager,
    i_batch: int,
    data_D: list[tinker.Datum],
    training_logprobs_D: list[torch.Tensor],
    do_compute_post_kl: bool,
    record_stability_info: bool = False,
    usage: TokenUsage = TokenUsage(),
    cost_tracker: CostTracker | None = None,
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
        record_stability_info (bool): Passed to
            :func:`save_checkpoint_and_get_sampling_client`. Defaults to False.
        usage (TokenUsage): The iteration's token usage so far (its rollouts
            and reference-KL scoring, from :func:`prepare_minibatch`); the
            trained tokens are added here.
        cost_tracker (CostTracker | None): Records this iteration's usage.
            Everything measured by the time the checkpoint is saved (rollouts,
            reference KL, training) is recorded before the save, so the
            totals stored with it cover this iteration. The post-update KL
            calls run against the new weights after the save, so their usage
            is recorded once they return (:meth:`CostTracker.extend_step`):
            it is in this step's metrics and in every later checkpoint, but
            not in the one saved by this call.

    Returns:
        tuple[tinker.SamplingClient, dict[str, Any]]: A sampling client
        loaded with post-update weights, and a dict of KL / checkpoint /
        cost metrics.
    """
    metrics: dict[str, Any] = {}

    # Compute KL metrics
    async with trace.scope_span("compute_kl_sample_train"):
        kl_sample_train_metrics = compute_kl_sample_train(data_D, training_logprobs_D)
        metrics.update(kl_sample_train_metrics)

    if cost_tracker is not None:
        metrics.update(cost_tracker.record_step(usage + training_usage(data_D)))

    # Get a sampling client using the new weights
    sampling_client, checkpoint_metrics = await save_checkpoint_and_get_sampling_client(
        training_client,
        checkpoint_mgr,
        i_batch,
        record_stability_info=record_stability_info,
        cost_tracker=cost_tracker,
    )
    metrics.update(checkpoint_metrics)

    # Compute post-KL metrics if configured
    if do_compute_post_kl:
        async with trace.scope_span("compute_post_kl"):
            post_kl_metrics, post_kl_usage = await compute_post_kl(data_D, sampling_client)
            metrics.update(post_kl_metrics)
        if cost_tracker is not None:
            metrics.update(cost_tracker.extend_step(post_kl_usage))

    return sampling_client, metrics


@trace.scope
async def do_train_step_streaming_and_get_sampling_client(
    config: Config,
    i_batch: int,
    trajectory_groups_queue: asyncio.Queue[WrappedTrajectoryGroup | _Shutdown | None],
    training_client: tinker.TrainingClient,
    checkpoint_mgr: checkpoint_utils.CheckpointManager,
    kl_reference_client: tinker.SamplingClient | None,
    tokenizer: Tokenizer,
    trajectory_group_filter: Callable[[WrappedTrajectoryGroup | None], bool] = lambda _: True,
    cost_tracker: CostTracker | None = None,
) -> tuple[tinker.SamplingClient | None, dict[str, Any], list[WrappedTrajectoryGroup]] | None:
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
        if a ``_Shutdown`` sentinel arrived before the step received any group.
        When every group of the batch was filtered out there is no update: the
        sampling client in the tuple is None (keep using the current one), the
        metrics carry the step's cost accounting, and the list is empty.
        A sentinel that arrives later ends the step early instead: the groups
        received so far, including a partial minibatch, are trained and published
        as a shorter step, and the sentinel is put back for the next call.
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
    usage = TokenUsage()

    # Run multiple optimizer substeps per training iteration
    all_data_D = []
    all_training_logprobs_D = []
    all_wrapped_trajectory_groups = []
    shutting_down = False
    for i_substep in range(config.num_substeps):
        # Run multiple minibatches per substep
        # Once we have enough trajectories for a minibatch, train on them
        wrapped_trajectory_groups = []
        forward_backward_futures: list[tinker.APIFuture[tinker.ForwardBackwardOutput]] = []
        i_minibatch = 0
        while i_minibatch < config.stream_minibatch_config.num_minibatches:
            wrapped_trajectory_group = await trajectory_groups_queue.get()
            if isinstance(wrapped_trajectory_group, _Shutdown):
                logger.info("[do_train_step_streaming] Received shutdown signal")
                # Finish the step with what it already has rather than drop those
                # groups or leave an applied substep unpublished: train any partial
                # minibatch below, then stop. Leave the sentinel for the next call.
                trajectory_groups_queue.put_nowait(wrapped_trajectory_group)
                shutting_down = True
                if not wrapped_trajectory_groups:
                    break
            else:
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

            data_D, prepare_minibatch_metrics, minibatch_usage = await prepare_minibatch(
                [g.env_group_builder for g in wrapped_trajectory_groups],
                [g.trajectory_group for g in wrapped_trajectory_groups],
                tokenizer,
                kl_reference_client,
                kl_penalty_coef=config.kl_penalty_coef,
                kl_discount_factor=config.kl_discount_factor,
            )
            metrics.update(prepare_minibatch_metrics)
            usage += minibatch_usage

            # Enqueue forward-backward (we'll await results after all minibatches are enqueued)
            async with trace.scope_span(
                f"train/fwd_bwd_substep_{i_substep}_mb_{i_minibatch}_enqueue"
            ):
                _check_provenance(data_D, config.training_loss_fn())
                forward_backward_futures.append(
                    await training_client.forward_backward_async(
                        [_remove_mask(d) for d in data_D],
                        loss_fn=config.training_loss_fn(),
                        loss_fn_config=config.loss_fn_config,
                    )
                )
            all_data_D.extend(data_D)
            all_wrapped_trajectory_groups.extend(wrapped_trajectory_groups)
            i_minibatch += 1
            wrapped_trajectory_groups = []

        if not forward_backward_futures:
            # Nothing to apply: every group of this substep was filtered out.
            if shutting_down:
                break
            continue

        # Enqueue optim_step before awaiting results (so they land on same clock cycle)
        adam_params = tinker.AdamParams(
            learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
        )
        async with trace.scope_span(f"train/optim_substep_{i_substep}_enqueue"):
            optim_future = await training_client.optim_step_async(adam_params)

        # Now consume all forward-backward results
        for i_mb, fwd_bwd_future in enumerate(forward_backward_futures):
            async with trace.scope_span(f"train/fwd_bwd_substep_{i_substep}_mb_{i_mb}_consume"):
                fwd_bwd_result = await fwd_bwd_future.result_async()
                all_training_logprobs_D.extend(_training_logprobs_from_fwd_bwd(fwd_bwd_result))

        async with trace.scope_span(f"train/optim_substep_{i_substep}_consume"):
            optim_result = await optim_future.result_async()

        if optim_result.metrics:
            metrics.update(optim_result.metrics)
        if shutting_down:
            break

    if not all_wrapped_trajectory_groups:
        if shutting_down:
            # Shut down before this step trained anything.
            return None
        # Every group of the batch was filtered out: no optimizer step, no new
        # weights, and a step row with zero counts. The groups' sampling was
        # counted as discarded when they were dropped and shows in the totals.
        logger.info(f"[stream_minibatch] Step {i_batch}: every group was filtered out, no update")
        if cost_tracker is not None:
            metrics.update(cost_tracker.record_step(TokenUsage()))
        return None, metrics, []

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
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        all_data_D,
        all_training_logprobs_D,
        config.compute_post_kl,
        record_stability_info=config.records_stability_info(),
        usage=usage,
        cost_tracker=cost_tracker,
    )
    metrics.update(full_batch_metrics)
    return sampling_client, metrics, all_wrapped_trajectory_groups


@trace.scope
async def do_train_step_and_get_sampling_client(
    config: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    checkpoint_mgr: checkpoint_utils.CheckpointManager,
    kl_reference_client: tinker.SamplingClient | None,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    cost_tracker: CostTracker | None = None,
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
    data_D, prepare_minibatch_metrics, usage = await prepare_minibatch(
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
        loss_fn=config.training_loss_fn(),
        loss_fn_config=config.loss_fn_config,
        metrics=metrics,
    )

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        checkpoint_mgr,
        # NOTE: saving the checkpoint as the i + 1 step
        i_batch + 1,
        data_D,
        training_logprobs_D,
        config.compute_post_kl,
        record_stability_info=config.records_stability_info(),
        usage=usage,
        cost_tracker=cost_tracker,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@trace.scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    config: Config,
    training_client: tinker.TrainingClient,
    kl_reference_client: tinker.SamplingClient | None,
    evaluators: list[SamplingClientEvaluator],
    dataset: RLDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    error_counter: RolloutErrorCounter | None = None,
    strategy: RolloutStrategy | None = None,
    checkpoint_mgr: checkpoint_utils.CheckpointManager | None = None,
    cost_tracker: CostTracker | None = None,
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
        checkpoint_mgr (checkpoint_utils.CheckpointManager | None): Manager
            for periodic and rolling checkpoint saves.
        cost_tracker (CostTracker | None): Accumulates per-step token counts
            into the ``tokens/`` and ``cost/`` metrics (see
            :meth:`CostTracker.record_step`); its totals are checkpointed for resume.
    """
    discarded_log = DiscardedWorkLog(cost_tracker, ml_logger)
    # Initial sampling client
    assert checkpoint_mgr is not None
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client,
        checkpoint_mgr,
        start_batch,
        start_batch,
        record_stability_info=config.records_stability_info(),
    )

    for i_batch in range(start_batch, end_batch):
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }

        with trace.trace_iteration(step=i_batch) as window:
            # Run evaluations
            metrics.update(
                await _maybe_run_evals(evaluators, sampling_client, config, i_batch, ml_logger)
            )

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
                                termination=config.effective_termination(),
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
                    kept_groups_P = remove_constant_reward_groups(trajectory_groups_P)
                    kept_ids = {id(group) for group in kept_groups_P}
                    for group in trajectory_groups_P:
                        if id(group) not in kept_ids:
                            count_discarded_rollout(cost_tracker, group)
                            discarded_log.discarded(i_batch)
                    trajectory_groups_P = kept_groups_P

                # Train step
                sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
                    config,
                    i_batch,
                    training_client,
                    checkpoint_mgr,
                    kl_reference_client,
                    tokenizer,
                    env_group_builders_P,
                    trajectory_groups_P,
                    cost_tracker=cost_tracker,
                )

                metrics.update(train_step_metrics)

                # Rolling checkpoint (fire-and-forget, overlaps with next iteration)
                if checkpoint_mgr is not None:
                    await checkpoint_mgr.maybe_save_rolling_async(
                        step=i_batch + 1, loop_state=_loop_state(i_batch + 1, cost_tracker)
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
        discarded_log.step_logged()

    if start_batch < end_batch:
        # Evals run before each iteration's train step, so the final weights need one more.
        final_metrics = await _maybe_run_evals(
            evaluators, sampling_client, config, end_batch, ml_logger, force=True
        )
        if final_metrics:
            ml_logger.log_metrics(final_metrics, step=end_batch)


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
            workers. Pass ``ProcessPoolExecutor(max_workers=N,
            mp_context=multiprocessing.get_context("spawn"))`` for
            multi-process execution, or any custom ``Executor`` (Ray,
            cluster dispatchers, etc.). Process pools must use the
            ``"spawn"`` (or ``"forkserver"``) start method: the Tinker
            client's background event-loop thread does not survive
            ``fork()``, so fork-started workers (the Linux default) hang
            silently. Default ``None`` runs rollouts as asyncio
            coroutines in-process.

    Raises:
        ConfigurationError: If ``kl_penalty_coef > 0`` but
            ``kl_reference_config`` is not set, or if the loss is
            ``stable_reinforce`` with ``kl_penalty_coef > 0`` and
            ``kl_discount_factor > 0``.

    Example::

        import asyncio
        from tinker_cookbook import cli_utils
        from tinker_cookbook.rl.train import Config, main

        config = Config(
            learning_rate=1e-5,
            dataset_builder=my_dataset_builder,
            model_name="Qwen/Qwen3.5-9B",
            renderer_name="qwen3_5_disable_thinking",
            max_tokens=2048,
            log_path=cli_utils.runs_path("my-rl-run"),
        )
        asyncio.run(main(config=config))
    """

    if (
        config.async_config is not None
        and config.loss_fn != "importance_sampling"
        and config.loss_fn != config.async_config.loss_fn
    ):
        raise ConfigurationError(
            f"Config.loss_fn={config.loss_fn!r} conflicts with "
            f"AsyncConfig.loss_fn={config.async_config.loss_fn!r}; async runs train with "
            "AsyncConfig.loss_fn, so set the loss there."
        )
    if (
        config.training_loss_fn() == "stable_reinforce"
        and config.kl_penalty_coef > 0
        and config.kl_discount_factor > 0
    ):
        # Discounting spreads the KL penalty onto prompt tokens, but stable_reinforce
        # only accepts nonzero advantages on sampled tokens.
        raise ConfigurationError(
            "kl_discount_factor > 0 is not supported with loss_fn='stable_reinforce' "
            "(the AsyncConfig default); set kl_discount_factor=0 or choose another "
            "loss_fn."
        )
    if rollout_executor is not None:
        set_rollout_executor(rollout_executor)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=config.wandb_name,
    )
    store = ml_logger.store
    if (
        config.async_config is not None
        and config.async_config.max_steps_off_policy > 0
        and config.training_loss_fn() in ("importance_sampling", "cispo")
    ):
        logger.warning(
            f"loss_fn={config.training_loss_fn()!r} can become unstable on the off-policy "
            "data that async training produces; prefer 'stable_reinforce' (the async default) "
            "or a clipped loss such as 'ppo'."
        )
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
        _drop_metrics_logged_after_checkpoint(store, start_batch)
        # Older checkpoints carry no totals; they resume from zero.
        resumed_usage = TokenUsage(**resume_info.get("cumulative_tokens", {}))
        resumed_discarded = TokenUsage(**resume_info.get("discarded_tokens", {}))
    else:
        start_batch = 0
        resumed_usage = TokenUsage()
        resumed_discarded = TokenUsage()
    cost_tracker = CostTracker(
        fetch_model_pricing(config.model_name),
        cumulative=resumed_usage,
        discarded=resumed_discarded,
    )

    session_display_name = resolve_session_display_name(
        session_name=config.session_name,
        wandb_name=config.wandb_name,
    )
    if session_display_name is not None:
        logger.info(f"Tinker session display name: {session_display_name}")
    with tinker.ServiceClient(
        base_url=config.base_url,
        user_metadata=recipe_user_metadata(config.recipe_name),
        display_name=session_display_name,
    ) as service_client:
        user_metadata: dict[str, str] = {}
        if wandb_link := ml_logger.get_logger_url():
            user_metadata["wandb_link"] = wandb_link
        checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)
        model_info.warn_if_renderer_not_recommended(config.model_name, config.renderer_name)

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
        else:
            training_client = await service_client.create_lora_training_client_async(
                config.model_name, rank=config.lora_rank, user_metadata=user_metadata
            )

        # Get tokenizer.
        tokenizer = get_tokenizer(config.model_name)

        # Create dataset from thunk
        dataset, maybe_test_dataset = await config.dataset_builder()
        # Build rollout strategy and error counter from config
        strategy = config.effective_rollout_strategy()
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

        end_batch = _resolve_end_batch(config.max_steps, len(dataset), config.num_epochs)
        dataset = EpochCycledRLDataset(dataset, config.num_epochs)
        logger.info(f"Will train on {end_batch} batches")

        # Create KL reference client once if KL penalty is enabled
        if config.kl_penalty_coef > 0:
            if config.kl_reference_config is None:
                raise ConfigurationError(
                    "kl_reference_config must be specified when kl_penalty_coef > 0"
                )
            kl_reference_client = service_client.create_sampling_client(
                base_model=config.kl_reference_config.base_model,
                model_path=config.kl_reference_config.load_checkpoint_path,
            )
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
            evaluators=evaluators,
            dataset=dataset,
            ml_logger=ml_logger,
            tokenizer=tokenizer,
            error_counter=error_counter,
            strategy=strategy,
            checkpoint_mgr=checkpoint_mgr,
            cost_tracker=cost_tracker,
        )

        # The final totals, discarded sampling included, even if no step ran. A
        # step row, not a summary: it comes after the last step, so every backend
        # publishes it, including those with no summary support.
        ml_logger.log_metrics(cost_tracker.cumulative_metrics(), step=end_batch)

        # Save final checkpoint
        if start_batch < end_batch:
            await checkpoint_mgr.save_final_async(loop_state=_loop_state(end_batch, cost_tracker))
        else:
            logger.info("Training was already complete; nothing to do")
            await checkpoint_mgr.finalize_async()

    # Cleanup
    if rollout_executor is not None:
        rollout_executor.shutdown(wait=True)
        set_rollout_executor(None)
    ml_logger.close()
    logger.info("Training completed successfully")
