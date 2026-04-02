"""
Self-Distilled Policy Optimization (SDPO) — core training logic.

SDPO (https://arxiv.org/abs/2601.20802) is an on-policy RL algorithm that
provides dense, token-level credit assignment by distilling from the model's
own successful trajectories.

Standard RL (e.g. GRPO) assigns a single scalar reward per sequence — every
token in a correct solution gets the same credit. SDPO instead constructs a
"teacher" by conditioning a reference model on a successful solution, then
uses the teacher's distribution as soft targets for training.

Two distillation modes are supported (controlled by ``Config.topk``):

- **Top-K distillation** (``topk > 0``, default): Recovers the teacher's top-K
  token distribution at each completion position using Tinker's
  ``topk_prompt_logprobs`` API and trains with ``cross_entropy`` loss. This
  approximates the paper's full-vocabulary JSD.

- **Per-token importance sampling** (``topk = 0``): Uses per-token
  ``advantage = teacher_lp - student_lp`` with ``importance_sampling`` loss.
  Single-token approximation — the teacher signal only covers the sampled token.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import chz
import tinker

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.distillation.sdft import (
    build_topk_distillation_datums,
    compute_sdft_advantages,
)
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.rollout_logging import (
    RolloutSummaryExportConfig,
    rollout_summaries_jsonl_path,
    write_rollout_summaries_jsonl,
)
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.train import gather_with_progress
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDatasetBuilder, TrajectoryGroup
from tinker_cookbook.sdpo.data import (
    extract_feedback,
    extract_response_tokens,
)
from tinker_cookbook.sdpo.teacher import (
    build_teacher_prompt,
    build_teacher_prompt_from_messages,
    strip_thinking_blocks,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import logtree, ml_log
from tinker_cookbook.utils.misc_utils import timed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Teacher prompt builder dispatch
# ---------------------------------------------------------------------------


def _resolve_teacher_prompt_builder(
    builder: EnvGroupBuilder,
) -> Callable[..., tinker.ModelInput]:
    """Return a callable that builds teacher prompts for the given builder type.

    For ProblemGroupBuilder (math, MCQ): uses ProblemEnv's question and renderer.
    For DeepcoderEnvGroupBuilder (code): uses the task's problem and tool schemas.
    """
    if isinstance(builder, ProblemGroupBuilder):
        env = cast(ProblemEnv, builder.env_thunk())
        return lambda reprompt_suffix, **kwargs: build_teacher_prompt(
            env, reprompt_suffix, **kwargs
        )

    # Import here to avoid circular dependency and keep code_rl optional.
    from tinker_cookbook.recipes.code_rl.code_env import DeepcoderEnvGroupBuilder

    if isinstance(builder, DeepcoderEnvGroupBuilder):
        from tinker_cookbook import model_info, tokenizer_utils
        from tinker_cookbook.recipes.code_rl.deepcoder_tool import DeepcoderTool
        from tinker_cookbook.renderers import get_renderer

        tok = tokenizer_utils.get_tokenizer(builder.model_name)
        rname = builder.renderer_name or model_info.get_recommended_renderer_name(
            builder.model_name
        )
        renderer = get_renderer(rname, tok)
        tool = DeepcoderTool(builder.task)
        tool_schemas = [tool.check_solution.to_spec()]
        convo_prefix = renderer.create_conversation_prefix_with_tools(tools=tool_schemas)
        question = builder.task.problem

        return lambda reprompt_suffix, **kwargs: build_teacher_prompt_from_messages(
            convo_prefix=convo_prefix,
            question=question,
            renderer=renderer,
            reprompt_suffix=reprompt_suffix,
            **kwargs,
        )

    raise TypeError(f"Unsupported builder type: {type(builder).__name__}")


# ---------------------------------------------------------------------------
# Logtree / rollout export helpers
# ---------------------------------------------------------------------------


@contextmanager
def _get_logtree_scope(
    log_path: str | None, num_groups_to_log: int, f_name: str, scope_name: str
) -> Iterator[None]:
    if log_path is None or num_groups_to_log <= 0:
        yield
        return

    logtree_path = str(Path(log_path) / f"{f_name}.html")
    logtree_json_path = str(Path(log_path) / f"{f_name}_logtree.json")
    trace = None
    try:
        with logtree.init_trace(scope_name, path=logtree_path) as trace:
            yield
    finally:
        if trace is not None:
            logtree.write_trace_json(trace, logtree_json_path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@chz.chz
class Config:
    """Configuration for SDPO training."""

    # Required
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    model_name: str
    dataset_builder: RLDatasetBuilder

    # Training
    learning_rate: float = 1e-5
    max_tokens: int = 2048
    temperature: float = 1.0
    lora_rank: int = 32

    # SDPO-specific
    success_reward_threshold: float = 0.5
    reprompt_suffix: str = "Correctly solve the original question."
    dont_reprompt_on_self_success: bool = True
    remove_thinking_from_demonstration: bool = True
    include_environment_feedback: bool = False
    max_context_length: int = 32768

    # Distillation mode
    topk: int = 20  # 0 = importance_sampling fallback

    # Infrastructure
    renderer_name: str | None = None
    base_url: str | None = None
    load_checkpoint_path: str | None = None

    # Checkpointing / eval
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    eval_every: int = 10
    save_every: int = 10
    ttl_seconds: int | None = 604800  # 7 days

    # Logging
    wandb_project: str | None = None
    wandb_name: str | None = None
    num_groups_to_log: int = 4
    rollout_json_export: bool = True


# ---------------------------------------------------------------------------
# SDPO: select successful trajectories and build teacher prompts
# ---------------------------------------------------------------------------


def _build_teacher_prompts_and_filter(
    trajectory_groups: list[TrajectoryGroup],
    env_group_builders: Sequence[EnvGroupBuilder],
    tokenizer: Tokenizer,
    config: Config,
) -> tuple[
    list[TrajectoryGroup],
    list[list[tinker.ModelInput | None]],
    dict[str, float | int],
]:
    """Identify successes, build per-trajectory teacher prompts, filter empty groups.

    Returns:
        filtered_groups: TrajectoryGroups with at least one teacher signal.
        teacher_prompts_P_G: Per-group, per-trajectory teacher prompts (None
            for trajectories that should not be trained on).
        metrics: SDPO-specific metrics (success rates, etc.).
    """
    filtered_groups: list[TrajectoryGroup] = []
    teacher_prompts_P_G: list[list[tinker.ModelInput | None]] = []

    n_groups_with_success = 0
    n_total_trajectories = 0
    n_sdpo_trajectories = 0
    total_success_rate = 0.0

    for group, builder in zip(trajectory_groups, env_group_builders, strict=True):
        rewards = group.get_total_rewards()
        n_total_trajectories += len(rewards)
        n_successes = sum(1 for r in rewards if r >= config.success_reward_threshold)
        total_success_rate += n_successes / len(rewards)

        successful_indices = [
            i for i, r in enumerate(rewards) if r >= config.success_reward_threshold
        ]

        has_solutions = len(successful_indices) > 0
        if not has_solutions and not config.include_environment_feedback:
            continue

        n_groups_with_success += 1 if has_solutions else 0

        teacher_prompt_builder = _resolve_teacher_prompt_builder(builder)
        prompts_G: list[tinker.ModelInput | None] = []

        for traj_idx, traj in enumerate(group.trajectories_G):
            response_tokens = extract_response_tokens(traj)
            if not response_tokens:
                prompts_G.append(None)
                continue

            # Solution: pick a successful rollout from the group.
            solution_text: str | None = None
            if has_solutions:
                if config.dont_reprompt_on_self_success:
                    other_successes = [i for i in successful_indices if i != traj_idx]
                    solution_idx = other_successes[0] if other_successes else successful_indices[0]
                else:
                    solution_idx = successful_indices[0]
                solution_tokens = extract_response_tokens(group.trajectories_G[solution_idx])
                solution_text = tokenizer.decode(solution_tokens)
                if config.remove_thinking_from_demonstration:
                    solution_text = strip_thinking_blocks(solution_text)

            # Feedback from this trajectory's own execution logs.
            feedback_text: str | None = None
            if config.include_environment_feedback:
                feedback_text = extract_feedback(traj)

            if solution_text is None and feedback_text is None:
                prompts_G.append(None)
                continue

            teacher_ob = teacher_prompt_builder(
                config.reprompt_suffix,
                solution_text=solution_text,
                feedback_text=feedback_text,
            )
            prompts_G.append(teacher_ob)
            n_sdpo_trajectories += 1

        # Only keep groups that have at least one trajectory with teacher signal.
        if any(p is not None for p in prompts_G):
            filtered_groups.append(group)
            teacher_prompts_P_G.append(prompts_G)

    n_groups = len(trajectory_groups)
    metrics: dict[str, float | int] = {
        "sdpo/groups_with_success": n_groups_with_success,
        "sdpo/groups_total": n_groups,
        "sdpo/success_fraction": n_groups_with_success / n_groups if n_groups else 0.0,
        "sdpo/trajectories_trained": n_sdpo_trajectories,
        "sdpo/trajectories_total": n_total_trajectories,
        "sdpo/mean_group_success_rate": total_success_rate / n_groups if n_groups else 0.0,
    }
    return filtered_groups, teacher_prompts_P_G, metrics


def _flatten_teacher_prompts(
    data_D: list[tinker.Datum],
    metadata_D: list[dict[str, int]],
    teacher_prompts_P_G: list[list[tinker.ModelInput | None]],
) -> tuple[list[tinker.ModelInput], list[dict[str, int]], list[int]]:
    """Build per-datum teacher prompts and remap metadata for the distillation functions.

    The SDFT distillation functions (``build_topk_distillation_datums`` and
    ``compute_sdft_advantages``) expect teacher prompts indexed by
    ``metadata[i]["group_idx"]``. SDPO's teacher prompts vary per trajectory
    (not just per group), so we flatten them into a per-datum list and remap
    each datum's ``group_idx`` to point to its own teacher prompt.

    Datums whose trajectory has no teacher prompt (None) are dropped.

    Returns:
        (teacher_prompts_flat, remapped_metadata, kept_indices): teacher prompts
        and metadata aligned with the filtered datums, plus indices into data_D
        for the caller to filter.
    """
    teacher_prompts_flat: list[tinker.ModelInput] = []
    remapped_metadata: list[dict[str, int]] = []
    kept_indices: list[int] = []

    for i, meta in enumerate(metadata_D):
        group_idx = meta["group_idx"]
        traj_idx = meta["traj_idx"]
        teacher_prompt = teacher_prompts_P_G[group_idx][traj_idx]
        if teacher_prompt is None:
            continue
        flat_idx = len(teacher_prompts_flat)
        teacher_prompts_flat.append(teacher_prompt)
        remapped_metadata.append({"group_idx": flat_idx, "traj_idx": traj_idx})
        kept_indices.append(i)

    return teacher_prompts_flat, remapped_metadata, kept_indices


def _scale_advantages(data_D: list[tinker.Datum], scale: float) -> list[tinker.Datum]:
    """Return new datums with advantages scaled by a constant factor."""
    scaled: list[tinker.Datum] = []
    for datum in data_D:
        new_inputs = dict(datum.loss_fn_inputs)
        adv = datum.loss_fn_inputs["advantages"].to_torch()
        new_inputs["advantages"] = tinker.TensorData.from_torch(adv * scale)
        scaled.append(tinker.Datum(model_input=datum.model_input, loss_fn_inputs=new_inputs))
    return scaled


# ---------------------------------------------------------------------------
# SDPO training iteration
# ---------------------------------------------------------------------------


async def sdpo_training_iteration(
    trajectory_groups: list[TrajectoryGroup],
    env_group_builders: Sequence[EnvGroupBuilder],
    training_client: tinker.TrainingClient,
    teacher_client: tinker.SamplingClient,
    tokenizer: Tokenizer,
    config: Config,
) -> dict[str, float | int]:
    """Run one SDPO training iteration.

    1. **Identify successes** and build per-trajectory teacher prompts.
    2. **Assemble datums** using the standard RL pipeline.
    3. **Distill**: either top-K cross_entropy or importance_sampling.
    4. **Train** via ``train_step``.
    """
    # Step 1: Build teacher prompts and filter to groups with teacher signal.
    filtered_groups, teacher_prompts_P_G, sdpo_metrics = _build_teacher_prompts_and_filter(
        trajectory_groups, env_group_builders, tokenizer, config,
    )

    if not filtered_groups:
        logger.warning(
            "No teacher signal in batch (no successes or feedback) — skipping SDPO update"
        )
        sdpo_metrics["sdpo/loss"] = 0.0
        return sdpo_metrics

    # Step 2: Assemble datums via the standard RL pipeline.
    # Advantages start as 0 (placeholder — overwritten by distillation step).
    advantages_P = compute_advantages(filtered_groups)
    data_D, metadata_D = assemble_training_data(filtered_groups, advantages_P)

    if not data_D:
        logger.warning("No datums after assembly — skipping SDPO update")
        sdpo_metrics["sdpo/loss"] = 0.0
        return sdpo_metrics

    # Step 3: Flatten teacher prompts to per-datum and filter.
    teacher_prompts_flat, remapped_metadata, kept_indices = _flatten_teacher_prompts(
        data_D, metadata_D, teacher_prompts_P_G,
    )

    filtered_data_D = [data_D[i] for i in kept_indices]
    if not filtered_data_D:
        logger.warning("No datums with teacher signal — skipping SDPO update")
        sdpo_metrics["sdpo/loss"] = 0.0
        return sdpo_metrics

    # Step 4: Compute IS advantages (needed for both combined and IS-only modes).
    is_metrics = await compute_sdft_advantages(
        filtered_data_D,
        remapped_metadata,
        teacher_client,
        teacher_prompts_flat,
        max_context_length=config.max_context_length,
    )
    sdpo_metrics.update(is_metrics)

    # Step 5: Distill and train.
    adam_params = tinker.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    if config.topk > 0:
        # Combined loss: 0.5 * CE (forward KL) + 0.5 * IS (reverse KL).
        # We issue two forward_backward calls and one optim_step; gradients
        # accumulate across the two calls. The 0.5 weighting is applied by
        # scaling the CE weights and IS advantages by 0.5.

        # Build top-K CE datums (forward KL term).
        topk_datums, topk_metrics = await build_topk_distillation_datums(
            filtered_data_D,
            remapped_metadata,
            teacher_client,
            teacher_prompts_flat,
            topk=config.topk,
            max_context_length=config.max_context_length,
            vocab_size=len(tokenizer),
        )
        sdpo_metrics.update(topk_metrics)

        # Scale CE weights by 0.5.
        for datum in topk_datums:
            w = datum.loss_fn_inputs["weights"].to_torch()
            datum.loss_fn_inputs["weights"] = tinker.TensorData.from_torch(w * 0.5)

        # Scale IS advantages by 0.5.
        is_datums = _scale_advantages(filtered_data_D, 0.5)

        # Two forward_backward calls, one optim_step — gradients accumulate.
        ce_future = await training_client.forward_backward_async(
            topk_datums, loss_fn="cross_entropy"
        )
        is_future = await training_client.forward_backward_async(
            is_datums, loss_fn="importance_sampling"
        )
        optim_future = await training_client.optim_step_async(adam_params)

        ce_result = await ce_future.result_async()
        is_result = await is_future.result_async()
        optim_result = await optim_future.result_async()

        if ce_result.metrics:
            sdpo_metrics.update({f"ce/{k}": v for k, v in ce_result.metrics.items()})
        if is_result.metrics:
            sdpo_metrics.update({f"is/{k}": v for k, v in is_result.metrics.items()})
        if optim_result.metrics:
            sdpo_metrics.update(optim_result.metrics)
    else:
        # IS-only mode (topk=0): single forward_backward + optim_step.
        is_future = await training_client.forward_backward_async(
            filtered_data_D, loss_fn="importance_sampling"
        )
        optim_future = await training_client.optim_step_async(adam_params)

        is_result = await is_future.result_async()
        optim_result = await optim_future.result_async()

        if is_result.metrics:
            sdpo_metrics.update(is_result.metrics)
        if optim_result.metrics:
            sdpo_metrics.update(optim_result.metrics)

    return sdpo_metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


async def main(config: Config):
    """Main SDPO training loop.

    Each iteration: (1) evaluate, (2) sample rollouts from the current policy,
    (3) run ``sdpo_training_iteration`` to distill from successful solutions,
    (4) refresh the sampling client with updated weights.

    The teacher model is frozen at initialization and never updated — this is
    the theta_ref from the paper (Table 4).
    """
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    # ---- Create clients ----

    service_client = tinker.ServiceClient(base_url=config.base_url)

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)

    if resume_info:
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
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, config.load_checkpoint_path, config.renderer_name
        )
        training_client = await service_client.create_training_client_from_state_async(
            config.load_checkpoint_path, user_metadata=user_metadata
        )
        logger.info(f"Loaded weights from {config.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            config.model_name,
            rank=config.lora_rank,
            user_metadata=user_metadata,
        )

    # Frozen teacher model (theta_ref).
    teacher_client = await training_client.save_weights_and_get_sampling_client_async("reference")

    tokenizer = training_client.get_tokenizer()

    # ---- Dataset ----

    dataset, maybe_test_dataset = await config.dataset_builder()
    num_batches = len(dataset)
    logger.info(f"Will train for {num_batches} iterations")

    start_batch = resume_info.batch if resume_info else 0

    # Evaluators.
    from tinker_cookbook.rl.metric_util import RLTestSetEvaluator

    evaluators: list[SamplingClientEvaluator] = [e() for e in config.evaluator_builders]
    if maybe_test_dataset is not None:
        evaluators.append(RLTestSetEvaluator(maybe_test_dataset, max_tokens=config.max_tokens))

    # Initial sampling client for rollouts.
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    # ---- Training loop ----

    for i_batch in range(start_batch, num_batches):
        metrics: dict[str, float | int | str] = {
            "progress/batch": i_batch,
            "progress/done_frac": (i_batch + 1) / num_batches,
            "optim/lr": config.learning_rate,
        }
        t_start = time.time()

        # ---- Evaluation ----
        if evaluators and config.eval_every > 0 and i_batch % config.eval_every == 0:
            with timed("eval", metrics):
                for evaluator in evaluators:
                    ev_name = getattr(evaluator, "name", "test")
                    eval_file_prefix = f"eval_{ev_name}_iteration_{i_batch:06d}"
                    with _get_logtree_scope(
                        log_path=config.log_path,
                        num_groups_to_log=config.num_groups_to_log,
                        f_name=eval_file_prefix,
                        scope_name=f"Running evaluation {ev_name} {i_batch}",
                    ):
                        rollout_summary_export = (
                            RolloutSummaryExportConfig(
                                path=rollout_summaries_jsonl_path(
                                    config.log_path, eval_file_prefix
                                ),
                                split=f"eval/{ev_name}",
                                iteration=i_batch,
                                sampling_client_step=i_batch,
                            )
                            if config.rollout_json_export
                            else None
                        )
                        eval_metrics = await evaluator(
                            sampling_client,
                            rollout_summary_export=rollout_summary_export,
                        )
                    metrics.update(eval_metrics)

        # ---- Checkpoint ----
        if config.save_every > 0 and i_batch > start_batch and i_batch % config.save_every == 0:
            with timed("checkpoint", metrics):
                await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=f"{i_batch:06d}",
                    log_path=config.log_path,
                    kind="both",
                    loop_state={"batch": i_batch},
                    ttl_seconds=config.ttl_seconds,
                )

        # ---- Rollouts ----
        env_group_builders = dataset.get_batch(i_batch)
        policy = TinkerTokenCompleter(
            sampling_client,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            max_context_length=config.max_context_length,
        )

        train_file_prefix = f"train_iteration_{i_batch:06d}"
        with _get_logtree_scope(
            config.log_path,
            config.num_groups_to_log,
            train_file_prefix,
            f"SDPO Iteration {i_batch}",
        ):
            with timed("rollout", metrics):
                trajectory_groups: list[TrajectoryGroup] = await gather_with_progress(
                    [do_group_rollout(builder, policy) for builder in env_group_builders],
                    desc=f"Rollouts batch {i_batch}",
                )

        # ---- Training rollout metrics (reward, accuracy, etc.) ----
        taglist_P = [b.logging_tags() for b in env_group_builders]
        metrics.update(compute_trajectory_metrics(trajectory_groups, taglist_P))

        # Export train rollout summaries
        if config.rollout_json_export:
            write_rollout_summaries_jsonl(
                rollout_summaries_jsonl_path(config.log_path, train_file_prefix),
                split="train",
                iteration=i_batch,
                trajectory_groups_P=trajectory_groups,
                taglist_P=taglist_P,
            )

        # ---- SDPO update ----
        with timed("sdpo_step", metrics):
            sdpo_metrics = await sdpo_training_iteration(
                trajectory_groups=trajectory_groups,
                env_group_builders=env_group_builders,
                training_client=training_client,
                teacher_client=teacher_client,
                tokenizer=tokenizer,
                config=config,
            )
        metrics.update(sdpo_metrics)

        # Refresh sampling client with updated policy weights.
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)

    # ---- Final checkpoint ----
    if start_batch < num_batches:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"batch": num_batches},
            ttl_seconds=config.ttl_seconds,
        )
    else:
        logger.info("Training was already complete; nothing to do")

    ml_logger.close()
    logger.info("SDPO training completed successfully")
