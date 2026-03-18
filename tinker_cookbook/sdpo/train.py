"""
Self-Distilled Policy Optimization (SDPO) — core training logic.

SDPO (https://arxiv.org/abs/2601.20802) is an on-policy RL algorithm that
provides dense, token-level credit assignment by distilling from the model's
own successful trajectories.

Standard RL (e.g. GRPO) assigns a single scalar reward per sequence — every
token in a correct solution gets the same credit. SDPO instead constructs a
"teacher" by conditioning a reference model on a successful solution, then
computes per-token advantages as:

    advantage_t = log pi_teacher(y_t) - log pi_student(y_t)

Tokens where the teacher is more confident than the student get positive
advantage (reinforced), and vice versa. From Proposition 2.1, this is
mathematically a policy gradient — so it maps directly to tinker's
``importance_sampling`` loss. We encode the SDPO signal as advantages in
the datum and use ``forward_backward(..., loss_fn="importance_sampling")``,
which is faster than ``forward_backward_custom`` and also provides free
off-policy correction via the importance weight.
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
import torch

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
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
    build_sdpo_datum,
    extract_feedback,
    extract_response_logprobs,
    extract_response_tokens,
)
from tinker_cookbook.sdpo.teacher import (
    build_teacher_prompt,
    build_teacher_prompt_from_messages,
    compute_teacher_logprobs,
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
    # Include environment feedback (e.g. compiler errors) in the teacher prompt.
    # Useful for code tasks. The paper (Table 6) shows feedback and solutions
    # are complementary: solution alone 42.6%, feedback alone 39.9%, both 48.3%.
    include_environment_feedback: bool = False
    # Maximum context length for the model. The teacher prompt is longer than
    # the student prompt (it includes a solution and/or feedback), so the
    # response tokens are truncated to fit within this limit when computing
    # teacher logprobs. Positions beyond the truncation get advantage=0.
    max_context_length: int = 32768

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
# SDPO training iteration
# ---------------------------------------------------------------------------


async def sdpo_training_iteration(
    trajectory_groups: list[TrajectoryGroup],
    env_group_builders: Sequence[EnvGroupBuilder],
    training_client: tinker.TrainingClient,
    reference_client: tinker.SamplingClient,
    tokenizer: Tokenizer,
    config: Config,
) -> dict[str, float | int]:
    """Run one SDPO training iteration.

    For each batch of trajectory groups:

    1. **Identify successes**: Find groups where at least one rollout solved
       the problem. Groups with no teacher signal (no successes and no
       environment feedback) are skipped.
    2. **Build teacher prompts**: Condition the reference model on a successful
       solution and/or environment feedback (e.g. compiler errors). This gives
       the teacher an informational advantage over the student.
    3. **Compute teacher logprobs**: The frozen reference model scores each
       trajectory's response tokens under the teacher prompt. This tells us
       "how likely is each token given the extra information?"
    4. **Build datums**: Set per-token advantages = teacher_lp - student_lp.
       Positive advantage means the teacher is more confident → reinforce.
       Near-zero means the student already matches → no gradient.
    5. **Train**: ``forward_backward(datums, loss_fn="importance_sampling")``
       followed by ``optim_step()``.
    """
    datums: list[tinker.Datum] = []
    teacher_logprob_coros: list = []
    # Store (datum_builder_args) to construct datums after teacher logprobs arrive.
    pending: list[tuple[tinker.ModelInput, list[int], list[float]]] = []

    n_groups_with_success = 0
    n_total_trajectories = 0
    n_sdpo_trajectories = 0
    n_teacher_truncated = 0
    total_tokens_truncated = 0
    total_success_rate = 0.0

    for group, builder in zip(trajectory_groups, env_group_builders, strict=True):
        rewards = group.get_total_rewards()
        n_total_trajectories += len(rewards)
        n_successes = sum(1 for r in rewards if r >= config.success_reward_threshold)
        total_success_rate += n_successes / len(rewards)

        successful_indices = [
            i for i, r in enumerate(rewards) if r >= config.success_reward_threshold
        ]

        # Skip groups with no teacher signal: need either a successful
        # solution or environment feedback (when enabled).
        has_solutions = len(successful_indices) > 0
        if not has_solutions and not config.include_environment_feedback:
            continue

        n_groups_with_success += 1 if has_solutions else 0

        # Resolve how to build the teacher prompt based on builder type.
        teacher_prompt_builder = _resolve_teacher_prompt_builder(builder)

        for traj_idx, traj in enumerate(group.trajectories_G):
            response_tokens = extract_response_tokens(traj)
            sampled_logprobs = extract_response_logprobs(traj)
            if not response_tokens:
                continue

            # --- Gather teacher conditioning: solution and/or feedback ---

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

            # Feedback: extract from this trajectory's own execution logs
            # (e.g. compiler errors, failing test cases).
            feedback_text: str | None = None
            if config.include_environment_feedback:
                feedback_text = extract_feedback(traj)

            # Need at least one conditioning signal for the teacher.
            if solution_text is None and feedback_text is None:
                continue

            teacher_ob = teacher_prompt_builder(
                config.reprompt_suffix,
                solution_text=solution_text,
                feedback_text=feedback_text,
            )

            student_ob = traj.transitions[0].ob
            pending.append((student_ob, response_tokens, sampled_logprobs))
            teacher_logprob_coros.append(
                compute_teacher_logprobs(
                    reference_client,
                    teacher_ob,
                    response_tokens,
                    max_context_length=config.max_context_length,
                )
            )
            # Track context window truncation.
            total_teacher_len = teacher_ob.length + len(response_tokens)
            if total_teacher_len > config.max_context_length:
                n_teacher_truncated += 1
                total_tokens_truncated += total_teacher_len - config.max_context_length
            n_sdpo_trajectories += 1

    n_groups = len(trajectory_groups)

    if not pending:
        logger.warning(
            "No teacher signal in batch (no successes or feedback) — skipping SDPO update"
        )
        return {
            "sdpo/loss": 0.0,
            "sdpo/groups_with_success": 0,
            "sdpo/groups_total": n_groups,
        }

    # Compute all teacher logprobs in parallel.
    teacher_logprobs_list: list[torch.Tensor] = await gather_with_progress(
        teacher_logprob_coros, desc="Teacher logprobs"
    )

    # Build datums with SDPO advantages.
    mean_advantage = 0.0
    for (student_ob, response_tokens, sampled_lps), teacher_lps in zip(
        pending, teacher_logprobs_list, strict=True
    ):
        datums.append(build_sdpo_datum(student_ob, response_tokens, sampled_lps, teacher_lps))
        # Track mean advantage for logging.
        min_len = min(len(sampled_lps), len(teacher_lps))
        if min_len > 0:
            mean_advantage += (
                sum(teacher_lps[k].item() - sampled_lps[k] for k in range(min_len)) / min_len
            )

    mean_advantage /= max(len(datums), 1)

    # ---- Forward-backward with importance_sampling + optimizer step ----

    fwd_bwd_future = await training_client.forward_backward_async(
        datums, loss_fn="importance_sampling"
    )
    adam_params = tinker.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )
    optim_future = await training_client.optim_step_async(adam_params)

    fwd_bwd_result = await fwd_bwd_future.result_async()
    await optim_future.result_async()

    metrics: dict[str, float | int] = {
        "sdpo/groups_with_success": n_groups_with_success,
        "sdpo/groups_total": n_groups,
        "sdpo/success_fraction": n_groups_with_success / n_groups if n_groups else 0.0,
        "sdpo/trajectories_trained": n_sdpo_trajectories,
        "sdpo/trajectories_total": n_total_trajectories,
        "sdpo/mean_group_success_rate": total_success_rate / n_groups if n_groups else 0.0,
        "sdpo/mean_advantage": mean_advantage,
        "sdpo/teacher_truncated_count": n_teacher_truncated,
        "sdpo/teacher_truncated_frac": n_teacher_truncated / n_sdpo_trajectories
        if n_sdpo_trajectories
        else 0.0,
        "sdpo/teacher_truncated_tokens_avg": total_tokens_truncated / n_teacher_truncated
        if n_teacher_truncated
        else 0.0,
    }
    if fwd_bwd_result.metrics:
        metrics.update(fwd_bwd_result.metrics)
    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


async def main(config: Config):
    """Main SDPO training loop.

    Each iteration: (1) evaluate, (2) sample rollouts from the current policy,
    (3) run ``sdpo_training_iteration`` to compute teacher logprobs and update,
    (4) refresh the sampling client with updated weights.

    The reference model (``reference_client``) is frozen at initialization and
    never updated — this is the theta_ref teacher from the paper (Table 4).
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
            service_client, resume_info["state_path"], config.renderer_name
        )
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"], user_metadata=user_metadata
            )
        )
        logger.info(f"Resumed training from {resume_info['state_path']}")
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

    # Frozen reference model for teacher logprobs (theta_ref).
    reference_client = await training_client.save_weights_and_get_sampling_client_async("reference")

    tokenizer = training_client.get_tokenizer()

    # ---- Dataset ----

    dataset, maybe_test_dataset = await config.dataset_builder()
    num_batches = len(dataset)
    logger.info(f"Will train for {num_batches} iterations")

    start_batch = resume_info["batch"] if resume_info else 0

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
        # No prefix — matches GRPO's convention so metrics are directly
        # comparable in W&B (e.g. env/all/correct, env/all/reward/total).
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
                reference_client=reference_client,
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
