"""
Core SDPO training logic.

Self-Distilled Policy Optimization (SDPO) augments on-policy RL by distilling
from the model's own successful trajectories. For each problem group, successful
solutions are used to construct "teacher prompts" that condition a frozen
reference model on the solution.

The SDPO gradient (Proposition 2.1) is a policy gradient with per-token
advantages equal to the log-ratio of teacher to student probabilities:

    nabla L = E[ sum_t (teacher_lp - student_lp) * nabla log pi_student ]

This maps directly to tinker's ``importance_sampling`` loss, where we set
``advantages = teacher_lp - student_lp``. The importance weight automatically
corrects for off-policy drift. This avoids the 1.5-3x overhead of
``forward_backward_custom``.

Reference: https://arxiv.org/abs/2601.20802
"""

from __future__ import annotations

import logging
import os
import time
from typing import Sequence, cast

import chz
import tinker
import torch

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.train import gather_with_progress
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDatasetBuilder, TrajectoryGroup
from tinker_cookbook.sdpo.data import (
    build_sdpo_datum,
    extract_response_logprobs,
    extract_response_tokens,
)
from tinker_cookbook.sdpo.teacher import (
    build_teacher_prompt,
    compute_teacher_logprobs,
    strip_thinking_blocks,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed

logger = logging.getLogger(__name__)


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

    1. For each group with a successful trajectory, build teacher prompts.
    2. Compute teacher logprobs from the frozen reference model.
    3. Build datums with advantages = teacher_lp - student_lp.
    4. Train with ``forward_backward(..., loss_fn="importance_sampling")``.
    """
    datums: list[tinker.Datum] = []
    teacher_logprob_coros: list = []
    # Store (datum_builder_args) to construct datums after teacher logprobs arrive.
    pending: list[tuple[tinker.ModelInput, list[int], list[float]]] = []

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
        if not successful_indices:
            continue

        n_groups_with_success += 1

        assert isinstance(builder, ProblemGroupBuilder)
        env = cast(ProblemEnv, builder.env_thunk())

        for traj_idx, traj in enumerate(group.trajectories_G):
            response_tokens = extract_response_tokens(traj)
            sampled_logprobs = extract_response_logprobs(traj)
            if not response_tokens:
                continue

            # Select solution (respecting dont_reprompt_on_self_success).
            # Prefer a different rollout's success as teacher; fall back to
            # self if this trajectory is the only success in the group.
            if config.dont_reprompt_on_self_success:
                other_successes = [i for i in successful_indices if i != traj_idx]
                solution_idx = other_successes[0] if other_successes else successful_indices[0]
            else:
                solution_idx = successful_indices[0]

            solution_tokens = extract_response_tokens(group.trajectories_G[solution_idx])
            solution_text = tokenizer.decode(solution_tokens)
            if config.remove_thinking_from_demonstration:
                solution_text = strip_thinking_blocks(solution_text)

            teacher_ob = build_teacher_prompt(env, solution_text, config.reprompt_suffix)

            student_ob = traj.transitions[0].ob
            pending.append((student_ob, response_tokens, sampled_logprobs))
            teacher_logprob_coros.append(
                compute_teacher_logprobs(reference_client, teacher_ob, response_tokens)
            )
            n_sdpo_trajectories += 1

    n_groups = len(trajectory_groups)

    if not pending:
        logger.warning("No successful solutions in batch — skipping SDPO update")
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
    }
    if fwd_bwd_result.metrics:
        metrics.update(fwd_bwd_result.metrics)
    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


async def main(config: Config):
    """SDPO training loop: rollout -> identify successes -> distill -> repeat."""
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
                    eval_metrics = await evaluator(sampling_client)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

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
        )

        with timed("rollout", metrics):
            trajectory_groups: list[TrajectoryGroup] = await gather_with_progress(
                [do_group_rollout(builder, policy) for builder in env_group_builders],
                desc=f"Rollouts batch {i_batch}",
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
