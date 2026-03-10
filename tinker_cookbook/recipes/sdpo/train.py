"""
Self-Distilled Policy Optimization (SDPO) recipe for math RL.

SDPO augments on-policy RL by distilling from the model's own successful
trajectories. For each problem group, successful solutions are used to
construct "teacher prompts" that condition a frozen reference model on the
solution. The training loss minimizes the token-level KL divergence between
the student policy and the solution-conditioned teacher.

From Proposition 2.1 of the paper, the SDPO gradient decomposes as a
policy gradient whose per-token advantages are the log-ratio of teacher
to student probabilities:

    nabla L = E[ sum_t (log pi_student - log pi_teacher) * nabla log pi_student ]

This gives dense, token-level credit assignment rather than the single
scalar reward used by GRPO.

Teacher regularization: we use a frozen reference model (theta_ref) as the
teacher base. Table 4 in the paper shows this gives 48.8 accuracy vs 36.1
for unregularized, and is simpler than EMA.

Reference: https://arxiv.org/abs/2601.20802
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Sequence, cast

import chz
import tinker
import torch
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, cli_utils, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.math_rl.math_env import MathEnv, get_math_dataset_builder
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import EnvGroupBuilder, Trajectory, TrajectoryGroup
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI Configuration
# ---------------------------------------------------------------------------


@chz.chz
class CLIConfig:
    """Configuration for SDPO training on math problems."""

    # Model
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment
    env: str = "math"  # Options: math, gsm8k, polaris, deepmath
    seed: int = 0

    # Training
    group_size: int = 8
    groups_per_batch: int = 64
    learning_rate: float = 1e-5
    max_tokens: int = 2048
    temperature: float = 1.0

    # SDPO-specific
    success_reward_threshold: float = 0.5
    reprompt_template: str = (
        "The above is a correct solution to the problem. Now solve the problem again."
    )

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Checkpointing / eval
    eval_every: int = 10
    save_every: int = 10

    # Service
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def extract_response_tokens(traj: Trajectory) -> list[int]:
    """Extract all action tokens from a single-turn trajectory."""
    tokens: list[int] = []
    for transition in traj.transitions:
        tokens.extend(transition.ac.tokens)
    return tokens


def build_full_sequence(
    ob: tinker.ModelInput, response_tokens: list[int]
) -> tinker.ModelInput:
    """Append response tokens to an observation ModelInput."""
    chunks = list(ob.chunks) + [tinker.EncodedTextChunk(tokens=response_tokens)]
    return tinker.ModelInput(chunks=chunks)


def build_student_datum(
    ob: tinker.ModelInput, response_tokens: list[int]
) -> tinker.Datum:
    """Build a training datum from (prompt, response) with weights on response tokens."""
    full_seq = build_full_sequence(ob, response_tokens)
    input_mi, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        list(full_seq.chunks)
    )

    prompt_len = ob.length
    # Weights: 0 for prompt positions, 1 for response positions.
    # target_tokens has length (prompt_len + response_len - 1); response
    # targets start at index (prompt_len - 1).
    weights = [0.0] * (prompt_len - 1) + [1.0] * len(response_tokens)
    weights = weights[: len(target_tokens)]

    return tinker.Datum(
        model_input=input_mi,
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=weights, dtype="float32", shape=[len(weights)]
            ),
            "target_tokens": tinker.TensorData(
                data=target_tokens, dtype="int64", shape=[len(target_tokens)]
            ),
        },
    )


def build_teacher_prompt(
    env: MathEnv,
    solution_text: str,
    reprompt_template: str,
) -> tinker.ModelInput:
    """Build a teacher prompt: original question + successful solution + re-prompt."""
    teacher_convo: list[renderers.Message] = env.convo_prefix + [
        {"role": "user", "content": env.get_question()},
        {"role": "assistant", "content": solution_text},
        {"role": "user", "content": reprompt_template},
    ]
    return env.renderer.build_generation_prompt(teacher_convo)


async def compute_teacher_logprobs(
    reference_client: tinker.SamplingClient,
    teacher_ob: tinker.ModelInput,
    response_tokens: list[int],
) -> torch.Tensor:
    """Compute reference model logprobs for response tokens under the teacher prompt.

    Returns a tensor of length len(response_tokens) containing the log probability
    of each response token conditioned on the teacher prompt and preceding response
    tokens.
    """
    teacher_full = build_full_sequence(teacher_ob, response_tokens)
    all_logprobs = await reference_client.compute_logprobs_async(teacher_full)
    # Extract only the logprobs for response token positions.
    # compute_logprobs[i] = log P(token_i | tokens_0..i-1)
    teacher_prompt_len = teacher_ob.length
    response_logprobs = all_logprobs[teacher_prompt_len:]
    return torch.tensor(response_logprobs, dtype=torch.float32)


# ---------------------------------------------------------------------------
# SDPO loss
# ---------------------------------------------------------------------------


def compute_sdpo_loss(
    data: list[tinker.Datum],
    student_logprobs_list: list[torch.Tensor],
    teacher_logprobs_list: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the SDPO token-level reverse-KL loss.

    From Proposition 2.1, the gradient of the SDPO objective is:

        nabla L = E[ sum_t (log pi_student - log pi_teacher) * nabla log pi_student ]

    We implement this as:

        loss = mean_t [ (student_lp - teacher_lp).detach() * student_lp ]

    The .detach() on the log-ratio makes it act as a per-token advantage that
    doesn't receive gradients, so only the student log-probs are differentiated.
    """
    losses: list[torch.Tensor] = []
    log_ratio_sum = 0.0
    token_count = 0

    for datum, student_lps, teacher_response_lps in zip(
        data, student_logprobs_list, teacher_logprobs_list, strict=True
    ):
        weights = torch.tensor(
            datum.loss_fn_inputs["weights"].data, dtype=torch.float32
        )

        # Response tokens start where weights become 1.
        response_start = int((weights == 0).sum().item())
        response_len = int(weights.sum().item())

        student_response = student_lps[response_start : response_start + response_len]

        # Align lengths in case of minor truncation differences.
        min_len = min(len(student_response), len(teacher_response_lps))
        if min_len == 0:
            continue

        s = student_response[:min_len].float()
        t = teacher_response_lps[:min_len].float()

        # Per-token advantage: how much the student overestimates vs teacher.
        log_ratio = (s - t).detach()
        per_token_loss = log_ratio * s
        losses.append(per_token_loss.mean())

        log_ratio_sum += log_ratio.sum().item()
        token_count += min_len

    if not losses:
        zero = torch.tensor(0.0, requires_grad=True)
        return zero, {"sdpo/loss": 0.0, "sdpo/mean_log_ratio": 0.0}

    loss = torch.stack(losses).mean()
    metrics = {
        "sdpo/loss": loss.item(),
        "sdpo/mean_log_ratio": log_ratio_sum / max(token_count, 1),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------


async def gather_with_progress(
    coros: list, desc: str = "Sampling"
) -> list:
    """Run coroutines concurrently with a tqdm progress bar."""
    pbar = tqdm(total=len(coros), desc=desc)

    async def track(coro):
        result = await coro
        pbar.update(1)
        return result

    try:
        return await asyncio.gather(*[track(c) for c in coros])
    finally:
        pbar.close()


# ---------------------------------------------------------------------------
# SDPO training iteration
# ---------------------------------------------------------------------------


async def sdpo_training_iteration(
    trajectory_groups: list[TrajectoryGroup],
    env_group_builders: Sequence[EnvGroupBuilder],
    training_client: tinker.TrainingClient,
    reference_client: tinker.SamplingClient,
    tokenizer: Tokenizer,
    learning_rate: float,
    success_threshold: float,
    reprompt_template: str,
) -> dict[str, float | int]:
    """Run one SDPO training iteration: collect teacher logprobs, then update.

    For each trajectory group:
      1. Find a successful trajectory (reward >= threshold).
      2. Build a "teacher prompt" = original question + successful solution + re-prompt.
      3. Compute reference-model logprobs for every trajectory under that teacher prompt.
      4. Build student datums and train with the SDPO loss.
    """
    student_datums: list[tinker.Datum] = []
    teacher_logprob_coros: list = []

    n_groups_with_success = 0
    n_total_trajectories = 0
    n_sdpo_trajectories = 0
    total_success_rate = 0.0

    for group, builder in zip(trajectory_groups, env_group_builders, strict=True):
        rewards = group.get_total_rewards()
        n_total_trajectories += len(rewards)
        n_successes = sum(1 for r in rewards if r >= success_threshold)
        total_success_rate += n_successes / len(rewards)

        # Find first successful trajectory in this group.
        successful_idx: int | None = None
        for i, reward in enumerate(rewards):
            if reward >= success_threshold:
                successful_idx = i
                break

        if successful_idx is None:
            continue  # No successful solution — skip this group.

        n_groups_with_success += 1

        # Decode the successful solution to text for the teacher prompt.
        solution_tokens = extract_response_tokens(group.trajectories_G[successful_idx])
        solution_text = tokenizer.decode(solution_tokens)

        # Recreate an env from the builder to access question/renderer/prefix.
        assert isinstance(builder, ProblemGroupBuilder)
        env = cast(MathEnv, builder.env_thunk())
        teacher_ob = build_teacher_prompt(env, solution_text, reprompt_template)

        # For every trajectory in the group, create a student datum and
        # schedule a teacher logprob computation.
        for traj in group.trajectories_G:
            response_tokens = extract_response_tokens(traj)
            if not response_tokens:
                continue

            student_ob = traj.transitions[0].ob
            student_datums.append(build_student_datum(student_ob, response_tokens))

            teacher_logprob_coros.append(
                compute_teacher_logprobs(reference_client, teacher_ob, response_tokens)
            )
            n_sdpo_trajectories += 1

    n_groups = len(trajectory_groups)

    if not student_datums:
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

    # ---- Custom loss closure (captures teacher_logprobs_list) ----

    captured_teacher = teacher_logprobs_list

    def sdpo_loss_fn(
        data: list[tinker.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return compute_sdpo_loss(data, logprobs_list, captured_teacher)

    # ---- Forward-backward + optimizer step ----

    backward_result = training_client.forward_backward_custom(
        student_datums, sdpo_loss_fn
    ).result()

    adam_params = tinker.AdamParams(
        learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )
    training_client.optim_step(adam_params).result()

    metrics: dict[str, float | int] = {
        **backward_result.metrics,
        "sdpo/groups_with_success": n_groups_with_success,
        "sdpo/groups_total": n_groups,
        "sdpo/success_fraction": n_groups_with_success / n_groups if n_groups else 0.0,
        "sdpo/trajectories_trained": n_sdpo_trajectories,
        "sdpo/trajectories_total": n_total_trajectories,
        "sdpo/mean_group_success_rate": total_success_rate / n_groups if n_groups else 0.0,
    }
    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


async def main(cli_config: CLIConfig):
    """SDPO training loop: rollout -> identify successes -> distill -> repeat."""

    # Resolve renderer.
    renderer_name = (
        await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
            model_name=cli_config.model_name,
            explicit_renderer_name=cli_config.renderer_name,
            load_checkpoint_path=cli_config.load_checkpoint_path,
            base_url=cli_config.base_url,
        )
    )

    # Log paths.
    model_slug = cli_config.model_name.replace("/", "-")
    run_name = (
        f"sdpo-{cli_config.env}-{model_slug}-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-seed{cli_config.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cli_config.log_path or f"/tmp/tinker-examples/sdpo/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    cli_utils.check_log_dir(
        log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists
    )

    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        config=cli_config,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    # ---- Create clients ----

    service_client = tinker.ServiceClient(base_url=cli_config.base_url)

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, renderer_name)

    resume_info = checkpoint_utils.get_last_checkpoint(log_path)

    if resume_info:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, resume_info["state_path"], renderer_name
        )
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info["state_path"], user_metadata=user_metadata
            )
        )
        logger.info(f"Resumed training from {resume_info['state_path']}")
    elif cli_config.load_checkpoint_path:
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, cli_config.load_checkpoint_path, renderer_name
        )
        training_client = await service_client.create_training_client_from_state_async(
            cli_config.load_checkpoint_path, user_metadata=user_metadata
        )
        logger.info(f"Loaded weights from {cli_config.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(
            cli_config.model_name,
            rank=cli_config.lora_rank,
            user_metadata=user_metadata,
        )

    # Frozen reference model for teacher logprobs (theta_ref).
    reference_client = await training_client.save_weights_and_get_sampling_client_async(
        "reference"
    )

    tokenizer = training_client.get_tokenizer()

    # ---- Dataset ----

    dataset_builder = get_math_dataset_builder(
        dataset_name=cli_config.env,
        batch_size=cli_config.groups_per_batch,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        seed=cli_config.seed,
    )
    dataset, test_dataset = await dataset_builder()
    num_batches = len(dataset)
    logger.info(f"Will train for {num_batches} iterations")

    start_batch = resume_info["batch"] if resume_info else 0

    # Test-set evaluator.
    from tinker_cookbook.rl.metric_util import RLTestSetEvaluator

    test_evaluator = (
        RLTestSetEvaluator(test_dataset, max_tokens=cli_config.max_tokens)
        if test_dataset is not None
        else None
    )

    # Initial sampling client for rollouts (tracks current policy).
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    # ---- Training loop ----

    for i_batch in range(start_batch, num_batches):
        metrics: dict[str, float | int | str] = {
            "progress/batch": i_batch,
            "progress/done_frac": (i_batch + 1) / num_batches,
            "optim/lr": cli_config.learning_rate,
        }
        t_start = time.time()

        # ---- Evaluation ----
        if (
            test_evaluator is not None
            and cli_config.eval_every > 0
            and i_batch % cli_config.eval_every == 0
        ):
            with timed("eval", metrics):
                eval_metrics = await test_evaluator(sampling_client)
                metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # ---- Checkpoint ----
        if (
            cli_config.save_every > 0
            and i_batch > start_batch
            and i_batch % cli_config.save_every == 0
        ):
            with timed("checkpoint", metrics):
                await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=f"{i_batch:06d}",
                    log_path=log_path,
                    kind="both",
                    loop_state={"batch": i_batch},
                )

        # ---- Rollouts ----
        env_group_builders = dataset.get_batch(i_batch)
        policy = TinkerTokenCompleter(
            sampling_client,
            max_tokens=cli_config.max_tokens,
            temperature=cli_config.temperature,
        )

        with timed("rollout", metrics):
            trajectory_groups: list[TrajectoryGroup] = await gather_with_progress(
                [
                    do_group_rollout(builder, policy)
                    for builder in env_group_builders
                ],
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
                learning_rate=cli_config.learning_rate,
                success_threshold=cli_config.success_reward_threshold,
                reprompt_template=cli_config.reprompt_template,
            )
        metrics.update(sdpo_metrics)

        # Refresh sampling client with updated policy weights.
        sampling_client = (
            await training_client.save_weights_and_get_sampling_client_async()
        )

        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)

    # ---- Final checkpoint ----
    if start_batch < num_batches:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    ml_logger.close()
    logger.info("SDPO training completed successfully")


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(main(cli_config))
