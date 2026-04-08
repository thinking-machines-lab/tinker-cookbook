"""
FIPO: Future-KL Influenced Policy Optimization

Reproduces the FIPO algorithm (arXiv:2603.19835) using the Tinker API.
FIPO modifies GRPO/PPO by reweighting per-token advantages with future-KL
influence weights, enabling deeper reasoning chains.

Implementation: uses forward_backward_custom with a closure that captures
RL-specific fields (sampling logprobs, advantages, mask). The closure
receives training logprobs from the forward pass and computes the full FIPO
loss (future-KL reweighted PPO) in a single forward-backward pass.

Usage:
    python -m tinker_cookbook.recipes.fipo.train \\
        model_name=Qwen/Qwen3-8B \\
        env=math \\
        group_size=8 \\
        groups_per_batch=16 \\
        max_tokens=4096 \\
        learning_rate=1e-6

Quick sanity check:
    python -m tinker_cookbook.recipes.fipo.train \\
        model_name=meta-llama/Llama-3.2-3B \\
        env=arithmetic \\
        group_size=4 \\
        groups_per_batch=4 \\
        max_tokens=32 \\
        max_steps=3
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import chz
import tinker
import torch

from tinker_cookbook import checkpoint_utils, cli_utils, model_info
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.recipes.math_rl import arithmetic_env, math_env
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
)
from tinker_cookbook.rl.fipo import make_fipo_loss_fn, prepare_custom_datums
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.metrics import compute_kl_sample_train
from tinker_cookbook.rl.rollout_logging import RolloutSummaryGroup
from tinker_cookbook.rl.rollout_strategy import rollout_strategy_from_config
from tinker_cookbook.rl.rollouts import do_group_rollout_and_filter_constant_reward
from tinker_cookbook.rl.train import (
    Config as RLConfig,
    _get_logtree_scope,
    _maybe_export_rollout_summary_jsonl,
    gather_with_progress,
    print_group,
    run_evaluations_parallel,
    save_checkpoint_and_get_sampling_client,
)
from tinker_cookbook.rl.types import RLDatasetBuilder, TrajectoryGroup
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.misc_utils import iteration_dir, safezip, split_list

logger = logging.getLogger(__name__)


@chz.chz
class FIPOConfig:
    """FIPO-specific hyperparameters."""

    # Future-KL decay half-life in tokens (τ). Paper uses 32 for 32B, code default 128.
    decay_half_life: float = 32.0
    # PPO clip range ε for policy ratio clipping
    clip_epsilon: float = 0.2
    # Dual-clip threshold c for participation mask
    dual_clip_threshold: float = 10.0
    # Influence weight clipping bounds [1 - clip_low, 1 + clip_high]
    # Paper uses [1.0, 1.2] for 32B, [0.8, 1.2] for 7B
    influence_clip_low: float = 0.8
    influence_clip_high: float = 1.2
    # Safety threshold: cap influence weights for negative-advantage high-IS-ratio tokens
    safety_threshold: float = 4.0


@chz.chz
class CLIConfig:
    """Command-line configuration for FIPO training."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    env: str = "math"  # Options: arithmetic, math, polaris, deepmath, gsm8k
    seed: int = 0

    # Training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 16
    learning_rate: float = 1e-6
    max_tokens: int = 4096
    temperature: float = 1.0

    # FIPO-specific hyperparameters
    fipo: FIPOConfig = chz.field(default_factory=FIPOConfig)

    # Number of substeps per batch (gradient accumulation)
    num_substeps: int = 1

    # Logging
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Evals and checkpointing
    eval_every: int = 10
    save_every: int = 10
    num_groups_to_log: int = 4

    # Limits
    max_steps: int | None = None

    # Service
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    remove_constant_reward_groups: bool = True


def get_dataset_builder(
    env: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
) -> RLDatasetBuilder:
    if env == "arithmetic":
        return arithmetic_env.ArithmeticDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            n_batches=100,
            include_fewshot=True,
            group_size=group_size,
        )
    elif env in ["math", "polaris", "deepmath", "gsm8k"]:
        return math_env.get_math_dataset_builder(
            dataset_name=env,
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown environment: {env}")


def _make_rl_config_for_infra(cli_config: CLIConfig, dataset_builder: RLDatasetBuilder, log_path: str) -> RLConfig:
    """Create an RLConfig for infrastructure functions that need it."""
    return RLConfig(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        max_tokens=cli_config.max_tokens,
        log_path=log_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        num_groups_to_log=cli_config.num_groups_to_log,
        rollout_json_export=True,
    )


async def fipo_train_step(
    rl_data_D: list[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    fipo_config: FIPOConfig,
    num_substeps: int = 1,
    metrics: dict[str, Any] | None = None,
) -> list[torch.Tensor]:
    """Run one FIPO training step using forward_backward_custom.

    Creates a closure-based loss function that captures RL fields (sampling
    logprobs, advantages, mask) and receives training logprobs from the
    forward pass. Computes future-KL influence weights and PPO-clipped loss.

    Args:
        rl_data_D: Original RL datums with full loss_fn_inputs.
        training_client: Tinker training client.
        learning_rate: Learning rate for Adam optimizer.
        fipo_config: FIPO hyperparameters.
        num_substeps: Number of gradient accumulation substeps.
        metrics: Dict to merge optimizer metrics into (in-place).

    Returns:
        Per-datum training logprobs from the forward pass.
    """
    if not rl_data_D:
        return []

    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)

    # Split into substep batches
    batches_rl = split_list(rl_data_D, min(num_substeps, len(rl_data_D)))
    training_logprobs_D: list[torch.Tensor] = []

    for batch_rl in batches_rl:
        # Create FIPO loss closure capturing RL fields
        loss_fn = make_fipo_loss_fn(
            rl_data_D=batch_rl,
            clip_epsilon=fipo_config.clip_epsilon,
            decay_half_life=fipo_config.decay_half_life,
            dual_clip_threshold=fipo_config.dual_clip_threshold,
            influence_clip_low=fipo_config.influence_clip_low,
            influence_clip_high=fipo_config.influence_clip_high,
            safety_threshold=fipo_config.safety_threshold,
        )

        # Create stripped datums for forward_backward_custom
        custom_datums = prepare_custom_datums(batch_rl)

        # Forward-backward with custom FIPO loss
        fwd_bwd_future = await training_client.forward_backward_custom_async(
            custom_datums,
            loss_fn,
        )
        fwd_bwd_result = await fwd_bwd_future.result_async()

        if metrics is not None:
            metrics.update(fwd_bwd_result.metrics)

        # Extract training logprobs
        for output in fwd_bwd_result.loss_fn_outputs:
            training_logprobs_D.append(output["logprobs"].to_torch())

        # Optimizer step
        optim_future = await training_client.optim_step_async(adam_params)
        optim_result = await optim_future.result_async()
        if metrics is not None and optim_result.metrics:
            metrics.update(optim_result.metrics)

    return training_logprobs_D


async def fipo_main(cli_config: CLIConfig):
    """Main FIPO training loop."""

    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    model_short = cli_config.model_name.replace("/", "-")
    run_name = (
        f"fipo-{cli_config.env}-{model_short}-{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-tau{cli_config.fipo.decay_half_life}-"
        f"seed{cli_config.seed}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    log_path = cli_config.log_path or f"/tmp/tinker-examples/fipo/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=cli_config.wandb_project,
        config=cli_config,
        wandb_name=wandb_name,
    )
    store = ml_logger.store

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    # Create service and training clients
    service_client = tinker.ServiceClient(base_url=cli_config.base_url)
    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, renderer_name)
    model_info.warn_if_renderer_not_recommended(cli_config.model_name, renderer_name)

    # Build dataset
    dataset_builder = get_dataset_builder(
        env=cli_config.env,
        batch_size=cli_config.groups_per_batch,
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        seed=cli_config.seed,
    )
    rl_config = _make_rl_config_for_infra(cli_config, dataset_builder, log_path)

    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    if resume_info:
        start_batch = resume_info.batch
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, resume_info.state_path, renderer_name
        )
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path, user_metadata=user_metadata
            )
        )
        logger.info(f"Resumed training from {resume_info.state_path}")
    elif cli_config.load_checkpoint_path:
        start_batch = 0
        await checkpoint_utils.check_renderer_name_for_checkpoint_async(
            service_client, cli_config.load_checkpoint_path, renderer_name
        )
        training_client = await service_client.create_training_client_from_state_async(
            cli_config.load_checkpoint_path, user_metadata=user_metadata
        )
        logger.info(f"Loaded weights from {cli_config.load_checkpoint_path}")
    else:
        start_batch = 0
        training_client = await service_client.create_lora_training_client_async(
            cli_config.model_name, rank=cli_config.lora_rank, user_metadata=user_metadata
        )

    tokenizer = training_client.get_tokenizer()
    dataset, maybe_test_dataset = await dataset_builder()

    strategy = rollout_strategy_from_config(False)

    evaluators: list[SamplingClientEvaluator] = []
    if maybe_test_dataset is not None:
        evaluators.append(
            RLTestSetEvaluator(
                maybe_test_dataset,
                max_tokens=cli_config.max_tokens,
                strategy=strategy,
            )
        )

    num_batches = len(dataset)
    end_batch = min(cli_config.max_steps, num_batches) if cli_config.max_steps else num_batches
    logger.info(f"FIPO training: {end_batch} batches, model={cli_config.model_name}")
    logger.info(f"FIPO config: decay_half_life={cli_config.fipo.decay_half_life}, "
                f"clip=[{cli_config.fipo.influence_clip_low}, {cli_config.fipo.influence_clip_high}], "
                f"ppo_clip={cli_config.fipo.clip_epsilon}")

    # Initial checkpoint and sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, log_path, cli_config.save_every,
        start_batch, store=store,
    )

    rolling_mgr = checkpoint_utils.RollingCheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=log_path,
        rolling_save_every=0,
        save_every=cli_config.save_every,
        rolling_ttl_seconds=7200,
        store=store,
    )

    # ---- Training loop ----
    for i_batch in range(start_batch, end_batch):
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "optim/lr": cli_config.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }

        with trace.trace_iteration(step=i_batch) as window:
            # Evaluations
            if cli_config.eval_every > 0 and i_batch % cli_config.eval_every == 0 and evaluators:
                eval_metrics = await run_evaluations_parallel(
                    evaluators, sampling_client, rl_config, i_batch, store=store,
                )
                metrics.update(eval_metrics)

            # Collect trajectories
            env_group_builders_P = dataset.get_batch(i_batch)
            iter_dir = iteration_dir(log_path, i_batch)

            async with trace.scope_span("sampling"):
                with _get_logtree_scope(
                    output_dir=iter_dir,
                    num_groups_to_log=cli_config.num_groups_to_log,
                    f_name="train",
                    scope_name=f"FIPO Iteration {i_batch}",
                    iteration=i_batch,
                    store=store,
                ):
                    results_P = await gather_with_progress(
                        (
                            do_group_rollout_and_filter_constant_reward(
                                sampling_client,
                                builder,
                                max_tokens=cli_config.max_tokens,
                                temperature=cli_config.temperature,
                                do_remove_constant_reward_groups=False,
                                enable_logging=i < cli_config.num_groups_to_log,
                                strategy=strategy,
                            )
                            for i, builder in enumerate(env_group_builders_P)
                        ),
                        desc=f"FIPO sampling batch {i_batch}",
                    )

            # Filter failed groups
            successful = [
                (builder, tg)
                for builder, tg in safezip(env_group_builders_P, results_P)
                if tg is not None
            ]
            if not successful:
                logger.warning(f"Batch {i_batch}: all groups failed, skipping")
                continue

            env_group_builders_P = [s[0] for s in successful]
            trajectory_groups_P: list[TrajectoryGroup] = [s[1] for s in successful]

            _maybe_export_rollout_summary_jsonl(
                config=rl_config,
                base_name="train",
                split="train",
                iteration=i_batch,
                groups_P=[
                    RolloutSummaryGroup(
                        trajectory_group=tg,
                        tags=builder.logging_tags(),
                        sampling_client_step=i_batch,
                    )
                    for builder, tg in safezip(env_group_builders_P, trajectory_groups_P)
                ],
                store=store,
            )

            if cli_config.remove_constant_reward_groups:
                trajectory_groups_P = remove_constant_reward_groups(trajectory_groups_P)

            # Prepare training data
            taglist_P = [b.logging_tags() for b in env_group_builders_P]
            metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))
            for tg in trajectory_groups_P[:2]:
                print_group(tg, tokenizer)

            advantages_P = compute_advantages(trajectory_groups_P)
            data_D, _metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

            # FIPO training step (forward_backward_custom with closure)
            async with trace.scope_span("fipo_train_step"):
                training_logprobs_D = await fipo_train_step(
                    rl_data_D=data_D,
                    training_client=training_client,
                    learning_rate=cli_config.learning_rate,
                    fipo_config=cli_config.fipo,
                    num_substeps=cli_config.num_substeps,
                    metrics=metrics,
                )

            # Post-step metrics and checkpoint
            if training_logprobs_D:
                kl_metrics = compute_kl_sample_train(data_D, training_logprobs_D)
                metrics.update(kl_metrics)

            sampling_client, ckpt_metrics = await save_checkpoint_and_get_sampling_client(
                training_client, i_batch + 1, log_path, cli_config.save_every,
                store=store,
            )
            metrics.update(ckpt_metrics)

        metrics.update(window.get_timing_metrics())
        window.save_timing(i_batch, store=store)
        ml_logger.log_metrics(metrics, step=i_batch)

    # Final checkpoint
    if start_batch < end_batch:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=log_path,
            kind="both",
            loop_state={"batch": end_batch},
            ttl_seconds=None,
            store=store,
        )

    await rolling_mgr.finalize_async()
    ml_logger.close()
    logger.info("FIPO training completed successfully")


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(fipo_main(cli_config))
