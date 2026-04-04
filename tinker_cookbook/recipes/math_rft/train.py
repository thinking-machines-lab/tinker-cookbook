"""
Rejection Sampling Fine-Tuning (RFT) for math reasoning.

Implements the iterative RFT / STaR (Self-Taught Reasoner) approach:
  1. Sample K solutions per problem from the current model
  2. Grade each solution using verifiable math rewards
  3. Fine-tune on correct solutions only (SFT loss)
  4. Repeat with updated model

This provides a clean comparison with GRPO (tinker_cookbook/recipes/math_rl):
both methods sample the same number of solutions per problem, but RFT trains
only on correct ones with uniform SFT loss, while GRPO trains on all solutions
weighted by advantage.

Usage:
    # MATH dataset (default)
    python -m tinker_cookbook.recipes.math_rft.train \\
        env=math \\
        groups_per_batch=32 \\
        learning_rate=1e-4

    # GSM8K (easier)
    python -m tinker_cookbook.recipes.math_rft.train \\
        env=gsm8k \\
        groups_per_batch=32

    # Debug run
    python -m tinker_cookbook.recipes.math_rft.train \\
        env=gsm8k \\
        groups_per_batch=4 group_size=4 max_steps=3
"""

import asyncio
import logging
import math
import time
from datetime import datetime
from typing import Literal

import chz
import tinker

from tinker_cookbook import checkpoint_utils, cli_utils, renderers
from tinker_cookbook.recipes.math_rft.datasets import load_gsm8k_problems, load_math_problems
from tinker_cookbook.recipes.math_rft.grading import grade_response
from tinker_cookbook.recipes.math_rl.math_env import MathEnv
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    # Model
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Dataset
    env: Literal["gsm8k", "math"] = "math"
    data_path: str | None = None  # Local data directory (e.g., ~/data)
    seed: int = 0

    # Sampling
    group_size: int = 16  # K: number of solutions sampled per problem
    groups_per_batch: int = 32  # Number of problems per batch
    max_tokens: int = 2048
    temperature: float = 1.0

    # Training
    learning_rate: float = 1e-4
    max_length: int = 3072  # Max token length for SFT datums
    max_datums_per_problem: int | None = None  # Limit SFT datums per problem (None = all correct)

    # Logging and checkpointing
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 5  # Evaluate every N batches
    save_every: int = 20
    ttl_seconds: int | None = 604800  # 7 days
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "resume"

    # Infrastructure
    base_url: str | None = None
    max_steps: int | None = None


def _get_question_suffix(env: str) -> str:
    if env == "gsm8k":
        return " Provide a numerical answer without units, written inside \\boxed{}."
    else:
        return " Write your answer in \\boxed{} format."


async def evaluate_pass_at_1(
    sampling_client: tinker.SamplingClient,
    test_data: list[dict[str, str]],
    renderer: renderers.Renderer,
    convo_prefix: list[renderers.Message],
    question_suffix: str,
    max_tokens: int,
    max_problems: int = 500,
    temperature: float = 0.0,
) -> dict[str, float]:
    """Evaluate pass@1 on the test set.

    Args:
        temperature: Sampling temperature for evaluation. Default 0.0 (greedy).
            For consistency with training, consider matching the training temperature.
    """
    problems = test_data[:max_problems]

    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )

    # Sample all problems concurrently
    async def sample_one(prob: dict[str, str]) -> tuple[tinker.SampleResponse, str]:
        convo = convo_prefix + [
            {"role": "user", "content": prob["problem"] + question_suffix}
        ]
        model_input = renderer.build_generation_prompt(convo)
        result = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        return result, prob["answer"]

    results = await asyncio.gather(*[sample_one(p) for p in problems])

    # Grade results
    n_correct = 0
    n_format = 0
    n_total = len(results)

    # Per-level tracking
    level_totals: dict[str, int] = {}
    level_correct: dict[str, int] = {}

    for i, (result, answer) in enumerate(results):
        tokens = result.sequences[0].tokens
        parsed_message, _parse_ok = renderer.parse_response(tokens)
        content = renderers.get_text_content(parsed_message)

        level = problems[i].get("level", "")
        if level:
            level_totals[level] = level_totals.get(level, 0) + 1

        try:
            _ = extract_boxed(content)
            n_format += 1
        except ValueError:
            pass

        if grade_response(content, answer):
            n_correct += 1
            if level:
                level_correct[level] = level_correct.get(level, 0) + 1

    eval_metrics: dict[str, float] = {
        "test/correct": n_correct / n_total,
        "test/format": n_format / n_total,
        "test/n_problems": float(n_total),
    }

    for level in sorted(level_totals.keys()):
        total = level_totals[level]
        correct = level_correct.get(level, 0)
        eval_metrics[f"test/correct_L{level}"] = correct / total if total else 0

    return eval_metrics


async def main(config: Config):
    logging.getLogger("httpx").setLevel(logging.WARN)

    # Resolve renderer
    renderer_name = (
        await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
            model_name=config.model_name,
            explicit_renderer_name=config.renderer_name,
            load_checkpoint_path=config.load_checkpoint_path,
            base_url=config.base_url,
        )
    )

    # Setup log path
    if config.log_path:
        log_path = config.log_path
    else:
        model_slug = config.model_name.split("/")[-1]
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        log_path = f"/tmp/tinker-examples/math_rft/{model_slug}_{config.env}_{timestamp}"

    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)

    wandb_name = config.wandb_name or log_path.split("/")[-1]
    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        wandb_name=wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    # Setup model
    tokenizer = get_tokenizer(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    logger.info(f"Model: {config.model_name}, Renderer: {renderer_name}")

    # Load data
    if config.env == "math":
        train_data, test_data = load_math_problems(
            data_path=config.data_path, seed=config.seed
        )
    elif config.env == "gsm8k":
        train_data, test_data = load_gsm8k_problems(seed=config.seed)
    else:
        raise ValueError(f"Unknown env: {config.env}")
    logger.info(f"Train: {len(train_data)} problems, Test: {len(test_data)} problems")

    question_suffix = _get_question_suffix(config.env)
    convo_prefix = MathEnv.standard_fewshot_prefix()

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    if resume_info:
        training_client = (
            await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path
            )
        )
        start_batch = resume_info.batch
        logger.info(f"Resuming from batch {start_batch}")
    elif config.load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(
            config.load_checkpoint_path
        )
        start_batch = 0
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
        )
        start_batch = 0

    # Training parameters
    adam_params = tinker.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )
    sampling_params = tinker.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stop=renderer.get_stop_sequences(),
    )

    # Compute number of batches
    n_batches = math.ceil(len(train_data) / config.groups_per_batch)
    if config.max_steps is not None:
        n_batches = min(n_batches, config.max_steps)
    logger.info(
        f"Training: {n_batches} batches, {config.groups_per_batch} problems/batch, "
        f"{config.group_size} samples/problem"
    )

    # Main training loop
    for batch_idx in range(start_batch, n_batches):
        t_start = time.time()
        metrics: dict[str, float | int | str] = {}

        # Get batch of problems
        batch_start = batch_idx * config.groups_per_batch
        batch_end = min(batch_start + config.groups_per_batch, len(train_data))
        batch_problems = train_data[batch_start:batch_end]

        if not batch_problems:
            logger.warning(f"Batch {batch_idx}: no problems, skipping")
            continue

        # Create a fresh sampling client from current weights
        sampling_client = (
            await training_client.save_weights_and_get_sampling_client_async()
        )

        # ---- Phase 1: Sample K solutions per problem concurrently ----
        async def sample_problem(
            prob: dict[str, str],
        ) -> tuple[tinker.SampleResponse, dict[str, str]]:
            convo = convo_prefix + [
                {"role": "user", "content": prob["problem"] + question_suffix}
            ]
            model_input = renderer.build_generation_prompt(convo)
            result = await sampling_client.sample_async(
                prompt=model_input,
                num_samples=config.group_size,
                sampling_params=sampling_params,
            )
            return result, prob

        sample_results = await asyncio.gather(
            *[sample_problem(p) for p in batch_problems]
        )

        # ---- Phase 2: Grade and collect correct solutions ----
        correct_datums: list[tinker.Datum] = []
        n_total_samples = 0
        n_correct_samples = 0
        n_problems_solved = 0

        # Per-level tracking (for MATH dataset)
        level_totals: dict[str, int] = {}
        level_solved: dict[str, int] = {}

        for result, prob in sample_results:
            problem_correct_count = 0
            level = prob.get("level", "")

            if level:
                level_totals[level] = level_totals.get(level, 0) + 1

            for seq in result.sequences:
                n_total_samples += 1
                parsed_message, _parse_ok = renderer.parse_response(seq.tokens)
                content = renderers.get_text_content(parsed_message)

                if grade_response(content, prob["answer"]):
                    n_correct_samples += 1
                    problem_correct_count += 1

                    # Optionally limit datums per problem to avoid
                    # overweighting easy problems in the SFT batch
                    if (
                        config.max_datums_per_problem is not None
                        and problem_correct_count > config.max_datums_per_problem
                    ):
                        continue

                    # Create SFT datum from this correct solution
                    conversation = convo_prefix + [
                        {
                            "role": "user",
                            "content": prob["problem"] + question_suffix,
                        },
                        {"role": "assistant", "content": content},
                    ]
                    datum = conversation_to_datum(
                        conversation,
                        renderer,
                        max_length=config.max_length,
                        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                    )
                    correct_datums.append(datum)

            if problem_correct_count > 0:
                n_problems_solved += 1
                if level:
                    level_solved[level] = level_solved.get(level, 0) + 1

        solve_rate = n_problems_solved / len(batch_problems) if batch_problems else 0
        sample_accuracy = n_correct_samples / n_total_samples if n_total_samples else 0

        metrics.update(
            {
                "train/n_problems": len(batch_problems),
                "train/n_total_samples": n_total_samples,
                "train/n_correct_samples": n_correct_samples,
                "train/sample_accuracy": sample_accuracy,
                "train/solve_rate": solve_rate,
                "train/n_sft_datums": len(correct_datums),
            }
        )

        # Per-level solve rates
        for level in sorted(level_totals.keys()):
            total = level_totals[level]
            solved = level_solved.get(level, 0)
            metrics[f"train/solve_rate_L{level}"] = solved / total if total else 0
            metrics[f"train/n_problems_L{level}"] = total

        logger.info(
            f"Batch {batch_idx}: {n_correct_samples}/{n_total_samples} correct "
            f"({sample_accuracy:.1%}), {n_problems_solved}/{len(batch_problems)} "
            f"problems solved ({solve_rate:.1%}), {len(correct_datums)} SFT datums"
        )
        if level_totals:
            level_summary = ", ".join(
                f"L{lv}={level_solved.get(lv, 0)}/{level_totals[lv]}"
                for lv in sorted(level_totals.keys())
            )
            logger.info(f"  Per-level solve rates: {level_summary}")

        # ---- Phase 3: Fine-tune on correct solutions ----
        if correct_datums:
            fwd_bwd_future = await training_client.forward_backward_async(
                correct_datums, loss_fn="cross_entropy"
            )
            optim_step_future = await training_client.optim_step_async(adam_params)

            fwd_bwd_result = await fwd_bwd_future.result_async()
            optim_result = await optim_step_future.result_async()

            if optim_result.metrics:
                metrics.update(optim_result.metrics)

            logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            weights = [d.loss_fn_inputs["weights"] for d in correct_datums]
            train_nll = compute_mean_nll(logprobs, weights)
            metrics["train/mean_nll"] = train_nll
        else:
            logger.warning(
                f"Batch {batch_idx}: no correct solutions found, skipping training step"
            )

        # ---- Phase 4: Evaluate ----
        if config.eval_every > 0 and batch_idx % config.eval_every == 0:
            eval_sampling_client = (
                await training_client.save_weights_and_get_sampling_client_async()
            )
            eval_metrics = await evaluate_pass_at_1(
                eval_sampling_client,
                test_data,
                renderer,
                convo_prefix,
                question_suffix,
                config.max_tokens,
            )
            metrics.update(eval_metrics)
            logger.info(
                f"Batch {batch_idx} eval: pass@1 = {eval_metrics['test/correct']:.4f}"
            )

        # ---- Phase 5: Checkpoint ----
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=log_path,
                kind="both",
                loop_state={"batch": batch_idx},
                ttl_seconds=config.ttl_seconds,
            )

        metrics["time/total"] = time.time() - t_start
        metrics["progress"] = (batch_idx + 1) / n_batches
        ml_logger.log_metrics(metrics=metrics, step=batch_idx)

    # Final checkpoint
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=log_path,
        kind="both",
        loop_state={"batch": n_batches},
        ttl_seconds=None,
    )

    # Final evaluation
    final_sampling_client = (
        await training_client.save_weights_and_get_sampling_client_async()
    )
    final_metrics = await evaluate_pass_at_1(
        final_sampling_client,
        test_data,
        renderer,
        convo_prefix,
        question_suffix,
        config.max_tokens,
    )
    logger.info(f"Final eval: pass@1 = {final_metrics['test/correct']:.4f}")
    ml_logger.log_metrics(metrics=final_metrics, step=n_batches)

    ml_logger.close()
    logger.info("RFT training completed")


def cli_main(config: Config):
    asyncio.run(main(config))


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
