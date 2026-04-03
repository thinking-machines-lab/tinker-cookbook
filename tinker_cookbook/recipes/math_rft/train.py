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
    python -m tinker_cookbook.recipes.math_rft.train \\
        model_name="Qwen/Qwen3-1.7B" \\
        env=gsm8k \\
        group_size=16 \\
        groups_per_batch=64 \\
        learning_rate=2e-5 \\
        max_tokens=1024
"""

import asyncio
import logging
import math
import time
from typing import Literal

import chz
import tinker

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import (
    MathEnv,
    extract_gsm8k_final_answer,
    safe_grade,
)
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    # Model
    model_name: str = "Qwen/Qwen3-1.7B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Dataset
    env: Literal["gsm8k", "math"] = "gsm8k"
    seed: int = 0

    # Sampling
    group_size: int = 16  # K: number of solutions sampled per problem
    groups_per_batch: int = 64  # Number of problems per batch
    max_tokens: int = 1024
    temperature: float = 1.0

    # Training
    learning_rate: float = 2e-5
    max_length: int = 2048  # Max token length for SFT datums

    # Logging and checkpointing
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    eval_every: int = 5  # Evaluate every N batches
    save_every: int = 20
    ttl_seconds: int | None = 604800  # 7 days

    # Infrastructure
    base_url: str | None = None
    max_steps: int | None = None


def _default_log_path(config: Config) -> str:
    if config.log_path:
        return config.log_path
    model_short = config.model_name.split("/")[-1]
    return f"/tmp/tinker-examples/math_rft/{model_short}_{config.env}"


def _get_question_suffix(env: str) -> str:
    if env == "gsm8k":
        return " Provide a numerical answer without units, written inside \\boxed{}."
    else:
        return " Write your answer in \\boxed{} format."


def _grade_response(response_text: str, ground_truth: str) -> bool:
    """Check if a response contains the correct answer in \\boxed{} format."""
    try:
        given_answer = extract_boxed(response_text)
    except ValueError:
        return False
    return safe_grade(given_answer, ground_truth, grader="sympy", timeout=1.0)


def _load_dataset(
    env: str, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Load train and test problems as lists of {problem, answer} dicts."""
    if env == "gsm8k":
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", name="main")
        train_data = []
        for row in ds["train"].shuffle(seed=seed):
            try:
                answer = extract_gsm8k_final_answer(row["answer"])
                train_data.append({"problem": row["question"], "answer": answer})
            except ValueError:
                continue
        test_data = []
        for row in ds["test"]:
            try:
                answer = extract_gsm8k_final_answer(row["answer"])
                test_data.append({"problem": row["question"], "answer": answer})
            except ValueError:
                continue
        return train_data, test_data
    elif env == "math":
        from tinker_cookbook.recipes.math_rl.math_env import (
            _get_hendrycks_math_test,
            _get_hendrycks_math_train,
        )

        train_ds = _get_hendrycks_math_train().shuffle(seed=seed)
        train_data = []
        for row in train_ds:
            try:
                answer = extract_boxed(row["solution"])
                train_data.append({"problem": row["problem"], "answer": answer})
            except ValueError:
                continue
        test_ds = _get_hendrycks_math_test()
        test_data = []
        for row in test_ds:
            try:
                answer = extract_boxed(row["solution"])
                test_data.append({"problem": row["problem"], "answer": answer})
            except ValueError:
                continue
        return train_data, test_data
    else:
        raise ValueError(f"Unknown env: {env}")


async def evaluate_pass_at_1(
    sampling_client: tinker.SamplingClient,
    test_data: list[dict[str, str]],
    renderer: renderers.Renderer,
    convo_prefix: list[renderers.Message],
    question_suffix: str,
    max_tokens: int,
    max_problems: int = 500,
) -> dict[str, float]:
    """Evaluate pass@1 on the test set using greedy decoding."""
    problems = test_data[:max_problems]

    sampling_params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # Greedy for eval
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

    for result, answer in results:
        tokens = result.sequences[0].tokens
        parsed_message, _parse_ok = renderer.parse_response(tokens)
        content = renderers.get_text_content(parsed_message)

        try:
            _ = extract_boxed(content)
            n_format += 1
        except ValueError:
            pass

        if _grade_response(content, answer):
            n_correct += 1

    return {
        "test/correct": n_correct / n_total,
        "test/format": n_format / n_total,
        "test/n_problems": float(n_total),
    }


async def main(config: Config):
    log_path = _default_log_path(config)
    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    # Setup model
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    logger.info(f"Model: {config.model_name}, Renderer: {renderer_name}")

    # Load data
    logger.info(f"Loading {config.env} dataset...")
    train_data, test_data = _load_dataset(config.env, config.seed)
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

        for result, prob in sample_results:
            problem_had_correct = False

            for seq in result.sequences:
                n_total_samples += 1
                parsed_message, _parse_ok = renderer.parse_response(seq.tokens)
                content = renderers.get_text_content(parsed_message)

                if _grade_response(content, prob["answer"]):
                    n_correct_samples += 1
                    problem_had_correct = True

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

            if problem_had_correct:
                n_problems_solved += 1

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

        logger.info(
            f"Batch {batch_idx}: {n_correct_samples}/{n_total_samples} correct "
            f"({sample_accuracy:.1%}), {n_problems_solved}/{len(batch_problems)} "
            f"problems solved ({solve_rate:.1%}), {len(correct_datums)} SFT datums"
        )

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


def cli_main():
    config = chz.make_from_argv(Config)
    asyncio.run(main(config))


if __name__ == "__main__":
    cli_main()
