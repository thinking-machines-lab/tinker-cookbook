"""
Minimal RL training loop using GRPO-style reward centering.

Variable naming convention (see CONTRIBUTING.md):
    _P: Problem dimension (different questions/prompts in a batch)
    _G: Group dimension (multiple rollouts per problem for variance reduction)
    _T: Token/Time dimension (sequence positions)
    _D: Datum dimension (training examples after flattening)

Example: `tokens_G_T` is a list of token sequences, one per group member.
In this script, datums_D has size P*G (one datum per rollout).
"""

import logging
import time
from concurrent.futures import Future

import chz
import datasets
import tinker
import torch
from tinker import types
from tqdm import tqdm
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/rl-loop"
    model_name: str = "meta-llama/Llama-3.1-8B"
    batch_size: int = 128
    group_size: int = 16
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 20  # 0 = disabled
    max_tokens: int = 256


def get_reward(response: str, answer: str) -> float:
    try:
        given_answer = extract_boxed(response)
        ground_truth = extract_gsm8k_final_answer(answer)
        return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
    except ValueError:
        return 0.0


def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load GSM8K dataset
    logger.info("Loading dataset...")
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]

    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."

    convo_prefix = [
        {
            "role": "user",
            "content": "How many r's are in strawberry?" + question_suffix,
        },
        {
            "role": "assistant",
            "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
        },
    ]

    n_train_batches = len(train_dataset) // config.batch_size

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info["state_path"]
        )
        start_batch = resume_info["batch"]
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    # Optimizer step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(f"Training for {n_train_batches} batches")

    # Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get training batch and convert to datums online
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        sampling_path = (
            training_client.save_weights_for_sampler(name=f"{batch_idx:06d}").result().path
        )
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        futures_P: list[Future[types.SampleResponse]] = []
        prompts_P: list[types.ModelInput] = []
        for question in batch_rows["question"]:
            convo = [
                *convo_prefix,
                {"role": "user", "content": question + question_suffix},
            ]
            model_input = renderer.build_generation_prompt(convo)

            # Generate group_size responses in a single call
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=config.group_size,
                sampling_params=sampling_params,
            )
            futures_P.append(future)
            prompts_P.append(model_input)

        for future, prompt, answer in tqdm(
            zip(futures_P, prompts_P, batch_rows["answer"]),
            total=len(futures_P),
            desc=f"Sampling batch {batch_idx}",
        ):
            sample_result = future.result()
            rewards_G: list[float] = []
            sampled_tokens_G_T: list[list[int]] = []
            logprobs_G_T: list[list[float]] = []
            for sequence in sample_result.sequences:
                sampled_tokens = sequence.tokens
                sampled_logprobs = sequence.logprobs
                assert sampled_logprobs is not None

                sampled_tokens_G_T.append(sampled_tokens)
                logprobs_G_T.append(sampled_logprobs)

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                content = renderers.get_text_content(parsed_message)
                reward = get_reward(content, answer)
                rewards_G.append(reward)

            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [reward - mean_reward for reward in rewards_G]
            rewards_P.append(mean_reward)

            # check if all advantages are zero
            if all(advantage == 0.0 for advantage in advantages_G):
                # Skip question because all advantages are the same
                continue

            for sampled_tokens, logprobs, advantage in zip(
                sampled_tokens_G_T, logprobs_G_T, advantages_G
            ):
                ob_len = prompt.length - 1
                model_input = prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))
                target_tokens = [0] * ob_len + sampled_tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                assert (
                    model_input.length
                    == len(target_tokens)
                    == len(padded_logprobs)
                    == len(padded_advantages)
                ), (
                    f"model_input.length: {model_input.length}, len(target_tokens): {len(target_tokens)}, "
                    f"len(padded_logprobs): {len(padded_logprobs)}, len(padded_advantages): {len(padded_advantages)}"
                )
                datum = types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                datums_D.append(datum)

        # Training step
        fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        # Log metrics
        metrics["time/total"] = time.time() - t_start
        metrics["reward/total"] = sum(rewards_P) / len(rewards_P)
        ml_logger.log_metrics(metrics, step=batch_idx)

        # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
