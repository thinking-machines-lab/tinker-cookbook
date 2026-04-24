"""GSPO training loop on GSM8K.

Mirrors rl_loop.py but replaces importance_sampling with a sequence-level
GSPO objective via forward_backward_custom. The only structural changes are:

1. Datum contains only target_tokens (forward_backward_custom restriction).
2. Old logprobs and advantages are passed to the loss via closure.
3. Advantages are normalized by within-group std (per the GSPO paper).

Run:
    python -m tinker_cookbook.recipes.gspo.train

Variable naming follows the cookbook convention:
    _P  problem dimension  (questions in a batch)
    _G  group dimension    (rollouts per question)
    _D  datum dimension    (flattened training examples, P*G after filtering)
"""

import logging
import time

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from tqdm import tqdm

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.gspo.loss import DEFAULT_CLIP_HIGH, DEFAULT_CLIP_LOW, make_gspo_loss
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/gspo"
    model_name: str = "meta-llama/Llama-3.1-8B"
    batch_size: int = 128
    group_size: int = 16
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 20
    max_tokens: int = 256
    ttl_seconds: int | None = 604800
    clip_low: float = DEFAULT_CLIP_LOW
    clip_high: float = DEFAULT_CLIP_HIGH


def get_reward(response: str, answer: str) -> float:
    try:
        given_answer = extract_boxed(response)
        ground_truth = extract_gsm8k_final_answer(answer)
        return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
    except ValueError:
        return 0.0


def main(config: Config):
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    logger.info("Loading GSM8K...")
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"]

    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."
    convo_prefix = [
        {"role": "user", "content": "How many r's are in strawberry?" + question_suffix},
        {
            "role": "assistant",
            "content": "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. \\boxed{3}",
        },
    ]

    n_train_batches = len(train_dataset) // config.batch_size

    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info.state_path
        )
        start_batch = resume_info.batch
        logger.info(f"Resuming from batch {start_batch}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_batch = 0

    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    logger.info(f"Training for {n_train_batches} batches")

    for batch_idx in range(start_batch, n_train_batches):
        t_start = time.time()
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        if config.save_every > 0 and batch_idx % config.save_every == 0 and batch_idx > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{batch_idx:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
                ttl_seconds=config.ttl_seconds,
            )

        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        sampling_client = training_client.save_weights_and_get_sampling_client()

        futures_P = []
        prompts_P = []
        for question in batch_rows["question"]:
            convo = [*convo_prefix, {"role": "user", "content": question + question_suffix}]
            model_input = renderer.build_generation_prompt(convo)
            future = sampling_client.sample(
                prompt=model_input,
                num_samples=config.group_size,
                sampling_params=sampling_params,
            )
            futures_P.append(future)
            prompts_P.append(model_input)

        # Closure data: old logprobs + advantages, indexed parallel to datums_D.
        datums_D: list[types.Datum] = []
        old_logprobs_D: list[torch.Tensor] = []
        ob_lens_D: list[int] = []
        advantages_D: list[float] = []
        rewards_P: list[float] = []

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
                sampled_tokens_G_T.append(sequence.tokens)
                assert sequence.logprobs is not None
                logprobs_G_T.append(sequence.logprobs)
                parsed_message, _ = renderer.parse_response(sequence.tokens)
                content = renderers.get_text_content(parsed_message)
                rewards_G.append(get_reward(content, answer))

            rewards_P.append(sum(rewards_G) / len(rewards_G))

            # Skip groups where all rewards are identical (zero gradient signal).
            if len(set(rewards_G)) == 1:
                continue

            # Normalize advantages within the group by std (GSPO paper §3).
            mean_r = sum(rewards_G) / len(rewards_G)
            variance = sum((r - mean_r) ** 2 for r in rewards_G) / len(rewards_G)
            std_r = variance**0.5
            advantages_G = [(r - mean_r) / (std_r + 1e-8) for r in rewards_G]

            ob_len = prompt.length - 1

            for sampled_tokens, logprobs, advantage in zip(
                sampled_tokens_G_T, logprobs_G_T, advantages_G
            ):
                model_input = prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))
                target_tokens = [0] * ob_len + sampled_tokens
                padded_logprobs = [0.0] * ob_len + logprobs

                assert model_input.length == len(target_tokens) == len(padded_logprobs)

                # forward_backward_custom only accepts target_tokens (and optionally weights).
                # Old logprobs and advantages go in the closure.
                datum = types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    },
                )
                datums_D.append(datum)
                old_logprobs_D.append(torch.tensor(padded_logprobs, dtype=torch.float32))
                ob_lens_D.append(ob_len)
                advantages_D.append(advantage)

        if not datums_D:
            logger.warning("Batch %d: all groups had uniform rewards, skipping.", batch_idx)
        else:
            gspo_loss = make_gspo_loss(
                old_logprobs_D=old_logprobs_D,
                ob_lens_D=ob_lens_D,
                advantages_D=advantages_D,
                clip_low=config.clip_low,
                clip_high=config.clip_high,
            )
            fwd_bwd_future = training_client.forward_backward_custom(datums_D, gspo_loss)
            optim_future = training_client.optim_step(adam_params)
            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_future.result()

            if fwd_bwd_result.metrics:
                for k, v in fwd_bwd_result.metrics.items():
                    metrics[f"train/{k}"] = v
            if optim_result.metrics:
                metrics.update(optim_result.metrics)

        metrics["time/total"] = time.time() - t_start
        metrics["reward/mean"] = sum(rewards_P) / len(rewards_P) if rewards_P else 0.0
        ml_logger.log_metrics(metrics, step=batch_idx)

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
        ttl_seconds=None,
    )
    ml_logger.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
