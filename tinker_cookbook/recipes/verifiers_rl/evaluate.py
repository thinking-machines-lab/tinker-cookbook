from __future__ import annotations

import asyncio
import json
import time

import chz
import tinker
import numpy as np
import verifiers as vf
from verifiers.utils.message_utils import messages_to_printable

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
from tinker_cookbook.tokenizer_utils import get_tokenizer


def log_results(
    results: vf.GenerateOutputs,
    vf_env_id: str,
    model_name: str,
    num_examples: int,
    rollouts_per_example: int,
    time_s: float,
):
    print(f"Evaluation completed in {time_s:.2f} seconds")
    print("--- Evaluation ---")
    print(f"Environment: {vf_env_id}")
    print(f"Model: {model_name}")
    print(f"Examples: {num_examples}")
    print(f"Rollouts per example: {rollouts_per_example}")
    print("--- Example ---")
    printable_prompts = [messages_to_printable(p) for p in results["prompt"]]
    printable_completions = [messages_to_printable(c) for c in results["completion"]]
    vf.print_prompt_completions_sample(
        printable_prompts, printable_completions, results["reward"], step=0
    )
    print("--- All ---")
    print("Rewards:")
    print(
        f"reward: avg - {sum(results['reward']) / len(results['reward']):.3f}, std - {np.std(results['reward']):.3f}"
    )
    r = rollouts_per_example
    n = len(results["reward"]) // r
    for i in range(r):
        # rounded to 3 decimal places
        trials = [round(results["reward"][(i * n) + j], 3) for j in range(n)]
        out = f"r{i + 1}: {trials}"
        print(out)
    for k in results["metrics"]:
        v = results["metrics"][k]
        print(f"{k}: avg - {sum(v) / len(v):.3f}, std - {np.std(v):.3f}")
        for i in range(r):
            # rounded to 3 decimal places
            trials = [round(v[(i * n) + j], 3) for j in range(n)]
            out = f"r{i + 1}: {trials}"
            print(out)


async def evaluate(
    vf_env_id: str,
    vf_env_args: dict,
    model_name: str | None,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    max_tokens: int,
    temperature: float,
    model_path: str | None = None,
):
    service = tinker.ServiceClient()

    # If model_path is provided, get the base model from the training run
    if model_path is not None:
        rest_client = service.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(model_path)
        if model_name:
            if model_name != training_run.base_model:
                raise ValueError(
                    f"Model name {model_name} does not match training run base model {training_run.base_model}"
                )
        else:
            model_name = training_run.base_model

    if model_name is None:
        raise ValueError("model_name or model_path must be provided")

    env = vf.load_environment(vf_env_id, **vf_env_args)
    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Create sampling client from checkpoint path or base model
    if model_path:
        sampling = service.create_sampling_client(model_path=model_path, base_model=model_name)
    else:
        sampling = service.create_sampling_client(base_model=model_name)

    client = TinkerAsyncOpenAIClient(sampling, renderer, tokenizer)
    start_time = time.time()
    results = env.evaluate_sync(
        client=client,
        model=model_name,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
        sampling_args={
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    end_time = time.time()
    log_results(
        results,
        vf_env_id,
        model_name,
        num_examples,
        rollouts_per_example,
        end_time - start_time,
    )
    return results


@chz.chz
class CLIConfig:
    model_name: str | None = None  # Base model name (auto-detected from checkpoint if not provided)
    model_path: str | None = None  # Path to checkpoint (e.g., from checkpoints.jsonl sampler_path)
    vf_env_id: str = "reverse-text"
    vf_env_args: str | None = None  # JSON string
    num_examples: int = 5
    rollouts_per_example: int = 3
    max_concurrent: int = 32
    max_tokens: int = 1024
    temperature: float = 1.0


async def cli_main(cfg: CLIConfig):
    env_args = json.loads(cfg.vf_env_args) if cfg.vf_env_args else {}
    return await evaluate(
        vf_env_id=cfg.vf_env_id,
        vf_env_args=env_args,
        model_name=cfg.model_name,
        num_examples=cfg.num_examples,
        rollouts_per_example=cfg.rollouts_per_example,
        max_concurrent=cfg.max_concurrent,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        model_path=cfg.model_path,
    )


if __name__ == "__main__":
    cfg = chz.entrypoint(CLIConfig)

    asyncio.run(cli_main(cfg))
