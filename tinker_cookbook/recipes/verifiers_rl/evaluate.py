from __future__ import annotations

import asyncio
import json
import time

import chz
import numpy as np
import tinker
import verifiers.v1 as vf

from tinker_cookbook.recipes.verifiers_rl.tinker_generate import TinkerGenerateServer
from tinker_cookbook.recipes.verifiers_rl.verifiers_env import load_vf_env
from tinker_cookbook.utils.git_rev import recipe_user_metadata


def _message_text(message: vf.Message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = [text for part in content or [] if (text := getattr(part, "text", None))]
    return "\n".join(parts)


def _episode_reward(episode: vf.Episode) -> float:
    return sum(trace.reward for trace in episode.traces)


def log_results(
    episodes_by_example: list[list[vf.Episode]],
    vf_env_id: str,
    model_name: str,
    time_s: float,
):
    episodes = [episode for group in episodes_by_example for episode in group]
    rewards_by_example = [[_episode_reward(e) for e in group] for group in episodes_by_example]
    rewards = [reward for group in rewards_by_example for reward in group]
    metric_names = sorted(
        {
            name
            for episode in episodes
            for trace in episode.traces
            for name, value in trace.metrics.items()
            if value is not None
        }
    )

    print(f"Evaluation completed in {time_s:.2f} seconds")
    print("--- Evaluation ---")
    print(f"Environment: {vf_env_id}")
    print(f"Model: {model_name}")
    print(f"Examples: {len(episodes_by_example)}")
    print(f"Rollouts per example: {len(episodes_by_example[0]) if episodes_by_example else 0}")
    failed = sum(1 for episode in episodes if not episode.ok)
    if failed:
        print(f"Failed episodes: {failed}/{len(episodes)}")
    print("--- Example ---")
    for episode in episodes[:1]:
        for trace in episode.traces:
            for message in trace.messages:
                print(f"[{message.role}] {_message_text(message)}")
    print("--- All ---")
    print("Rewards:")
    print(f"reward: avg - {np.mean(rewards):.3f}, std - {np.std(rewards):.3f}")
    rollouts_per_example = max((len(group) for group in rewards_by_example), default=0)
    for r in range(rollouts_per_example):
        trials = [round(group[r], 3) for group in rewards_by_example if r < len(group)]
        print(f"r{r + 1}: {trials}")
    for name in metric_names:
        values = [
            float(value)
            for episode in episodes
            for trace in episode.traces
            if (value := trace.metrics.get(name)) is not None
        ]
        print(f"{name}: avg - {np.mean(values):.3f}, std - {np.std(values):.3f}")


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
    chat_template_kwargs: dict | None = None,
) -> list[list[vf.Episode]]:
    service = tinker.ServiceClient(
        user_metadata=recipe_user_metadata("eval_verifiers_rl"),
    )

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

    vf_env = load_vf_env(vf_env_id, vf_env_args)
    tasks = list(vf_env.taskset.head(num_examples))

    # Create sampling client from checkpoint path or base model
    if model_path:
        sampling = service.create_sampling_client(model_path=model_path, base_model=model_name)
    else:
        sampling = service.create_sampling_client(base_model=model_name)

    semaphore = asyncio.Semaphore(max_concurrent)
    start_time = time.time()
    async with TinkerGenerateServer(sampling) as server, vf_env.serving():
        ctx = vf.ModelContext(
            model=model_name,
            client=server.train_client_config(model_name),
            sampling=vf.SamplingConfig.model_validate(
                {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **(
                        {"chat_template_kwargs": chat_template_kwargs}
                        if chat_template_kwargs
                        else {}
                    ),
                }
            ),
        )
        slots_by_example = [vf_env.slots(task, rollouts_per_example) for task in tasks]
        episodes = await asyncio.gather(
            *(vf_env.run_slot(slot, ctx, semaphore) for slots in slots_by_example for slot in slots)
        )
    end_time = time.time()
    episodes_by_example = [
        episodes[i * rollouts_per_example : (i + 1) * rollouts_per_example]
        for i in range(len(tasks))
    ]
    log_results(episodes_by_example, vf_env_id, model_name, end_time - start_time)
    return episodes_by_example


@chz.chz
class CLIConfig:
    model_name: str | None = None  # Base model name (auto-detected from checkpoint if not provided)
    model_path: str | None = None  # Path to checkpoint (e.g., from checkpoints.jsonl sampler_path)
    vf_env_id: str = "reverse-text"
    vf_env_args: str | None = None  # JSON vf.EnvConfig data merged over the taskset id
    num_examples: int = 5
    rollouts_per_example: int = 3
    max_concurrent: int = 32
    max_tokens: int = 1024
    temperature: float = 1.0
    # JSON chat-template kwargs passed through verifiers' train client to the
    # `renderers` chat template that tokenizes
    # each turn, e.g. '{"enable_thinking": false}' for Qwen thinking control.
    chat_template_kwargs: str | None = None


async def cli_main(config: CLIConfig):
    env_args = json.loads(config.vf_env_args) if config.vf_env_args else {}
    return await evaluate(
        vf_env_id=config.vf_env_id,
        vf_env_args=env_args,
        model_name=config.model_name,
        num_examples=config.num_examples,
        rollouts_per_example=config.rollouts_per_example,
        max_concurrent=config.max_concurrent,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        model_path=config.model_path,
        chat_template_kwargs=(
            json.loads(config.chat_template_kwargs) if config.chat_template_kwargs else None
        ),
    )


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)

    asyncio.run(cli_main(config))
