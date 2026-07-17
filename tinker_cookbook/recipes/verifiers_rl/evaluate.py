from __future__ import annotations

import asyncio
import statistics
import time
import tomllib
from pathlib import Path

import chz
import tinker
import verifiers.v1 as vf

from tinker_cookbook.recipes.verifiers_rl.tinker_client import TinkerClient
from tinker_cookbook.utils.git_rev import recipe_user_metadata


def print_results(traces: list[vf.Trace], elapsed: float) -> None:
    rewards = [trace.reward for trace in traces]
    errors = [trace for trace in traces if trace.has_error]
    print(f"Evaluation completed in {elapsed:.2f} seconds")
    print(f"Rollouts: {len(traces)} ({len(errors)} errors)")
    if rewards:
        print(f"Reward: {statistics.mean(rewards):.3f} ± {statistics.pstdev(rewards):.3f}")
    if traces:
        print("--- Example trace ---")
        print(traces[0].transcript)


async def evaluate(
    env_config_path: str,
    model_name: str | None,
    model_path: str | None,
    renderer_model_name: str | None,
    renderer_pool_size: int,
    num_tasks: int | None,
    rollouts_per_task: int,
    max_concurrent: int,
    max_tokens: int,
    temperature: float,
) -> list[vf.Trace]:
    service = tinker.ServiceClient(user_metadata=recipe_user_metadata("eval_verifiers_rl_v1"))
    if model_path is not None:
        run = await service.create_rest_client().get_training_run_by_tinker_path_async(model_path)
        if model_name is not None and model_name != run.base_model:
            raise ValueError(
                f"model_name {model_name!r} does not match checkpoint base model "
                f"{run.base_model!r}"
            )
        model_name = run.base_model
    if model_name is None:
        raise ValueError("model_name or model_path must be provided")

    sampling_client = (
        service.create_sampling_client(model_path=model_path, base_model=model_name)
        if model_path
        else service.create_sampling_client(base_model=model_name)
    )
    client = TinkerClient(
        sampling_client,
        renderer_model_name=renderer_model_name or model_name,
        renderer_pool_size=renderer_pool_size,
    )
    context = vf.ModelContext(
        model=model_name,
        client=client,
        sampling=vf.SamplingConfig(max_tokens=max_tokens, temperature=temperature),
    )
    raw_config = tomllib.loads(Path(env_config_path).read_text())
    env = vf.Environment(vf.EnvConfig.model_validate(raw_config))
    tasks = list(env.taskset.load())
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

    start = time.monotonic()
    async with env.serving():
        groups = await asyncio.gather(
            *(env.episode(task, context, n=rollouts_per_task).run(semaphore) for task in tasks)
        )
    traces = [trace for group in groups for trace in group]
    print_results(traces, time.monotonic() - start)
    return traces


@chz.chz
class CLIConfig:
    env_config_path: str
    model_name: str | None = None
    model_path: str | None = None
    renderer_model_name: str | None = None
    renderer_pool_size: int = 16
    num_tasks: int | None = 5
    rollouts_per_task: int = 3
    max_concurrent: int = 32
    max_tokens: int = 1024
    temperature: float = 1.0


async def cli_main(config: CLIConfig) -> list[vf.Trace]:
    return await evaluate(
        env_config_path=config.env_config_path,
        model_name=config.model_name,
        model_path=config.model_path,
        renderer_model_name=config.renderer_model_name,
        renderer_pool_size=config.renderer_pool_size,
        num_tasks=config.num_tasks,
        rollouts_per_task=config.rollouts_per_task,
        max_concurrent=config.max_concurrent,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
