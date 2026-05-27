import asyncio
from pathlib import Path

import chz

from tinker_cookbook.recipes.harbor_rl.eval import EvalConfig, TaskResult, run_eval
from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    default_sandbox_factory,
    load_harbor_tasks,
)

DATASETS: dict[str, str] = {
    "terminal_bench": "terminal-bench-2.0/terminal-bench",
    "swe_bench": "swebench-verified-1.0/swebench-verified",
}


@chz.chz
class CLIConfig:
    model_name: str = "moonshotai/Kimi-K2.6"
    checkpoint_url: str | None = None
    benchmarks: str = "terminal_bench"
    output_path: str = "/tmp/tinker-full-reruns/harbor_eval"

    max_turns: int = 200
    max_tokens: int = 8192
    temperature: float = 0.1
    sandbox_timeout: int = 3600
    command_timeout: int = 120
    grader_timeout: int = 60
    max_tasks: int | None = None

    base_url: str | None = None
    renderer_name: str | None = None


def parse_benchmarks(benchmarks: str) -> list[str]:
    names = [name.strip() for name in benchmarks.split(",") if name.strip()]
    unknown = sorted(set(names) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown Harbor benchmark(s): {unknown}. Available: {sorted(DATASETS)}")
    if not names:
        raise ValueError("At least one benchmark must be specified")
    return names


def print_summary(benchmark: str, results: list[TaskResult]) -> None:
    total = len(results)
    passed = sum(1 for result in results if result.error is None and result.reward > 0)
    errored = sum(1 for result in results if result.error is not None)
    failed = total - passed - errored
    pass_rate = passed / total if total else 0.0
    summary = (
        f"{benchmark}: total={total} pass={passed} fail={failed} "
        f"error={errored} pass_rate={pass_rate:.1%}"
    )
    print(summary)


async def run_benchmark(cli_config: CLIConfig, benchmark: str) -> list[TaskResult]:
    eval_config = EvalConfig(
        model_name=cli_config.model_name,
        output_path=str(Path(cli_config.output_path) / benchmark),
        max_turns=cli_config.max_turns,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        sandbox_timeout=cli_config.sandbox_timeout,
        command_timeout=cli_config.command_timeout,
        grader_timeout=cli_config.grader_timeout,
        max_tasks=cli_config.max_tasks,
        checkpoint_url=cli_config.checkpoint_url,
        base_url=cli_config.base_url,
        renderer_name=cli_config.renderer_name,
    )
    tasks = load_harbor_tasks(DATASETS[benchmark])
    print(f"Running {benchmark} on {len(tasks)} tasks")
    results = await run_eval(eval_config, tasks, sandbox_factory=default_sandbox_factory)
    print_summary(benchmark, results)
    return results


async def main(cli_config: CLIConfig) -> None:
    for benchmark in parse_benchmarks(cli_config.benchmarks):
        _ = await run_benchmark(cli_config, benchmark)


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    asyncio.run(main(config))
