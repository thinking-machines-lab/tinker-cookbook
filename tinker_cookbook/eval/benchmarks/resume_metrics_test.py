"""Resumed summaries retain the counters collected during the original rollout."""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
import tinker

from tinker_cookbook.completers import StopCondition
from tinker_cookbook.eval.benchmarks._runner import load_trajectories, run_benchmark
from tinker_cookbook.eval.benchmarks._types import (
    METRIC_MAX_TOKENS_REACHED,
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
    Logs,
)
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Action, ActionExtra, Env, Observation, StepResult

Mode = Literal["success", "truncated", "custom", "error"]


class _Tokenizer:
    def decode(self, tokens: list[int]) -> str:
        return " ".join(str(token) for token in tokens)


class _Renderer:
    tokenizer = _Tokenizer()


class _Sampler:
    def __init__(self, mode: Mode):
        self.mode = mode
        self.calls = 0

    async def sample_async(
        self,
        prompt: tinker.ModelInput,
        num_samples: int = 1,
        sampling_params: tinker.SamplingParams | None = None,
    ) -> tinker.types.SampleResponse:
        self.calls += 1
        if self.mode == "error" and prompt.to_ints()[-1] == 1:
            raise ValueError("Synthetic sampling failure")
        return tinker.types.SampleResponse(
            sequences=[
                tinker.types.SampledSequence(
                    tokens_np=np.array([20, 21], dtype=np.int32),
                    logprobs_np=np.array([-1.0, -1.0], dtype=np.float32),
                    stop_reason="stop",
                    sequence_id=f"synthetic-{self.calls}-{index}",
                )
                for index in range(num_samples)
            ]
        )


class _CounterEnv(Env):
    def __init__(self, index: int, mode: Mode):
        self.example_id = f"counter-{index}"
        self.index = index
        self.mode = mode

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return tinker.ModelInput.from_ints([10, 11, self.index]), []

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        metrics = {"correct": float(self.mode != "truncated")}
        if self.mode == "truncated":
            metrics[METRIC_MAX_TOKENS_REACHED] = 1.0
        return StepResult(
            reward=metrics["correct"],
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics=metrics,
            logs={"fixture": "counter"},
        )


class _CounterBenchmark(BenchmarkBuilder):
    name = "resume_counters"

    def __init__(self, mode: Mode):
        self.mode: Mode = mode

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        count = min(config.max_examples or 2, 2)
        return [_CounterEnv(index, self.mode) for index in range(count)]


def _custom_grade(response: str, logs: Logs) -> float:
    assert response == "20 21" and logs["fixture"] == "counter"
    return 0.5


@pytest.mark.parametrize("cached_examples", [1, 2])
@pytest.mark.parametrize("mode", ["success", "truncated", "custom", "error"])
def test_resume_preserves_rollout_summary(tmp_path: Path, cached_examples: int, mode: Mode) -> None:
    renderer = cast(Renderer, _Renderer())
    benchmark = _CounterBenchmark(mode)
    sampler = _Sampler(mode)
    fresh_sampler = _Sampler(mode)
    grade_fn = _custom_grade if mode == "custom" else None
    cached_path = str(tmp_path / "resumed")

    async def run() -> tuple[BenchmarkResult, BenchmarkResult]:
        await run_benchmark(
            benchmark,
            cast(tinker.SamplingClient, sampler),
            renderer,
            BenchmarkConfig(save_dir=cached_path, max_examples=cached_examples, grade_fn=grade_fn),
        )
        assert sampler.calls == cached_examples
        resumed = await run_benchmark(
            benchmark,
            cast(tinker.SamplingClient, sampler),
            renderer,
            BenchmarkConfig(save_dir=cached_path, grade_fn=grade_fn),
        )
        fresh = await run_benchmark(
            benchmark,
            cast(tinker.SamplingClient, fresh_sampler),
            renderer,
            BenchmarkConfig(save_dir=str(tmp_path / "fresh"), grade_fn=grade_fn),
        )
        return resumed, fresh

    resumed, fresh = asyncio.run(run())
    assert sampler.calls == fresh_sampler.calls == 2
    successful = 1 if mode == "error" else 2
    assert fresh.num_errors == int(mode == "error")
    assert fresh.num_truncated == (2 if mode == "truncated" else 0)
    assert fresh.score == (0.0 if mode == "truncated" else successful / 2)
    assert fresh.metrics == {
        "total_ac_tokens": 2 * successful,
        "total_ob_tokens": 3 * successful,
        "total_turns": successful,
        "turns_per_episode": successful / 2,
        "ac_tokens_per_turn": 2.0,
        "ob_tokens_per_turn": 3.0,
    }
    assert resumed.metrics == fresh.metrics
    assert resumed.score == fresh.score
    assert resumed.num_errors == fresh.num_errors
    assert resumed.num_truncated == fresh.num_truncated
    for row in load_trajectories(cached_path, benchmark.name):
        if row.error is not None:
            continue
        assert row.metrics["_eval_turns"] == 1
        assert row.metrics["_eval_ac_tokens"] == 2
        assert row.metrics["_eval_ob_tokens"] == 3
        if mode == "custom":
            assert row.metrics["custom_graded"] == 1.0
            assert row.reward == 0.5
