"""Resume-cache fingerprinting tests.

The resume path in ``_runner.py`` only reuses stored trajectory rows whose
fingerprint (model, sampling config, effective system prompt, benchmark,
timeout, grade_fn identity) matches the current run. These tests pin the
guarantees:

- same model + same config: resume works, nothing is re-sampled
- different model, or any reward-relevant config change: nothing is reused
- rows from other fingerprints stay in the file and do not poison a later
  matching run (per-row filtering, not file-level invalidation)
- rows without a fingerprint (pre-fingerprint caches) are never reused
- timeout rows do not survive a ``timeout_seconds`` increase
"""

import json
from pathlib import Path
from typing import cast

import pytest
import tinker

from tinker_cookbook import renderers
from tinker_cookbook.eval.benchmarks._runner import (
    _run_fingerprint,
    load_result,
    run_benchmark,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.rl.types import Env, StepResult


class FakeTokenizer:
    """Byte-level codec: enough for the runner's decode() calls."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def __call__(self, text: str) -> list[int]:
        return self.encode(text)


class FakeRenderer:
    tokenizer = FakeTokenizer()


class FakeSeq:
    def __init__(self, tokens: list[int]):
        self.tokens = tokens
        self.logprobs = [0.0] * len(tokens)
        self.stop_reason = "stop"


class FakeSampleResult:
    def __init__(self, sequences: list[FakeSeq]):
        self.sequences = sequences


class FakeSamplingClient:
    """A 'model' whose every answer is ``self.answer``."""

    def __init__(self, model_path: str, answer: str):
        self.model_path = model_path
        self.answer = answer
        self.calls = 0

    async def sample_async(self, prompt, sampling_params=None, num_samples=1):
        self.calls += 1
        toks = FakeTokenizer().encode(self.answer)
        return FakeSampleResult([FakeSeq(toks) for _ in range(num_samples)])


QUESTIONS = [f"question {i}: what is the answer?" for i in range(10)]


class ResumeTestEnv(Env):
    """Single-turn env: reward 1.0 iff the response contains ANSWER_CORRECT."""

    def __init__(self, question: str, example_id: str):
        self.question = question
        self.example_id = example_id

    async def initial_observation(self):
        return tinker.ModelInput.from_ints(FakeTokenizer().encode(self.question)), []

    async def step(self, action, *, extra=None) -> StepResult:
        response = FakeTokenizer().decode(action)
        reward = 1.0 if "ANSWER_CORRECT" in response else 0.0
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.from_ints([]),
            next_stop_condition=[],
            metrics={},
            logs={"expected": "ANSWER_CORRECT", "got": response},
        )


class ResumeTestBenchmark(BenchmarkBuilder):
    name = "resumetest"
    multi_turn = False

    def make_envs(self, renderer, config):
        return [ResumeTestEnv(q, f"resumetest::{i}") for i, q in enumerate(QUESTIONS)]


def _stamp(save_dir: Path, entries: list[dict]) -> None:
    """Hand-write trajectories.jsonl rows, fingerprinted exactly as the runner would."""
    traj_dir = save_dir / "resumetest"
    traj_dir.mkdir(parents=True, exist_ok=True)
    with open(traj_dir / "trajectories.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _row(idx: int, reward: float, error: str | None, fingerprint: dict | None) -> dict:
    return {
        "idx": idx,
        "benchmark": "resumetest",
        "example_id": f"resumetest::{idx}",
        "turns": [],
        "reward": reward,
        "metrics": {},
        "logs": {},
        "error": error,
        "time_seconds": 1.0,
        "fingerprint": fingerprint,
    }


BENCH = ResumeTestBenchmark()
RENDERER = FakeRenderer()


async def _run(client: FakeSamplingClient, config: BenchmarkConfig):
    """run_benchmark with the fakes cast to their real parameter types."""
    return await run_benchmark(
        BENCH,
        cast(tinker.SamplingClient, client),
        cast(renderers.Renderer, RENDERER),
        config,
    )


GOOD = "ANSWER_CORRECT"
BAD = "wrong answer"


@pytest.mark.asyncio
async def test_same_model_and_config_resumes_without_resampling(tmp_path):
    client = FakeSamplingClient("model-A", GOOD)
    cfg = BenchmarkConfig(save_dir=str(tmp_path), timeout_seconds=30)
    first = await _run(client, cfg)
    assert first.score == 1.0 and client.calls == 10

    again = await _run(client, cfg)
    assert again.score == 1.0
    assert again.num_reused == 10
    assert client.calls == 10  # nothing re-sampled
    # error rows also resume (is_error counted), still no resampling
    assert again.num_errors == 0


@pytest.mark.asyncio
async def test_different_model_reuses_nothing(tmp_path):
    model_a = FakeSamplingClient("model-A-step500", GOOD)
    await _run(model_a, BenchmarkConfig(save_dir=str(tmp_path)))

    model_b = FakeSamplingClient("model-B-step5000", BAD)
    res_b = await _run(model_b, BenchmarkConfig(save_dir=str(tmp_path)))
    assert model_b.calls == 10  # B ran everything itself
    assert res_b.score == 0.0  # honest score, not A's 1.0
    assert res_b.num_reused == 0


@pytest.mark.asyncio
async def test_reward_relevant_config_change_reuses_nothing(tmp_path):
    client = FakeSamplingClient("model-A", GOOD)
    await _run(client, BenchmarkConfig(save_dir=str(tmp_path), temperature=0.6))

    for changed in [
        BenchmarkConfig(save_dir=str(tmp_path), temperature=0.0),
        BenchmarkConfig(save_dir=str(tmp_path), temperature=0.6, max_tokens=1024),
        BenchmarkConfig(save_dir=str(tmp_path), temperature=0.6, system_prompt="Answer briefly."),
    ]:
        client.calls = 0
        res = await _run(client, changed)
        assert client.calls == 10, "config change must force a fresh run"
        assert res.num_reused == 0


@pytest.mark.asyncio
async def test_same_qualname_different_grade_fn_body_reuses_nothing(tmp_path):
    """Two graders sharing a qualname but not a body must not blend rows."""
    client = FakeSamplingClient("model-A", BAD)  # responses never contain GOOD

    def grade_strict(_response, _logs):
        return 0.0

    grade_strict.__qualname__ = "grade"  # same qualname as the lenient grader below

    await _run(client, BenchmarkConfig(save_dir=str(tmp_path), grade_fn=grade_strict))

    def grade_lenient(_response, _logs):
        return 1.0

    grade_lenient.__qualname__ = "grade"

    client.calls = 0
    res = await _run(client, BenchmarkConfig(save_dir=str(tmp_path), grade_fn=grade_lenient))
    assert res.num_reused == 0
    assert client.calls == 10
    assert res.score == 1.0  # the lenient grader's own honest score


@pytest.mark.asyncio
async def test_timeout_increase_retries_timed_out_examples(tmp_path):
    client = FakeSamplingClient("model-A", GOOD)
    # A previous run under a 300s limit recorded example 4 as a timeout.
    fp = _run_fingerprint(
        BENCH,
        BenchmarkConfig(save_dir=str(tmp_path), timeout_seconds=300),
        cast(tinker.SamplingClient, client),
    )
    _stamp(
        tmp_path,
        [_row(4, 0.0, "timeout (300s)", fp)],
    )
    res = await _run(client, BenchmarkConfig(save_dir=str(tmp_path), timeout_seconds=3600))
    assert res.num_reused == 0  # the timeout row is from a different fingerprint
    assert client.calls == 10
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_stale_rows_do_not_poison_later_matching_run(tmp_path):
    model_a = FakeSamplingClient("model-A", GOOD)
    model_b = FakeSamplingClient("model-B", BAD)
    cfg = BenchmarkConfig(save_dir=str(tmp_path))
    await _run(model_a, cfg)
    await _run(model_b, cfg)

    # Third run matches model A again: A's rows are still in the append-only
    # file and must be reused despite B's rows interleaved after them.
    model_a.calls = 0
    res = await _run(model_a, cfg)
    assert model_a.calls == 0
    assert res.score == 1.0
    assert res.num_reused == 10


@pytest.mark.asyncio
async def test_legacy_rows_without_fingerprint_are_never_reused(tmp_path):
    client = FakeSamplingClient("model-A", GOOD)
    _stamp(tmp_path, [_row(i, 1.0, None, None) for i in range(10)])  # pre-fingerprint cache
    res = await _run(client, BenchmarkConfig(save_dir=str(tmp_path)))
    assert res.num_reused == 0
    assert client.calls == 10
    assert res.score == 1.0


@pytest.mark.asyncio
async def test_partial_cache_of_same_model_resumes_only_missing(tmp_path):
    client = FakeSamplingClient("model-A", GOOD)
    fp = _run_fingerprint(
        BENCH, BenchmarkConfig(save_dir=str(tmp_path)), cast(tinker.SamplingClient, client)
    )
    _stamp(
        tmp_path,
        [
            _row(0, 1.0, None, fp),
            _row(1, 1.0, None, fp),
            _row(2, 1.0, None, fp),
            _row(3, 0.0, None, fp),  # a recorded failure stays recorded
            _row(4, 0.0, "timeout (300s)", fp),
        ],
    )
    res = await _run(client, BenchmarkConfig(save_dir=str(tmp_path)))
    assert res.num_reused == 5
    assert client.calls == 5  # only the missing five ran
    # cached: 3 correct of 5; fresh: 5 correct of 5 -> 8/10
    assert res.score == 0.8


@pytest.mark.asyncio
async def test_result_json_records_fingerprint_and_reuse(tmp_path):
    client = FakeSamplingClient("model-A", GOOD)
    cfg = BenchmarkConfig(save_dir=str(tmp_path))
    await _run(client, cfg)
    await _run(client, cfg)

    loaded = load_result(str(tmp_path), "resumetest")
    assert loaded is not None
    assert loaded.num_reused == 10
    assert loaded.fingerprint is not None
    assert loaded.fingerprint["model"] == "model-A"
    assert loaded.fingerprint["benchmark"] == "resumetest"
