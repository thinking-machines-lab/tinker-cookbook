"""Tests for the benchmark evaluation framework."""

import pytest

from tinker_cookbook.eval.benchmarks._types import (
    BenchmarkBuilder,
    BenchmarkConfig,
    BenchmarkResult,
    StoredTrajectory,
    StoredTurn,
)


class TestBenchmarkResult:
    def test_construction(self):
        r = BenchmarkResult(name="test", score=0.8, num_examples=100, num_correct=80)
        assert r.score == 0.8
        assert r.num_correct == 80

    def test_default_metrics(self):
        r = BenchmarkResult(name="test", score=0.5, num_examples=10, num_correct=5)
        assert r.metrics == {}


class TestBenchmarkConfig:
    def test_defaults(self):
        c = BenchmarkConfig()
        assert c.max_examples is None
        assert c.concurrency == 64
        assert c.agent_concurrency == 8
        assert c.max_tokens == 32768
        assert c.temperature == 0.6
        assert c.save_dir is None
        assert c.judge_sampling_client is None


class TestGSM8KImport:
    def test_gsm8k_registered(self):
        # Importing the module triggers registration
        import tinker_cookbook.eval.benchmarks.gsm8k  # noqa: F401
        from tinker_cookbook.eval.benchmarks import REGISTRY

        assert "gsm8k" in REGISTRY
        assert REGISTRY["gsm8k"].name == "gsm8k"

    def test_answer_extraction(self):
        from tinker_cookbook.eval.benchmarks.gsm8k import extract_gsm8k_answer

        assert extract_gsm8k_answer("The answer is 42") == "42"
        assert extract_gsm8k_answer("\\boxed{123}") == "123"
        assert extract_gsm8k_answer("#### 7") == "7"
        assert extract_gsm8k_answer("So we get 3.14 as the result") == "3.14"

    def test_check_gsm8k(self):
        from tinker_cookbook.eval.benchmarks.gsm8k import check_gsm8k

        assert check_gsm8k("The answer is 42.", "42")
        assert check_gsm8k("\\boxed{42}", "42")
        assert not check_gsm8k("The answer is 43.", "42")


class TestDefaultAggregate:
    def test_accuracy(self):
        builder = type("B", (BenchmarkBuilder,), {
            "name": "test",
            "make_envs": lambda self, r, c: [],
        })()
        result = builder.aggregate([1.0, 0.0, 1.0, 1.0], [{}, {}, {}, {}])
        assert result.score == 0.75
        assert result.num_correct == 3
        assert result.num_examples == 4

    def test_empty(self):
        builder = type("B", (BenchmarkBuilder,), {
            "name": "test",
            "make_envs": lambda self, r, c: [],
        })()
        result = builder.aggregate([], [])
        assert result.score == 0.0
        assert result.num_examples == 0


class TestStoredTrajectory:
    def test_roundtrip(self):
        traj = StoredTrajectory(
            idx=42,
            benchmark="gsm8k",
            turns=[
                StoredTurn(role="user", content="What is 2+2?", token_count=5),
                StoredTurn(role="assistant", content="The answer is 4.", token_count=6),
            ],
            reward=1.0,
            metrics={"correct": 1.0},
            logs={"expected": "4", "extracted": "4"},
            time_seconds=1.5,
        )
        d = traj.to_dict()
        restored = StoredTrajectory.from_dict(d)
        assert restored.idx == 42
        assert restored.reward == 1.0
        assert len(restored.turns) == 2
        assert restored.turns[0].role == "user"
        assert restored.turns[1].content == "The answer is 4."
        assert restored.metrics["correct"] == 1.0

    def test_error_trajectory(self):
        traj = StoredTrajectory(
            idx=0,
            benchmark="test",
            turns=[],
            reward=0.0,
            error="Connection timeout",
        )
        d = traj.to_dict()
        assert d["error"] == "Connection timeout"
        assert d["turns"] == []

    def test_multi_turn(self):
        traj = StoredTrajectory(
            idx=0,
            benchmark="terminal_bench",
            turns=[
                StoredTurn(role="user", content="Fix the bug in server.py"),
                StoredTurn(role="assistant", content="cat server.py"),
                StoredTurn(role="environment", content="import flask\n..."),
                StoredTurn(role="assistant", content="sed -i 's/old/new/' server.py"),
                StoredTurn(role="environment", content="File modified"),
            ],
            reward=1.0,
            time_seconds=45.0,
        )
        assert len(traj.turns) == 5
        assert traj.turns[2].role == "environment"


# ---------------------------------------------------------------------------
# Import / registration tests for all benchmark modules
# ---------------------------------------------------------------------------


class TestBenchmarkRegistration:
    """Verify that importing each benchmark module registers it in the REGISTRY."""

    def _check_registered(self, module_path: str, expected_name: str):
        import importlib
        importlib.import_module(module_path)
        from tinker_cookbook.eval.benchmarks import REGISTRY
        assert expected_name in REGISTRY, f"{expected_name} not found in REGISTRY"
        assert REGISTRY[expected_name].name == expected_name

    def test_mmlu_pro_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.mmlu_pro", "mmlu_pro")

    def test_mmlu_redux_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.mmlu_redux", "mmlu_redux")

    def test_gpqa_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.gpqa", "gpqa")

    def test_math500_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.math500", "math500")

    def test_aime_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.aime", "aime")

    def test_ifeval_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.ifeval", "ifeval")

    def test_ifbench_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.ifbench", "ifbench")

    def test_bfcl_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.bfcl", "bfcl")

    def test_longbench_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.longbench", "longbench")

    def test_tau2_bench_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.tau2_bench", "tau2_bench")

    def test_arena_hard_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.arena_hard", "arena_hard")

    def test_swe_bench_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.swe_bench", "swe_bench")

    def test_terminal_bench_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.terminal_bench", "terminal_bench")

    def test_mbpp_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.mbpp", "mbpp")

    def test_livecodebench_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.livecodebench", "livecodebench")


class TestMCQExtraction:
    """Test answer extraction functions from MCQ benchmarks."""

    def test_mmlu_pro_extract(self):
        from tinker_cookbook.eval.benchmarks.mmlu_pro import extract_mcq_answer
        assert extract_mcq_answer("The answer is (B).", "ABCDEFGHIJ") == "B"
        assert extract_mcq_answer("\\boxed{C}", "ABCDEFGHIJ") == "C"
        assert extract_mcq_answer("I think D is correct.", "ABCDEFGHIJ") == "D"

    def test_bfcl_function_matching(self):
        from tinker_cookbook.eval.benchmarks.bfcl import _match_function_call
        gen = {"name": "get_weather", "arguments": {"city": "London"}}
        exp = {"name": "get_weather", "arguments": {"city": "london"}}
        assert _match_function_call(gen, exp)

        gen_wrong = {"name": "get_weather", "arguments": {"city": "Paris"}}
        assert not _match_function_call(gen_wrong, exp)

    def test_tau2_bench_action_extraction(self):
        from tinker_cookbook.eval.benchmarks.tau2_bench import _extract_action
        text = '```json\n{"action": "refund", "arguments": {"order_id": "123"}}\n```'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "refund"
