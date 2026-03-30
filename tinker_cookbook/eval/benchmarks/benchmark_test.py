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


class TestGradingConsistency:
    """Validate that new framework grading matches recipe implementation."""

    def test_gsm8k_extraction_matches_recipe(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_gsm8k_answer as old
        from tinker_cookbook.eval.benchmarks.gsm8k import extract_gsm8k_answer as new

        cases = [
            "The answer is 42",
            "\\boxed{123}",
            "#### 7",
            "Result is 3.14159",
            "99 + 1 = 100",
        ]
        for resp in cases:
            assert old(resp) == new(resp), f"Mismatch on: {resp}"

    def test_gsm8k_check_matches_recipe(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.gsm8k import _check_gsm8k as old
        from tinker_cookbook.eval.benchmarks.gsm8k import check_gsm8k as new

        cases = [
            ("The answer is 42", "42", True),
            ("\\boxed{42}", "42", True),
            ("The answer is 43", "42", False),
            ("#### 0", "0", True),
        ]
        for resp, expected, should_be in cases:
            assert old(resp, expected) == new(resp, expected), f"Mismatch: {resp} vs {expected}"

    def test_mcq_extraction_matches_recipe(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_mcq_answer as old
        from tinker_cookbook.eval.benchmarks.mmlu_pro import extract_mcq_answer as new

        cases = [
            "The answer is (B).",
            "\\boxed{C}",
            "I believe A is correct.",
            "D",
        ]
        for resp in cases:
            assert old(resp) == new(resp), f"Mismatch on: {resp}"


class TestEvalStore:
    def test_create_and_list_runs(self, tmp_path):
        from tinker_cookbook.eval.store import EvalStore

        store = EvalStore(tmp_path / "eval_store")
        run_id = store.create_run(
            model_name="test-model",
            benchmarks=["gsm8k", "mmlu_pro"],
            checkpoint_name="step500",
        )
        assert "step500" in run_id

        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0].model_name == "test-model"
        assert runs[0].benchmarks == ["gsm8k", "mmlu_pro"]

    def test_run_dir(self, tmp_path):
        from tinker_cookbook.eval.store import EvalStore

        store = EvalStore(tmp_path / "eval_store")
        run_id = store.create_run(
            model_name="test", benchmarks=["gsm8k"], run_id="my_run",
        )
        assert run_id == "my_run"
        assert "my_run" in store.run_dir(run_id)

    def test_finalize_run(self, tmp_path):
        import json
        from tinker_cookbook.eval.store import EvalStore

        store = EvalStore(tmp_path / "eval_store")
        run_id = store.create_run(
            model_name="test", benchmarks=["gsm8k"], run_id="test_run",
        )

        # Simulate a result file
        result_dir = tmp_path / "eval_store" / "runs" / "test_run" / "gsm8k"
        result_dir.mkdir(parents=True)
        with open(result_dir / "result.json", "w") as f:
            json.dump({"score": 0.85, "num_correct": 85, "num_examples": 100}, f)

        meta = store.finalize_run(run_id)
        assert meta.scores["gsm8k"] == 0.85

    def test_compare_runs(self, tmp_path):
        import json
        from tinker_cookbook.eval.store import EvalStore
        from tinker_cookbook.eval.benchmarks._types import StoredTrajectory

        store = EvalStore(tmp_path / "eval_store")

        # Create two runs
        store.create_run(model_name="test", benchmarks=["gsm8k"], run_id="run_a",
                        checkpoint_name="step100")
        store.create_run(model_name="test", benchmarks=["gsm8k"], run_id="run_b",
                        checkpoint_name="step200")

        # Write trajectories with stable example_ids
        for run_id, rewards in [("run_a", [1.0, 0.0, 1.0]), ("run_b", [1.0, 1.0, 0.0])]:
            traj_dir = tmp_path / "eval_store" / "runs" / run_id / "gsm8k"
            traj_dir.mkdir(parents=True)
            with open(traj_dir / "trajectories.jsonl", "w") as f:
                for i, r in enumerate(rewards):
                    t = StoredTrajectory(
                        idx=i, benchmark="gsm8k", example_id=f"q_{i}",
                        reward=r, logs={"example_id": f"q_{i}"},
                    )
                    f.write(json.dumps(t.to_dict()) + "\n")

            # Write result
            with open(traj_dir / "result.json", "w") as f:
                json.dump({"score": sum(rewards) / len(rewards)}, f)

        store.finalize_run("run_a")
        store.finalize_run("run_b")

        comp = store.compare_runs("run_a", "run_b", "gsm8k")
        assert comp.num_shared == 3
        assert len(comp.regressions) == 1   # q_2: A correct, B wrong
        assert len(comp.improvements) == 1  # q_1: A wrong, B correct
        assert "q_2" in comp.regressions
        assert "q_1" in comp.improvements
