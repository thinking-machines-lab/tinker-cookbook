"""Tests for the benchmark evaluation framework."""

import pytest

from tinker_cookbook.eval.benchmarks._runner import (
    _choose_k_values,
    _compute_pass_at_k,
    _compute_token_turn_summary,
    _pass_at_k_single,
)
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

    def test_score_completed(self):
        # 100 examples: 70 correct, 10 errors, 10 truncated, 10 wrong
        r = BenchmarkResult(
            name="test",
            score=0.7,
            num_examples=100,
            num_correct=70,
            num_errors=10,
            num_truncated=10,
        )
        # 80 completed (100 - 10 errors - 10 truncated), 70 correct
        assert abs(r.score_completed - 70 / 80) < 1e-9

    def test_score_completed_no_truncation(self):
        r = BenchmarkResult(name="test", score=0.8, num_examples=100, num_correct=80)
        assert abs(r.score_completed - 0.8) < 1e-9

    def test_score_completed_all_truncated(self):
        r = BenchmarkResult(
            name="test", score=0.0, num_examples=10, num_correct=0, num_truncated=10
        )
        assert r.score_completed == 0.0


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
        builder = type(
            "B",
            (BenchmarkBuilder,),
            {
                "name": "test",
                "make_envs": lambda self, r, c: [],
            },
        )()
        result = builder.aggregate([1.0, 0.0, 1.0, 1.0], [{}, {}, {}, {}])
        assert result.score == 0.75
        assert result.num_correct == 3
        assert result.num_examples == 4

    def test_empty(self):
        builder = type(
            "B",
            (BenchmarkBuilder,),
            {
                "name": "test",
                "make_envs": lambda self, r, c: [],
            },
        )()
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

    # Experimental benchmarks (_-prefixed modules)

    def test_bfcl_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks._bfcl", "bfcl")

    def test_longbench_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks._longbench", "longbench")

    def test_tau2_bench_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks._tau2_bench", "tau2_bench")

    def test_arena_hard_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks._arena_hard", "arena_hard")

    def test_swe_bench_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks._swe_bench", "swe_bench")

    def test_terminal_bench_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks._terminal_bench", "terminal_bench")

    def test_mbpp_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.mbpp", "mbpp")

    def test_livecodebench_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks._livecodebench", "livecodebench")

    def test_ceval_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.ceval", "ceval")

    def test_supergpqa_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks.supergpqa", "supergpqa")

    def test_hmmt_feb_2025_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks._hmmt", "hmmt_feb_2025")

    def test_hmmt_nov_2025_registered(self):
        self._check_registered("tinker_cookbook.eval.benchmarks._hmmt", "hmmt_nov_2025")


class TestHMMTGrading:
    """Test the HMMT LaTeX normalization and sympy-based grading."""

    def test_normalize_basic(self):
        from tinker_cookbook.eval.benchmarks._hmmt import _normalize_latex

        assert _normalize_latex("  42  ") == "42"
        assert _normalize_latex("$42$") == "42"
        assert _normalize_latex("\\frac{1}{2}") == "\\frac{1}{2}"

    def test_normalize_text_wrapper(self):
        from tinker_cookbook.eval.benchmarks._hmmt import _normalize_latex

        assert _normalize_latex("\\text{yes}") == "yes"
        assert _normalize_latex("\\mathrm{cm}") == "cm"

    def test_normalize_left_right(self):
        from tinker_cookbook.eval.benchmarks._hmmt import _normalize_latex

        assert _normalize_latex("\\left(x\\right)") == "(x)"

    def test_check_math_equal_string_match(self):
        from tinker_cookbook.eval.benchmarks._hmmt import _check_math_equal

        assert _check_math_equal("42", "42")
        assert not _check_math_equal("42", "43")

    def test_check_math_equal_numeric(self):
        from tinker_cookbook.eval.benchmarks._hmmt import _check_math_equal

        assert _check_math_equal("0.5", "0.5")
        assert _check_math_equal("103", "103")

    def test_check_math_equal_sympy_fraction(self):
        from tinker_cookbook.eval.benchmarks._hmmt import _check_math_equal

        # These require antlr4-python3-runtime for sympy.parsing.latex
        try:
            from sympy.parsing.latex import parse_latex

            parse_latex("1")  # test if it actually works
        except (ImportError, Exception):
            pytest.skip("antlr4-python3-runtime not installed")
        assert _check_math_equal("\\frac{1}{2}", "0.5")
        assert _check_math_equal("\\frac{3}{4}", "0.75")

    def test_check_math_equal_sympy_expression(self):
        from tinker_cookbook.eval.benchmarks._hmmt import _check_math_equal

        # String-normalized matches work without antlr4
        assert _check_math_equal("\\frac{1}{576}", "\\frac{1}{576}")
        assert _check_math_equal("2\\sqrt{3}", "2\\sqrt{3}")

    def test_check_math_equal_empty(self):
        from tinker_cookbook.eval.benchmarks._hmmt import _check_math_equal

        assert not _check_math_equal("", "42")
        assert not _check_math_equal("42", "")


class TestMCQExtraction:
    """Test answer extraction functions from MCQ benchmarks."""

    def test_mmlu_pro_extract(self):
        from tinker_cookbook.eval.benchmarks.mmlu_pro import extract_mcq_answer

        assert extract_mcq_answer("The answer is (B).", "ABCDEFGHIJ") == "B"
        assert extract_mcq_answer("\\boxed{C}", "ABCDEFGHIJ") == "C"
        assert extract_mcq_answer("I think D is correct.", "ABCDEFGHIJ") == "D"

    def test_bfcl_function_matching(self):
        from tinker_cookbook.eval.benchmarks._bfcl import _match_function_call

        gen = {"name": "get_weather", "arguments": {"city": "London"}}
        exp = {"name": "get_weather", "arguments": {"city": "london"}}
        assert _match_function_call(gen, exp)

        gen_wrong = {"name": "get_weather", "arguments": {"city": "Paris"}}
        assert not _match_function_call(gen_wrong, exp)

    def test_tau2_bench_action_matching(self):
        from tinker_cookbook.eval.benchmarks._tau2_bench import _check_actions

        predicted = [
            {"name": "get_user_details", "arguments": {"user_id": "alice_123"}},
            {"name": "get_reservation_details", "arguments": {"reservation_id": "ABC"}},
        ]
        expected = [
            {"name": "get_user_details", "arguments": {"user_id": "alice_123"}},
            {"name": "get_reservation_details", "arguments": {"reservation_id": "ABC"}},
        ]
        score, metrics = _check_actions(predicted, expected)
        assert score == 1.0
        assert metrics["actions_matched"] == 2


class TestGradingConsistency:
    """Validate that new framework grading matches recipe implementation."""

    def test_gsm8k_extraction_matches_recipe(self):
        recipe_common = pytest.importorskip(
            "tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common"
        )
        old = recipe_common.extract_gsm8k_answer
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
        recipe_gsm8k = pytest.importorskip(
            "tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.gsm8k"
        )
        old = recipe_gsm8k._check_gsm8k
        from tinker_cookbook.eval.benchmarks.gsm8k import check_gsm8k as new

        cases = [
            ("The answer is 42", "42", True),
            ("\\boxed{42}", "42", True),
            ("The answer is 43", "42", False),
            ("#### 0", "0", True),
        ]
        for resp, expected, _should_be in cases:
            assert old(resp, expected) == new(resp, expected), f"Mismatch: {resp} vs {expected}"

    def test_mcq_extraction_matches_recipe(self):
        recipe_common = pytest.importorskip(
            "tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common"
        )
        old = recipe_common.extract_mcq_answer
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
        from tinker_cookbook.stores.eval_store import EvalStore

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
        from tinker_cookbook.stores.eval_store import EvalStore

        store = EvalStore(tmp_path / "eval_store")
        run_id = store.create_run(
            model_name="test",
            benchmarks=["gsm8k"],
            run_id="my_run",
        )
        assert run_id == "my_run"
        assert "my_run" in store.run_dir(run_id)

    def test_finalize_run(self, tmp_path):
        import json

        from tinker_cookbook.stores.eval_store import EvalStore

        store = EvalStore(tmp_path / "eval_store")
        run_id = store.create_run(
            model_name="test",
            benchmarks=["gsm8k"],
            run_id="test_run",
        )

        # Simulate a result file
        result_dir = tmp_path / "eval_store" / "runs" / "test_run" / "gsm8k"
        result_dir.mkdir(parents=True)
        with open(result_dir / "result.json", "w") as f:
            json.dump({"score": 0.85, "num_correct": 85, "num_examples": 100}, f)

        meta = store.finalize_run(run_id)
        assert meta.scores["gsm8k"] == 0.85


class TestConfigValidation:
    def test_rejects_zero_concurrency(self):
        with pytest.raises(ValueError):
            BenchmarkConfig(concurrency=0)

    def test_rejects_negative_timeout(self):
        with pytest.raises(ValueError):
            BenchmarkConfig(timeout_seconds=-1)

    def test_rejects_zero_num_samples(self):
        with pytest.raises(ValueError):
            BenchmarkConfig(num_samples=0)

    def test_rejects_negative_context_window(self):
        with pytest.raises(ValueError):
            BenchmarkConfig(context_window=-1)

    def test_accepts_valid_config(self):
        c = BenchmarkConfig(concurrency=32, num_samples=5)
        assert c.concurrency == 32
        assert c.num_samples == 5


class TestPassAtK:
    def test_pass_at_k_single_all_correct(self):
        assert _pass_at_k_single(5, 5, 1) == 1.0

    def test_pass_at_k_single_none_correct(self):
        assert _pass_at_k_single(5, 0, 1) == 0.0

    def test_pass_at_k_single_partial(self):
        result = _pass_at_k_single(10, 3, 1)
        assert abs(result - 0.3) < 1e-9

    def test_pass_at_k_single_k_greater_than_n_minus_c(self):
        assert _pass_at_k_single(5, 4, 2) == 1.0

    def test_compute_pass_at_k(self):
        per_example = {
            "a": [1.0, 0.0, 0.0, 0.0, 0.0],  # 1 correct out of 5
            "b": [1.0, 1.0, 1.0, 1.0, 1.0],  # 5 correct out of 5
        }
        result = _compute_pass_at_k(per_example, [1, 5])
        # pass@1: mean of _pass_at_k_single(5,1,1)=0.2 and _pass_at_k_single(5,5,1)=1.0 => 0.6
        assert abs(result[1] - 0.6) < 1e-9
        # pass@5: mean of _pass_at_k_single(5,1,5)=1.0 and _pass_at_k_single(5,5,5)=1.0 => 1.0
        assert abs(result[5] - 1.0) < 1e-9

    def test_choose_k_values(self):
        assert _choose_k_values(10) == [1, 5, 10]

    # --- Edge cases for pass@k correctness ---

    def test_pass_at_k_single_n_equals_1(self):
        """Degenerate case: single sample."""
        assert _pass_at_k_single(1, 1, 1) == 1.0
        assert _pass_at_k_single(1, 0, 1) == 0.0

    def test_pass_at_k_single_k_equals_n(self):
        """When k == n, pass@k == 1.0 iff c >= 1."""
        assert _pass_at_k_single(5, 1, 5) == 1.0
        assert _pass_at_k_single(5, 0, 5) == 0.0

    def test_pass_at_k_single_large_k(self):
        """Codex formula: pass@k with k=10, n=10, c=5."""
        # C(5, 10) / C(10, 10) — but n-c=5 < k=10, so returns 1.0
        assert _pass_at_k_single(10, 5, 10) == 1.0
        # n-c=8 >= k=3, so 1 - C(8,3)/C(10,3) = 1 - 56/120 = 0.5333...
        result = _pass_at_k_single(10, 2, 3)
        assert abs(result - (1.0 - 56 / 120)) < 1e-9

    def test_errors_counted_as_incorrect_in_pass_at_k(self):
        """Errors (reward=0.0) should count as incorrect samples, not be excluded.

        This is the key semantic: an errored sample lowers n's effective
        correctness rate, making pass@k lower — same as a wrong answer.
        """
        # 2 correct, 3 errors (reward=0.0) out of 5 samples
        per_example = {"a": [1.0, 0.0, 1.0, 0.0, 0.0]}
        result = _compute_pass_at_k(per_example, [1])
        # pass@1 = _pass_at_k_single(5, 2, 1) = 0.4
        assert abs(result[1] - 0.4) < 1e-9

    def test_compute_pass_at_k_empty(self):
        """Empty input returns empty dict."""
        assert _compute_pass_at_k({}, [1, 5]) == {}

    def test_compute_pass_at_k_skips_k_greater_than_n(self):
        """Examples with fewer samples than k are excluded from that k."""
        per_example = {
            "a": [1.0, 0.0, 0.0],  # n=3
            "b": [0.0],  # n=1
        }
        result = _compute_pass_at_k(per_example, [1, 3, 5])
        # k=1: both examples contribute — mean of pass@1(3,1,1)=1/3 and pass@1(1,0,1)=0.0
        assert abs(result[1] - 1 / 6) < 1e-9
        # k=3: only "a" contributes — pass@3(3,1,3) = 1.0
        assert abs(result[3] - 1.0) < 1e-9
        # k=5: neither has enough samples
        assert 5 not in result

    def test_compute_pass_at_k_uniform_samples(self):
        """All examples have the same number of samples — standard case."""
        per_example = {
            "a": [1.0, 1.0, 0.0, 0.0],  # c=2, n=4
            "b": [0.0, 0.0, 0.0, 0.0],  # c=0, n=4
            "c": [1.0, 1.0, 1.0, 1.0],  # c=4, n=4
        }
        result = _compute_pass_at_k(per_example, [1, 4])
        # pass@1: mean of 0.5, 0.0, 1.0 = 0.5
        assert abs(result[1] - 0.5) < 1e-9
        # pass@4: mean of 1.0, 0.0, 1.0 = 2/3
        assert abs(result[4] - 2 / 3) < 1e-9

    def test_choose_k_values_small(self):
        """num_samples=1 should return [1]."""
        assert _choose_k_values(1) == [1]

    def test_choose_k_values_non_standard(self):
        """num_samples=4 should return [1, 4] (4 is not a standard candidate)."""
        assert _choose_k_values(4) == [1, 4]

    def test_choose_k_values_large(self):
        """num_samples=200 should include all standard values plus 200."""
        assert _choose_k_values(200) == [1, 5, 10, 25, 50, 100, 200]

    def test_choose_k_values_exact_candidate(self):
        """num_samples matching a standard candidate shouldn't duplicate it."""
        assert _choose_k_values(50) == [1, 5, 10, 25, 50]


class TestTokenTurnSummary:
    """Tests for _compute_token_turn_summary."""

    def test_basic(self):
        result = _compute_token_turn_summary(
            [
                {"_eval_turns": 2, "_eval_ac_tokens": 100, "_eval_ob_tokens": 50},
                {"_eval_turns": 3, "_eval_ac_tokens": 150, "_eval_ob_tokens": 75},
            ]
        )
        assert result["total_ac_tokens"] == 250
        assert result["total_ob_tokens"] == 125
        assert result["total_turns"] == 5
        assert result["turns_per_episode"] == 2.5
        assert result["ac_tokens_per_turn"] == 50.0
        assert result["ob_tokens_per_turn"] == 25.0

    def test_empty_list(self):
        result = _compute_token_turn_summary([])
        assert result["total_turns"] == 0
        assert result["turns_per_episode"] == 0
        assert "ac_tokens_per_turn" not in result

    def test_missing_keys_default_to_zero(self):
        result = _compute_token_turn_summary(
            [
                {"some_other_metric": 1.0},
                {"_eval_turns": 1, "_eval_ac_tokens": 10, "_eval_ob_tokens": 5},
            ]
        )
        assert result["total_ac_tokens"] == 10
        assert result["total_turns"] == 1

    def test_zero_turns(self):
        result = _compute_token_turn_summary(
            [
                {"_eval_turns": 0, "_eval_ac_tokens": 0, "_eval_ob_tokens": 0},
            ]
        )
        assert result["total_turns"] == 0
        assert "ac_tokens_per_turn" not in result
