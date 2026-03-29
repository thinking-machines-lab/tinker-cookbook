"""Unit tests for the Nemotron-Cascade-2 eval system.

These tests verify:
  - EvalResult dataclass construction
  - Answer extraction functions (math, MCQ)
  - Each benchmark module can be imported
  - No Tinker API calls needed (all mocked)
"""

from __future__ import annotations

import pytest

from tinker_cookbook.recipes.nemotron_cascade.eval.base import EvalResult


# ---------------------------------------------------------------------------
# EvalResult dataclass
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_basic_construction(self):
        r = EvalResult(benchmark="test", score=0.85, num_examples=100, num_correct=85)
        assert r.benchmark == "test"
        assert r.score == 0.85
        assert r.num_examples == 100
        assert r.num_correct == 85
        assert r.metrics == {}
        assert r.examples == []

    def test_with_metrics_and_examples(self):
        r = EvalResult(
            benchmark="gsm8k",
            score=0.9,
            num_examples=50,
            num_correct=45,
            metrics={"gsm8k/accuracy": 0.9, "gsm8k/extra": 1.0},
            examples=[{"input": "q", "output": "a", "correct": True}],
        )
        assert r.metrics["gsm8k/accuracy"] == 0.9
        assert len(r.examples) == 1

    def test_zero_examples(self):
        r = EvalResult(benchmark="empty", score=0.0, num_examples=0, num_correct=0)
        assert r.score == 0.0


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------


class TestExtractBoxed:
    def test_simple(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_boxed

        assert extract_boxed(r"The answer is \boxed{42}") == "42"

    def test_nested_braces(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_boxed

        assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_no_boxed(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_boxed

        assert extract_boxed("no boxed here") is None

    def test_unclosed(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_boxed

        assert extract_boxed(r"\boxed{42") is None


class TestExtractNumber:
    def test_plain_number(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_number

        assert extract_number("42") == "42"

    def test_latex(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_number

        assert extract_number(r"\text{answer} 123") == "123"

    def test_commas(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_number

        assert extract_number("1,234") == "1234"

    def test_negative(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_number

        assert extract_number("-7") == "-7"


class TestExtractGsm8kAnswer:
    def test_boxed(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_gsm8k_answer

        assert extract_gsm8k_answer(r"blah \boxed{42} blah") == "42"

    def test_hash_format(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_gsm8k_answer

        assert extract_gsm8k_answer("step 1...\n#### 99") == "99"

    def test_answer_is(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_gsm8k_answer

        assert extract_gsm8k_answer("The answer is $150") == "150"

    def test_last_number(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_gsm8k_answer

        assert extract_gsm8k_answer("foo 10 bar 20 baz 30") == "30"

    def test_empty(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_gsm8k_answer

        assert extract_gsm8k_answer("no numbers here") == ""


class TestExtractMcqAnswer:
    def test_boxed_letter(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_mcq_answer

        assert extract_mcq_answer(r"\boxed{B}") == "B"

    def test_answer_is_pattern(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_mcq_answer

        assert extract_mcq_answer("The answer is (C)") == "C"

    def test_last_letter(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_mcq_answer

        assert extract_mcq_answer("Analyzing... A is wrong, B might work, but D is correct") == "D"

    def test_no_letter(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_mcq_answer

        assert extract_mcq_answer("I don't know") == ""

    def test_custom_valid_letters(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks._common import extract_mcq_answer

        assert extract_mcq_answer("The answer is F", "ABCDEFGHIJ") == "F"


# ---------------------------------------------------------------------------
# Benchmark module imports
# ---------------------------------------------------------------------------


class TestBenchmarkImports:
    """Verify that every benchmark module can be imported and has an ``evaluate`` callable."""

    def test_import_gsm8k(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import gsm8k
        assert callable(gsm8k.evaluate)

    def test_import_ifeval(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import ifeval
        assert callable(ifeval.evaluate)

    def test_import_mmlu_pro(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import mmlu_pro
        assert callable(mmlu_pro.evaluate)

    def test_import_math500(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import math500
        assert callable(math500.evaluate)

    def test_import_gpqa(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import gpqa
        assert callable(gpqa.evaluate)

    def test_import_aime(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import aime
        assert callable(aime.evaluate)

    def test_import_mbpp(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import mbpp
        assert callable(mbpp.evaluate)

    def test_import_longbench(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import longbench
        assert callable(longbench.evaluate)

    def test_import_livecodebench(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import livecodebench
        assert callable(livecodebench.evaluate)

    def test_import_mmlu_redux(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import mmlu_redux
        assert callable(mmlu_redux.evaluate)

    def test_import_arena_hard(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import arena_hard
        assert callable(arena_hard.evaluate)

    def test_import_bfcl(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import bfcl
        assert callable(bfcl.evaluate)

    def test_import_ifbench(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import ifbench
        assert callable(ifbench.evaluate)

    def test_import_swe_bench(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import swe_bench
        assert callable(swe_bench.evaluate)

    def test_import_tau2_bench(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import tau2_bench
        assert callable(tau2_bench.evaluate)

    def test_import_terminal_bench(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import terminal_bench
        assert callable(terminal_bench.evaluate)


class TestBenchmarksRegistry:
    """Verify the BENCHMARKS registry in the benchmarks __init__."""

    def test_registry_has_all_tier1(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import BENCHMARKS

        for name in ("gsm8k", "ifeval", "mmlu_pro", "math500", "gpqa", "aime", "mbpp", "longbench"):
            assert name in BENCHMARKS, f"Missing tier-1 benchmark: {name}"

    def test_registry_has_all_tier2(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import BENCHMARKS

        for name in ("livecodebench", "mmlu_redux", "arena_hard", "bfcl", "ifbench"):
            assert name in BENCHMARKS, f"Missing tier-2 benchmark: {name}"

    def test_registry_has_all_tier3(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import BENCHMARKS

        for name in ("swe_bench", "tau2_bench", "terminal_bench"):
            assert name in BENCHMARKS, f"Missing tier-3 benchmark: {name}"

    def test_all_entries_are_callable(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks import BENCHMARKS

        for name, fn in BENCHMARKS.items():
            assert callable(fn), f"BENCHMARKS['{name}'] is not callable"


# ---------------------------------------------------------------------------
# GSM8K grading helper
# ---------------------------------------------------------------------------


class TestGsm8kGrading:
    def test_exact_match(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.gsm8k import _check_gsm8k

        assert _check_gsm8k(r"The answer is \boxed{42}", "42") is True

    def test_numeric_tolerance(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.gsm8k import _check_gsm8k

        assert _check_gsm8k(r"\boxed{3.14}", "3.14") is True

    def test_wrong_answer(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.gsm8k import _check_gsm8k

        assert _check_gsm8k(r"\boxed{99}", "42") is False

    def test_comma_in_expected(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.gsm8k import _check_gsm8k

        assert _check_gsm8k(r"\boxed{1234}", "1,234") is True


# ---------------------------------------------------------------------------
# MBPP code extraction
# ---------------------------------------------------------------------------


class TestMbppCodeExtraction:
    def test_python_block(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.mbpp import _extract_python_code

        text = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
        assert "def add(a, b):" in _extract_python_code(text)

    def test_generic_block(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.mbpp import _extract_python_code

        text = "```\nprint('hello')\n```"
        assert "print('hello')" in _extract_python_code(text)

    def test_no_block(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.mbpp import _extract_python_code

        text = "def foo(): pass"
        assert _extract_python_code(text) == "def foo(): pass"


# ---------------------------------------------------------------------------
# BFCL function call extraction
# ---------------------------------------------------------------------------


class TestBfclExtraction:
    def test_json_extraction(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.bfcl import _extract_function_call

        text = 'I will call: {"name": "get_weather", "arguments": {"city": "NYC"}}'
        result = _extract_function_call(text)
        assert result is not None
        assert result["name"] == "get_weather"

    def test_code_block_json(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.bfcl import _extract_function_call

        text = '```json\n{"name": "search", "arguments": {"q": "test"}}\n```'
        result = _extract_function_call(text)
        assert result is not None
        assert result["name"] == "search"

    def test_no_function_call(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.bfcl import _extract_function_call

        assert _extract_function_call("I cannot call any function") is None


class TestBfclMatching:
    def test_exact_match(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.bfcl import _match_function_call

        gen = {"name": "get_weather", "arguments": {"city": "NYC"}}
        exp = {"name": "get_weather", "arguments": {"city": "NYC"}}
        assert _match_function_call(gen, exp) is True

    def test_case_insensitive_values(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.bfcl import _match_function_call

        gen = {"name": "get_weather", "arguments": {"city": "nyc"}}
        exp = {"name": "get_weather", "arguments": {"city": "NYC"}}
        assert _match_function_call(gen, exp) is True

    def test_wrong_function_name(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.bfcl import _match_function_call

        gen = {"name": "search", "arguments": {"q": "test"}}
        exp = {"name": "get_weather", "arguments": {"city": "NYC"}}
        assert _match_function_call(gen, exp) is False

    def test_missing_argument(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.bfcl import _match_function_call

        gen = {"name": "get_weather", "arguments": {}}
        exp = {"name": "get_weather", "arguments": {"city": "NYC"}}
        assert _match_function_call(gen, exp) is False


# ---------------------------------------------------------------------------
# Arena-Hard judge score extraction
# ---------------------------------------------------------------------------


class TestArenaHardScoreExtraction:
    def test_bracket_format(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.arena_hard import _extract_judge_score

        assert _extract_judge_score("Good response. Rating: [[8]]") == 8

    def test_rating_format(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.arena_hard import _extract_judge_score

        assert _extract_judge_score("Rating: 6") == 6

    def test_no_score(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.arena_hard import _extract_judge_score

        assert _extract_judge_score("I like this response") is None


# ---------------------------------------------------------------------------
# run_evals backward compat
# ---------------------------------------------------------------------------


class TestRunEvalsBackwardCompat:
    def test_aliases_resolve(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.run_evals import _resolve_benchmark_name

        assert _resolve_benchmark_name("mmlu") == "mmlu_pro"
        assert _resolve_benchmark_name("aime2025") == "aime"
        assert _resolve_benchmark_name("gsm8k") == "gsm8k"

    def test_run_eval_importable(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.run_evals import run_eval
        assert callable(run_eval)


# ---------------------------------------------------------------------------
# SWE-bench helpers
# ---------------------------------------------------------------------------


class TestSweBenchExtraction:
    def test_extract_patch_diff_block(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.swe_bench import _extract_patch

        text = "Here is the fix:\n```diff\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n```\n"
        result = _extract_patch(text)
        assert "--- a/foo.py" in result
        assert "+new" in result

    def test_extract_patch_generic_block(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.swe_bench import _extract_patch

        text = "```\nsome patch content\n```"
        assert _extract_patch(text) == "some patch content"

    def test_extract_patch_no_block(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.swe_bench import _extract_patch

        assert _extract_patch("raw patch text") == "raw patch text"

    def test_extract_judge_score_bracket(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.swe_bench import _extract_judge_score

        assert _extract_judge_score("The patch is good. Score: [[0.8]]") == 0.8

    def test_extract_judge_score_fallback(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.swe_bench import _extract_judge_score

        assert _extract_judge_score("Score: 0.6") == 0.6

    def test_extract_judge_score_none(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.swe_bench import _extract_judge_score

        assert _extract_judge_score("No score here") is None


# ---------------------------------------------------------------------------
# tau2-Bench helpers
# ---------------------------------------------------------------------------


class TestTau2BenchExtraction:
    def test_extract_action_json_block(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.tau2_bench import _extract_action

        text = '```json\n{"action": "refund_order", "arguments": {"order_id": "123"}}\n```'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "refund_order"
        assert result["arguments"]["order_id"] == "123"

    def test_extract_action_inline(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.tau2_bench import _extract_action

        text = 'I will perform: {"action": "cancel", "arguments": {}}'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "cancel"

    def test_extract_action_none(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.tau2_bench import _extract_action

        assert _extract_action("I cannot help with that") is None

    def test_normalize_action(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.tau2_bench import _normalize_action

        a = {"action": "Refund", "arguments": {"id": "ABC"}}
        b = {"action": "refund", "arguments": {"id": "abc"}}
        assert _normalize_action(a) == _normalize_action(b)


# ---------------------------------------------------------------------------
# Terminal-Bench helpers
# ---------------------------------------------------------------------------


class TestTerminalBenchExtraction:
    def test_extract_script_bash_block(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.terminal_bench import _extract_script

        text = "Here is the solution:\n```bash\nfind / -name '*.log' -delete\n```\n"
        assert "find" in _extract_script(text)

    def test_extract_script_generic_block(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.terminal_bench import _extract_script

        text = "```\nls -la\n```"
        assert _extract_script(text) == "ls -la"

    def test_extract_script_no_block(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.terminal_bench import _extract_script

        assert _extract_script("ls -la") == "ls -la"

    def test_extract_judge_verdict_correct(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.terminal_bench import _extract_judge_verdict

        assert _extract_judge_verdict("The solution is right. Verdict: [[CORRECT]]") is True

    def test_extract_judge_verdict_incorrect(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.terminal_bench import _extract_judge_verdict

        assert _extract_judge_verdict("Wrong approach. Verdict: [[INCORRECT]]") is False

    def test_extract_judge_verdict_fallback(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.terminal_bench import _extract_judge_verdict

        assert _extract_judge_verdict("Verdict: CORRECT") is True

    def test_extract_judge_verdict_none(self):
        from tinker_cookbook.recipes.nemotron_cascade.eval.benchmarks.terminal_bench import _extract_judge_verdict

        assert _extract_judge_verdict("No verdict given") is None
