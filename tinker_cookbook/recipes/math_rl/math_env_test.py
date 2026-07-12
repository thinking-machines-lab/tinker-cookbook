"""Unit tests for math grading, answer extraction, and environment logic.

These tests exercise pure functions with no network or API dependencies.
The MathDatasetBuilder integration test (which downloads from HuggingFace)
lives in tests/integration/test_math_dataset_builder.py.
"""

from typing import Any, cast

import pytest

from tinker_cookbook.recipes.math_rl.math_env import (
    DeepMathDataset,
    Gsm8kDataset,
    MathDataset,
    PolarisDataset,
    _coerce_difficulty,
    _problem_row_metadata,
    extract_gsm8k_final_answer,
)
from tinker_cookbook.recipes.math_rl.math_grading import (
    extract_boxed,
    grade_answer,
    normalize_answer,
    split_tuple,
)


def _dataset_stub(cls: Any, split: str = "train") -> Any:
    """Build a dataset instance without loading the HF dataset."""
    ds = cls.__new__(cls)
    ds.renderer = cast(Any, None)
    ds.convo_prefix = None
    ds.batch_size = 1
    ds.group_size = 4
    ds.split = split
    return ds


class TestExtractBoxed:
    def test_simple(self):
        assert extract_boxed("The answer is \\boxed{42}") == "42"

    def test_nested_braces(self):
        assert extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_multiple_boxed_returns_last(self):
        assert extract_boxed("\\boxed{1} then \\boxed{2}") == "2"

    def test_boxed_without_braces(self):
        assert extract_boxed("\\boxed 7") == "7"

    def test_no_boxed_raises(self):
        with pytest.raises(ValueError, match="No boxed"):
            extract_boxed("no answer here")

    def test_empty_boxed(self):
        assert extract_boxed("\\boxed{}") == ""

    def test_boxed_with_latex(self):
        assert extract_boxed("\\boxed{x^2 + 1}") == "x^2 + 1"

    def test_deeply_nested(self):
        assert extract_boxed("\\boxed{\\sqrt{\\frac{3}{4}}}") == "\\sqrt{\\frac{3}{4}}"


class TestExtractGsm8kFinalAnswer:
    def test_standard_format(self):
        text = "Some reasoning.\n#### 42"
        assert extract_gsm8k_final_answer(text) == "42"

    def test_with_commas_stripped(self):
        text = "Calculation.\n#### 1,234"
        assert extract_gsm8k_final_answer(text) == "1234"

    def test_with_colon(self):
        text = "####: 55"
        assert extract_gsm8k_final_answer(text) == "55"

    def test_multiple_lines_returns_last(self):
        text = "#### 10\nMore reasoning.\n#### 20"
        assert extract_gsm8k_final_answer(text) == "20"

    def test_no_answer_raises(self):
        with pytest.raises(ValueError, match="No GSM8K final answer"):
            extract_gsm8k_final_answer("no hash marks here")

    def test_inline_format(self):
        text = "The total is #### 99 dollars."
        assert extract_gsm8k_final_answer(text) == "99 dollars."


class TestNormalizeAnswer:
    def test_strips_whitespace(self):
        assert normalize_answer("  42  ") == "42"

    def test_removes_text_wrapper(self):
        assert normalize_answer("\\text{hello}") == "hello"

    def test_none_returns_none(self):
        assert normalize_answer(None) is None

    def test_fraction_normalization(self):
        result = normalize_answer("\\frac12")
        assert result is not None
        assert "frac" in result
        assert "{1}" in result
        assert "{2}" in result

    def test_removes_dollars(self):
        result = normalize_answer("\\$100")
        assert result is not None
        assert "\\$" not in result

    def test_removes_percent(self):
        result = normalize_answer("50\\%")
        assert result is not None
        assert "%" not in result


class TestGradeAnswer:
    def test_exact_match(self):
        assert grade_answer("42", "42") is True

    def test_none_given(self):
        assert grade_answer(None, "42") is False  # type: ignore[arg-type]

    def test_wrong_answer(self):
        assert grade_answer("43", "42") is False

    def test_latex_fraction_vs_decimal(self):
        # The grader normalizes \frac{1}{2} and 0.5 as equivalent (special case)
        assert grade_answer("\\frac{1}{2}", "0.5") is True
        # \frac{3}{4} vs 0.75 is caught by the sympy fallback
        assert grade_answer("\\frac{3}{4}", "0.75") is True

    def test_integer_mismatch_strict(self):
        # If ground truth is an integer, given answer must also be an integer
        assert grade_answer("5.0", "5") is True
        assert grade_answer("5.1", "5") is False

    def test_sympy_equivalence(self):
        # Sympy-based equivalence only applies when both sides are non-integer
        assert grade_answer("\\frac{2}{4}", "\\frac{1}{2}") is False

    def test_tuple_matching(self):
        assert grade_answer("(1, 2)", "(1, 2)") is True
        assert grade_answer("(1, 2)", "(2, 1)") is False

    def test_with_text_wrapper(self):
        assert grade_answer("\\text{yes}", "yes") is True


class TestCoerceDifficulty:
    def test_numeric_passthrough(self):
        assert _coerce_difficulty(3) == 3
        assert _coerce_difficulty(4.5) == 4.5

    def test_hendrycks_level_string(self):
        assert _coerce_difficulty("Level 3") == 3

    def test_decimal_string(self):
        assert _coerce_difficulty("difficulty 2.5") == 2.5

    def test_non_numeric_string_kept(self):
        assert _coerce_difficulty("Level ?") == "Level ?"


class TestBuilderMetadata:
    """Group metadata wired through ProblemGroupBuilder for token DB capture."""

    def test_math_row_metadata(self):
        row = {"problem": "1+1?", "solution": "\\boxed{2}", "level": "Level 3", "type": "Algebra"}
        builder = _dataset_stub(MathDataset)._make_env_group_builder(row, 4, 12)
        assert builder is not None
        assert builder.metadata() == {"dataset": "math", "row_id": "math-train-12", "level": 3}

    def test_math500_unique_id_wins_over_row_index(self):
        row = {
            "problem": "p",
            "solution": "\\boxed{2}",
            "unique_id": "test/algebra/1.json",
            "level": 4,
        }
        builder = _dataset_stub(MathDataset, split="test")._make_env_group_builder(row, 1, 0)
        assert builder is not None
        assert builder.metadata() == {
            "dataset": "math",
            "row_id": "test/algebra/1.json",
            "level": 4,
        }

    def test_gsm8k_row_metadata(self):
        row = {"question": "How many?", "answer": "reasoning\n#### 42"}
        builder = _dataset_stub(Gsm8kDataset)._make_env_group_builder(row, 4, 3)
        assert builder is not None
        assert builder.metadata() == {"dataset": "gsm8k", "row_id": "gsm8k-train-3"}

    def test_deepmath_difficulty_kept_numeric(self):
        row = {"question": "q", "final_answer": "2", "difficulty": 4.5}
        builder = _dataset_stub(DeepMathDataset)._make_env_group_builder(row, 4, 9)
        assert builder is not None
        assert builder.metadata() == {
            "dataset": "deepmath",
            "row_id": "deepmath-train-9",
            "difficulty": 4.5,
        }
        assert builder.logging_tags() == ["deepmath"]

    def test_polaris_without_difficulty(self):
        row = {"problem": "q", "answer": "2"}
        builder = _dataset_stub(PolarisDataset)._make_env_group_builder(row, 4, 1)
        assert builder is not None
        assert builder.metadata() == {"dataset": "polaris", "row_id": "polaris-train-1"}

    def test_arithmetic_metadata(self):
        import numpy as np

        from tinker_cookbook.recipes.math_rl.arithmetic_env import ArithmeticDataset

        dataset = ArithmeticDataset(
            batch_size=1, renderer=cast(Any, None), group_size=2, n_batches=1
        )
        rng = np.random.RandomState(0)
        builder = dataset._make_env_group_builder(rng)
        meta = dict(builder.metadata())
        assert meta["dataset"] == "arithmetic"
        assert str(meta["row_id"]).startswith("arithmetic-")

    def test_problem_row_metadata_skips_absent_fields(self):
        meta = _problem_row_metadata("math", "train", 5, {"problem": "p"})
        assert meta == {"dataset": "math", "row_id": "math-train-5"}


class TestCaptureEndToEnd:
    """A MathEnv rollout captured to parquet carries the routed dimensions."""

    def test_math_rollout_rows_carry_metadata(self, tmp_path):
        pytest.importorskip("pyarrow")
        import asyncio
        from unittest.mock import MagicMock

        import tinker

        from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
        from tinker_cookbook.renderers.base import ParseTermination
        from tinker_cookbook.rl.rollout_logging import RolloutSummaryGroup
        from tinker_cookbook.rl.rollouts import do_single_rollout
        from tinker_cookbook.rl.types import TrajectoryGroup
        from tinker_cookbook.tokendb.capture import record_groups
        from tinker_cookbook.tokendb.writer import TokenDbWriter
        from tinker_cookbook.tokendb.writer_test import read_all_segments

        renderer = MagicMock()
        renderer.build_generation_prompt = MagicMock(
            return_value=tinker.ModelInput.from_ints([1, 2, 3])
        )
        renderer.get_stop_sequences = MagicMock(return_value=["\n"])
        renderer.parse_response = MagicMock(
            return_value=(
                {"role": "assistant", "content": "\\boxed{2}"},
                ParseTermination.STOP_SEQUENCE,
            )
        )

        dataset = _dataset_stub(MathDataset)
        dataset.renderer = renderer
        row = {"problem": "1+1?", "solution": "\\boxed{2}", "level": "Level 3"}
        builder = dataset._make_env_group_builder(row, 1, 7)
        assert builder is not None
        envs = asyncio.run(builder.make_envs())

        class FixedPolicy(TokenCompleter):
            async def __call__(
                self, model_input: tinker.ModelInput, stop: StopCondition
            ) -> TokensWithLogprobs:
                return TokensWithLogprobs(tokens=[7], maybe_logprobs=[-0.1])

        trajectory = asyncio.run(do_single_rollout(FixedPolicy(), envs[0]))
        group = TrajectoryGroup(trajectories_G=[trajectory], final_rewards_G=[0.0], metrics_G=[{}])
        with TokenDbWriter(tmp_path, flush_interval_s=3600.0) as writer:
            record_groups(
                writer,
                [
                    RolloutSummaryGroup(
                        trajectory_group=group,
                        tags=list(builder.logging_tags()),
                        metadata=builder.metadata(),
                    )
                ],
                split="train",
                iteration=0,
            )
        got = read_all_segments(tmp_path).to_pylist()[0]
        assert dict(got["attrs"]) == {"dataset": "math"}
        metrics = dict(got["metrics"])
        assert metrics["level"] == 3.0
        assert metrics["correct"] == 1.0
        assert got["env_row_id"] == "math-train-7"


class TestSplitTuple:
    def test_single_value(self):
        assert split_tuple("42") == ["42"]

    def test_parenthesized_tuple(self):
        assert split_tuple("(1, 2, 3)") == ["1", "2", "3"]

    def test_bracketed(self):
        assert split_tuple("[1, 2]") == ["1", "2"]

    def test_empty(self):
        assert split_tuple("") == []

    def test_comma_in_number(self):
        assert split_tuple("1,000") == ["1000"]
