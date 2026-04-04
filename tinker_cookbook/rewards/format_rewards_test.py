"""Tests for format reward functions."""

import pytest

from tinker_cookbook.rewards.format_rewards import (
    check_has_answer_prefix,
    check_has_boxed,
    check_has_code_block,
    check_has_xml_tag,
    check_is_valid_json,
    extract_after_prefix,
    extract_xml_content,
    score_format,
)


class TestCheckHasBoxed:
    def test_present(self):
        assert check_has_boxed("The answer is \\boxed{42}") is True

    def test_absent(self):
        assert check_has_boxed("no boxed answer") is False


class TestCheckHasCodeBlock:
    def test_present(self):
        assert check_has_code_block("```python\nprint('hi')\n```") is True

    def test_absent(self):
        assert check_has_code_block("just text") is False

    def test_no_language(self):
        assert check_has_code_block("```\ncode\n```") is True


class TestCheckHasXmlTag:
    def test_present(self):
        assert check_has_xml_tag("<answer>42</answer>", "answer") is True

    def test_absent(self):
        assert check_has_xml_tag("no tags here", "answer") is False

    def test_multiline(self):
        assert check_has_xml_tag("<answer>\n42\n</answer>", "answer") is True

    def test_wrong_tag(self):
        assert check_has_xml_tag("<score>5</score>", "answer") is False


class TestExtractXmlContent:
    def test_simple(self):
        assert extract_xml_content("<answer>42</answer>", "answer") == "42"

    def test_missing(self):
        assert extract_xml_content("no tags", "answer") is None

    def test_multiple_returns_last(self):
        assert extract_xml_content("<x>1</x> and <x>2</x>", "x") == "2"

    def test_strips_whitespace(self):
        assert extract_xml_content("<x>  hello  </x>", "x") == "hello"


class TestCheckIsValidJson:
    def test_valid_object(self):
        assert check_is_valid_json('{"key": "value"}') is True

    def test_valid_array(self):
        assert check_is_valid_json("[1, 2, 3]") is True

    def test_invalid(self):
        assert check_is_valid_json("not json") is False

    def test_empty_string(self):
        assert check_is_valid_json("") is False


class TestCheckHasAnswerPrefix:
    def test_present(self):
        assert check_has_answer_prefix("Answer: 42") is True

    def test_absent(self):
        assert check_has_answer_prefix("The result is 42") is False

    def test_custom_prefix(self):
        assert check_has_answer_prefix("Result: 42", prefix="Result:") is True


class TestExtractAfterPrefix:
    def test_simple(self):
        assert extract_after_prefix("Answer: 42") == "42"

    def test_missing(self):
        assert extract_after_prefix("no prefix here") is None

    def test_multiple_occurrences(self):
        assert extract_after_prefix("Answer: first Answer: second") is None


class TestScoreFormat:
    def test_boxed_pass(self):
        assert score_format("\\boxed{42}", check_fn="boxed") == 0.0

    def test_boxed_fail(self):
        assert score_format("no boxed", check_fn="boxed") == pytest.approx(-0.1)

    def test_code_block_pass(self):
        assert score_format("```python\ncode\n```", check_fn="code_block") == 0.0

    def test_json_pass(self):
        assert score_format('{"a": 1}', check_fn="json") == 0.0

    def test_custom_coef(self):
        assert score_format("no boxed", check_fn="boxed", format_coef=0.5) == pytest.approx(-0.5)

    def test_unknown_check_fn(self):
        with pytest.raises(ValueError, match="Unknown check_fn"):
            score_format("text", check_fn="unknown")
