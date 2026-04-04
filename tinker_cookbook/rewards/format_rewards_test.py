"""Tests for format reward functions."""

import pytest

from tinker_cookbook.rewards.format_rewards import (
    extract_after_prefix,
    extract_xml_tag,
    format_reward,
    has_answer_prefix,
    has_boxed_answer,
    has_code_block,
    has_xml_tag,
    is_valid_json,
)


class TestHasBoxedAnswer:
    def test_present(self):
        assert has_boxed_answer("The answer is \\boxed{42}") is True

    def test_absent(self):
        assert has_boxed_answer("no boxed answer") is False


class TestHasCodeBlock:
    def test_present(self):
        assert has_code_block("```python\nprint('hi')\n```") is True

    def test_absent(self):
        assert has_code_block("just text") is False

    def test_no_language(self):
        assert has_code_block("```\ncode\n```") is True


class TestHasXmlTag:
    def test_present(self):
        assert has_xml_tag("<answer>42</answer>", "answer") is True

    def test_absent(self):
        assert has_xml_tag("no tags here", "answer") is False

    def test_multiline(self):
        assert has_xml_tag("<answer>\n42\n</answer>", "answer") is True

    def test_wrong_tag(self):
        assert has_xml_tag("<score>5</score>", "answer") is False


class TestExtractXmlTag:
    def test_simple(self):
        assert extract_xml_tag("<answer>42</answer>", "answer") == "42"

    def test_missing(self):
        assert extract_xml_tag("no tags", "answer") is None

    def test_multiple_returns_last(self):
        assert extract_xml_tag("<x>1</x> and <x>2</x>", "x") == "2"

    def test_strips_whitespace(self):
        assert extract_xml_tag("<x>  hello  </x>", "x") == "hello"


class TestIsValidJson:
    def test_valid_object(self):
        assert is_valid_json('{"key": "value"}') is True

    def test_valid_array(self):
        assert is_valid_json("[1, 2, 3]") is True

    def test_invalid(self):
        assert is_valid_json("not json") is False

    def test_empty_string(self):
        assert is_valid_json("") is False


class TestHasAnswerPrefix:
    def test_present(self):
        assert has_answer_prefix("Answer: 42") is True

    def test_absent(self):
        assert has_answer_prefix("The result is 42") is False

    def test_custom_prefix(self):
        assert has_answer_prefix("Result: 42", prefix="Result:") is True


class TestExtractAfterPrefix:
    def test_simple(self):
        assert extract_after_prefix("Answer: 42") == "42"

    def test_missing(self):
        assert extract_after_prefix("no prefix here") is None

    def test_multiple_occurrences(self):
        assert extract_after_prefix("Answer: first Answer: second") is None


class TestFormatReward:
    def test_boxed_pass(self):
        assert format_reward("\\boxed{42}", check_fn="boxed") == 0.0

    def test_boxed_fail(self):
        assert format_reward("no boxed", check_fn="boxed") == pytest.approx(-0.1)

    def test_code_block_pass(self):
        assert format_reward("```python\ncode\n```", check_fn="code_block") == 0.0

    def test_json_pass(self):
        assert format_reward('{"a": 1}', check_fn="json") == 0.0

    def test_custom_coef(self):
        assert format_reward("no boxed", check_fn="boxed", format_coef=0.5) == pytest.approx(-0.5)

    def test_unknown_check_fn(self):
        with pytest.raises(ValueError, match="Unknown check_fn"):
            format_reward("text", check_fn="unknown")
