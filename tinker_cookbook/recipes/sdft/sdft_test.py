"""Unit tests for SDFT recipe (no API key needed)."""

import math

import pytest

from tinker_cookbook.distillation.sdft import (
    DEFAULT_DEMO_TEMPLATE,
    build_sdft_teacher_prompt,
)
from tinker_cookbook.recipes.sdft.datasets import SDFTDataset, _format_sciknoweval_choices
from tinker_cookbook.recipes.sdft.eval import (
    evaluate_science_correctness,
    evaluate_tooluse_correctness,
    extract_action_inputs,
    extract_actions,
    extract_xml_answer,
)
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER_NAME = "qwen3_instruct"


@pytest.fixture
def renderer():
    tokenizer = get_tokenizer(MODEL_NAME)
    return get_renderer(RENDERER_NAME, tokenizer=tokenizer)


@pytest.fixture
def tokenizer():
    return get_tokenizer(MODEL_NAME)


class TestBuildSDFTTeacherPrompt:
    def test_basic(self, renderer, tokenizer):
        prompt = build_sdft_teacher_prompt(
            question="What is the capital of France?",
            golden_answer="Paris",
            renderer=renderer,
        )
        assert prompt.length > 0
        decoded = tokenizer.decode(prompt.to_ints())
        assert "What is the capital of France?" in decoded
        assert "Paris" in decoded
        assert "example for a response" in decoded

    def test_with_system_prompt(self, renderer, tokenizer):
        prompt = build_sdft_teacher_prompt(
            question="What is 2+2?",
            golden_answer="4",
            renderer=renderer,
            system_prompt="You are a math tutor.",
        )
        decoded = tokenizer.decode(prompt.to_ints())
        assert "math tutor" in decoded
        assert "What is 2+2?" in decoded

    def test_custom_template(self, renderer, tokenizer):
        custom_template = "Q: {question}\nDemo: {golden_answer}\nNow answer:"
        prompt = build_sdft_teacher_prompt(
            question="Hello",
            golden_answer="World",
            renderer=renderer,
            demo_template=custom_template,
        )
        decoded = tokenizer.decode(prompt.to_ints())
        assert "Q: Hello" in decoded
        assert "Demo: World" in decoded

    def test_default_template_has_both_placeholders(self):
        formatted = DEFAULT_DEMO_TEMPLATE.format(question="Q", golden_answer="A")
        assert "Q" in formatted
        assert "A" in formatted


class TestSDFTDataset:
    def test_get_batch(self, renderer):
        questions = ["q1", "q2", "q3", "q4"]
        golden_answers = ["a1", "a2", "a3", "a4"]
        dataset = SDFTDataset(
            questions=questions,
            golden_answers=golden_answers,
            batch_size=2,
            group_size=1,
            renderer=renderer,
        )
        builders, batch_q, batch_a = dataset.get_batch(0)
        assert len(builders) == 2
        assert len(batch_q) == 2
        assert len(batch_a) == 2
        assert batch_q == ["q1", "q2"]
        assert batch_a == ["a1", "a2"]

    def test_get_batch_second(self, renderer):
        dataset = SDFTDataset(
            questions=["q1", "q2", "q3"],
            golden_answers=["a1", "a2", "a3"],
            batch_size=2,
            group_size=1,
            renderer=renderer,
        )
        builders, batch_q, batch_a = dataset.get_batch(1)
        assert len(builders) == 1
        assert batch_q == ["q3"]
        assert batch_a == ["a3"]

    def test_len(self, renderer):
        dataset = SDFTDataset(
            questions=["q1", "q2", "q3"],
            golden_answers=["a1", "a2", "a3"],
            batch_size=2,
            group_size=1,
            renderer=renderer,
        )
        assert len(dataset) == math.ceil(3 / 2)

    def test_mismatched_lengths_raises(self, renderer):
        with pytest.raises(AssertionError):
            SDFTDataset(
                questions=["q1", "q2"],
                golden_answers=["a1"],
                batch_size=1,
                group_size=1,
                renderer=renderer,
            )


class TestFormatSciKnowEvalChoices:
    def test_basic(self):
        choices = {"label": ["A", "B", "C"], "text": ["Apple", "Banana", "Cherry"]}
        result = _format_sciknoweval_choices(choices)
        assert "A: Apple" in result
        assert "B: Banana" in result
        assert "C: Cherry" in result

    def test_empty(self):
        result = _format_sciknoweval_choices({"label": [], "text": []})
        assert result == ""


class TestScienceEval:
    def test_extract_xml_answer(self):
        assert extract_xml_answer("Some reasoning <answer>B</answer> extra") == "B"
        assert extract_xml_answer("<answer>C</answer>") == "C"
        assert extract_xml_answer("no tags here") == "no tags here"

    def test_evaluate_science_correctness(self):
        responses = [
            "Let me think... <answer>A</answer>",
            "The answer is <answer>B</answer>.",
            "I think <answer>C</answer>",
        ]
        answers = ["A", "B", "A"]
        scores = evaluate_science_correctness(responses, answers)
        assert scores == [1, 1, 0]


class TestToolUseEval:
    def test_extract_actions(self):
        text = "Action: Search\nSome text\nAction: Click"
        assert extract_actions(text) == ["Search", "Click"]

    def test_extract_action_inputs(self):
        text = 'Action Input: {"query": "hello"}\nAction Input: {"page": "1"}'
        result = extract_action_inputs(text)
        assert result == {"query": "hello", "page": "1"}

    def test_extract_action_inputs_invalid_json(self):
        text = 'Action Input: {invalid json}\nAction Input: {"key": "val"}'
        result = extract_action_inputs(text)
        assert result == {"key": "val"}

    def test_evaluate_tooluse_correctness(self):
        responses = [
            'Action: Search\nAction Input: {"query": "weather"}',
            'Action: Click\nAction Input: {"button": "submit"}',
        ]
        golden_answers = [
            [{"Action": "Search", "Action_Input": '{"query": "weather"}'}],
            [{"Action": "Click", "Action_Input": '{"button": "wrong"}'}],
        ]
        scores = evaluate_tooluse_correctness(responses, golden_answers)
        assert scores == [1, 0]
