"""Unit tests for SDFT recipe (no API key needed)."""

import math
from unittest.mock import AsyncMock, MagicMock

import pytest
import tinker
import torch

from tinker_cookbook.distillation.sdft import (
    DEFAULT_DEMO_TEMPLATE,
    _build_teacher_forced_sequence,
    _extract_completion_tokens,
    build_sdft_teacher_prompt,
    build_topk_distillation_datums,
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


def _make_datum(prompt_tokens: list[int], completion_tokens: list[int]) -> tinker.Datum:
    """Helper: build a Datum mimicking what trajectory_to_data produces."""
    full_tokens = prompt_tokens + completion_tokens
    # model_input = full_tokens[:-1] (left-shifted)
    input_tokens = full_tokens[:-1]
    # target_tokens = full_tokens[1:] (right-shifted)
    target_tokens = full_tokens[1:]
    N = len(input_tokens)
    prompt_len = len(prompt_tokens) - 1  # -1 for left shift
    mask = [0.0] * prompt_len + [1.0] * (N - prompt_len)
    logprobs = [0.0] * prompt_len + [-1.0] * (N - prompt_len)
    advantages = [0.0] * N
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs)),
            "advantages": tinker.TensorData.from_torch(torch.tensor(advantages)),
            "mask": tinker.TensorData.from_torch(torch.tensor(mask)),
        },
    )


class TestExtractCompletionTokens:
    def test_basic(self):
        datum = _make_datum([10, 20, 30], [40, 50, 60])
        teacher_prompt = tinker.ModelInput.from_ints([100, 200])
        tokens, tpl, cstart, trunc = _extract_completion_tokens(datum, teacher_prompt, 1024)
        assert tokens == [40, 50, 60]
        assert tpl == 2
        assert not trunc

    def test_empty_completion(self):
        # All tokens are "prompt" (mask all zeros)
        full = [10, 20, 30]
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(full[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(torch.tensor(full[1:])),
                "logprobs": tinker.TensorData.from_torch(torch.tensor([0.0] * (len(full) - 1))),
                "advantages": tinker.TensorData.from_torch(torch.tensor([0.0] * (len(full) - 1))),
                "mask": tinker.TensorData.from_torch(torch.tensor([0.0] * (len(full) - 1))),
            },
        )
        teacher_prompt = tinker.ModelInput.from_ints([100])
        tokens, _, _, _ = _extract_completion_tokens(datum, teacher_prompt, 1024)
        assert tokens == []

    def test_truncation(self):
        datum = _make_datum([10, 20], [30, 40, 50, 60, 70])
        teacher_prompt = tinker.ModelInput.from_ints(list(range(100)))  # 100 tokens
        # max_context = 103 → only 3 completion tokens fit
        tokens, _, _, trunc = _extract_completion_tokens(datum, teacher_prompt, 103)
        assert len(tokens) == 3
        assert trunc


class TestBuildTeacherForcedSequence:
    def test_basic(self):
        teacher_prompt = tinker.ModelInput.from_ints([100, 200])
        result = _build_teacher_forced_sequence(teacher_prompt, [300, 400])
        assert result.to_ints() == [100, 200, 300, 400]


class TestBuildTopkDistillationDatums:
    @pytest.mark.asyncio
    async def test_basic(self):
        """Test that top-K datums have correct shape and renormalized weights."""
        K = 3
        datum = _make_datum([10, 20, 30], [40, 50, 60])
        teacher_prompt = tinker.ModelInput.from_ints([100, 200])

        # Mock teacher sampling client
        # The teacher-forced sequence is [100, 200, 40, 50, 60] (teacher prompt + completion)
        # topk_prompt_logprobs should have entries at positions 2, 3, 4 (completion positions)
        mock_topk = [
            None,  # pos 0: no previous context
            [  # pos 1
                (201, -1.0),
                (202, -2.0),
                (203, -3.0),
            ],
            [  # pos 2 (first completion token)
                (40, -0.5),
                (41, -1.5),
                (42, -2.5),
            ],
            [  # pos 3
                (50, -0.3),
                (51, -1.3),
                (52, -2.3),
            ],
            [  # pos 4
                (60, -0.8),
                (61, -1.8),
                (62, -2.8),
            ],
        ]
        mock_response = MagicMock()
        mock_response.topk_prompt_logprobs = mock_topk

        mock_client = AsyncMock()
        mock_client.sample_async = AsyncMock(return_value=mock_response)

        new_datums, metrics = await build_topk_distillation_datums(
            data_D=[datum],
            metadata_D=[{"group_idx": 0, "traj_idx": 0}],
            teacher_client=mock_client,
            teacher_prompts_P=[teacher_prompt],
            topk=K,
            max_context_length=32768,
        )

        assert len(new_datums) == 1
        new_datum = new_datums[0]

        # Check shapes: (N, K) where N = model_input.length = 5
        target_tokens = new_datum.loss_fn_inputs["target_tokens"].to_torch()
        weights = new_datum.loss_fn_inputs["weights"].to_torch()
        assert target_tokens.shape == (5, K)
        assert weights.shape == (5, K)

        # Prompt positions (0, 1) should have zero weights
        assert weights[0].sum() == 0.0
        assert weights[1].sum() == 0.0

        # Completion positions (2, 3, 4) should have non-zero weights
        # Weights are normalized by n_completion_tokens (3) * n_datums (1)
        # so each position sums to ~1/3 instead of ~1.0
        n_comp = 3  # 3 completion positions
        n_datums = 1
        expected_sum = 1.0 / n_comp / n_datums
        for pos in [2, 3, 4]:
            assert weights[pos].sum() > 0.0
            assert abs(weights[pos].sum().item() - expected_sum) < 0.01

        # Check token IDs at completion positions
        assert target_tokens[2, 0].item() == 40
        assert target_tokens[3, 0].item() == 50
        assert target_tokens[4, 0].item() == 60

        assert metrics["sdft/num_datums"] == 1.0
        assert metrics["sdft/topk"] == float(K)

    @pytest.mark.asyncio
    async def test_no_completion_tokens(self):
        """Datum with no completion tokens produces zero-weight datum."""
        full = [10, 20, 30]
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(full[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(torch.tensor(full[1:])),
                "logprobs": tinker.TensorData.from_torch(torch.tensor([0.0, 0.0])),
                "advantages": tinker.TensorData.from_torch(torch.tensor([0.0, 0.0])),
                "mask": tinker.TensorData.from_torch(torch.tensor([0.0, 0.0])),
            },
        )
        teacher_prompt = tinker.ModelInput.from_ints([100])

        mock_response = MagicMock()
        mock_response.topk_prompt_logprobs = [None]
        mock_client = AsyncMock()
        mock_client.sample_async = AsyncMock(return_value=mock_response)

        new_datums, _ = await build_topk_distillation_datums(
            data_D=[datum],
            metadata_D=[{"group_idx": 0, "traj_idx": 0}],
            teacher_client=mock_client,
            teacher_prompts_P=[teacher_prompt],
            topk=5,
        )

        weights = new_datums[0].loss_fn_inputs["weights"].to_torch()
        assert weights.sum() == 0.0
