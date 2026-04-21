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
    build_reverse_kl_datums,
    build_sdft_teacher_prompt,
    build_topk_distillation_datums,
    reverse_kl_custom_loss,
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

        # First 3 completion positions are skipped (skip_first_n_tokens=3 default)
        # so all positions should have zero weights in this 3-token completion
        # Let's test with skip_first_n_tokens=0 to verify the logic
        # Re-run without skipping
        new_datums2, _ = await build_topk_distillation_datums(
            data_D=[datum],
            metadata_D=[{"group_idx": 0, "traj_idx": 0}],
            teacher_client=mock_client,
            teacher_prompts_P=[teacher_prompt],
            topk=K,
            max_context_length=32768,
            skip_first_n_tokens=0,
        )
        target_tokens2 = new_datums2[0].loss_fn_inputs["target_tokens"].to_torch()
        weights2 = new_datums2[0].loss_fn_inputs["weights"].to_torch()

        # Prompt positions (0, 1) should have zero weights
        assert weights2[0].sum() == 0.0
        assert weights2[1].sum() == 0.0

        # Completion positions (2, 3, 4) should have non-zero weights
        # Weights are normalized by n_comp (3) * n_datums (1)
        for pos in [2, 3, 4]:
            assert weights2[pos].sum() > 0.0

        # Check token IDs at completion positions
        assert target_tokens2[2, 0].item() == 40
        assert target_tokens2[3, 0].item() == 50
        assert target_tokens2[4, 0].item() == 60

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


def _pack_weights(
    teacher_log_renorm_NK: torch.Tensor,
    position_mask_N: torch.Tensor,
    k_valid_N: torch.Tensor,
) -> torch.Tensor:
    """Test helper: pack teacher renorm probs into the `0` sentinel encoding."""
    N, K = teacher_log_renorm_NK.shape
    weights = torch.zeros(N, K, dtype=torch.float32)
    probs = teacher_log_renorm_NK.exp()
    for n in range(N):
        if position_mask_N[n].item() > 0:
            k = int(k_valid_N[n].item())
            weights[n, :k] = probs[n, :k]
    return weights


class TestBuildReverseKLDatums:
    @pytest.mark.asyncio
    async def test_shapes_and_renorm(self):
        """Reverse-KL datums carry (N, K) target_tokens + weights (log q renorm
        at valid slots, -inf elsewhere)."""
        K = 3
        datum = _make_datum([10, 20, 30], [40, 50, 60, 70, 80])
        teacher_prompt = tinker.ModelInput.from_ints([100, 200])

        # Teacher-forced length = 2 prompt + 5 completion = 7 positions
        mock_topk = [None] * 2 + [
            [(40, -0.5), (41, -1.5), (42, -2.5)],
            [(50, -0.3), (51, -1.3), (52, -2.3)],
            [(60, -0.8), (61, -1.8), (62, -2.8)],
            [(70, -0.1), (71, -1.1), (72, -2.1)],
            [(80, -0.4), (81, -1.4), (82, -2.4)],
        ]
        mock_response = MagicMock()
        mock_response.topk_prompt_logprobs = mock_topk
        mock_client = AsyncMock()
        mock_client.sample_async = AsyncMock(return_value=mock_response)

        rev_datums, metrics = await build_reverse_kl_datums(
            data_D=[datum],
            metadata_D=[{"group_idx": 0, "traj_idx": 0}],
            teacher_client=mock_client,
            teacher_prompts_P=[teacher_prompt],
            topk=K,
            skip_first_n_tokens=0,
        )

        rd = rev_datums[0]
        target_tokens = rd.loss_fn_inputs["target_tokens"].to_torch()
        weights = rd.loss_fn_inputs["weights"].to_torch()

        N = datum.model_input.length
        assert target_tokens.shape == (N, K)
        assert weights.shape == (N, K)

        prompt_len = 2  # [10, 20] → input_tokens length before completion mask starts
        for pos in range(prompt_len):
            # Prompt positions are fully masked (all zeros)
            assert (weights[pos] == 0).all()
        for pos in range(prompt_len, N):
            assert (weights[pos] > 0).all()  # all K slots valid for this mock
            # Teacher probs sum to 1 (renormalized over top-K)
            assert math.isclose(weights[pos].sum().item(), 1.0, rel_tol=1e-5)

        assert metrics["sdft/num_datums"] == 1.0
        assert metrics["sdft/topk"] == float(K)
        assert metrics["sdft/mean_teacher_entropy"] > 0.0

    @pytest.mark.asyncio
    async def test_skip_first_n_tokens(self):
        """skip_first_n_tokens=3 fully masks the first three completion positions."""
        K = 2
        datum = _make_datum([10, 20], [30, 40, 50, 60, 70])
        teacher_prompt = tinker.ModelInput.from_ints([100])

        mock_topk = [None] + [
            [(30, -0.5), (31, -1.5)],
            [(40, -0.5), (41, -1.5)],
            [(50, -0.5), (51, -1.5)],
            [(60, -0.5), (61, -1.5)],
            [(70, -0.5), (71, -1.5)],
        ]
        mock_response = MagicMock()
        mock_response.topk_prompt_logprobs = mock_topk
        mock_client = AsyncMock()
        mock_client.sample_async = AsyncMock(return_value=mock_response)

        rev_datums, _ = await build_reverse_kl_datums(
            data_D=[datum],
            metadata_D=[{"group_idx": 0, "traj_idx": 0}],
            teacher_client=mock_client,
            teacher_prompts_P=[teacher_prompt],
            topk=K,
            skip_first_n_tokens=3,
        )
        weights = rev_datums[0].loss_fn_inputs["weights"].to_torch()
        # Of model_input length 6, only 2 positions should have any positive weight.
        active_positions = (weights > 0).any(dim=-1).sum().item()
        assert active_positions == 2


class TestReverseKLCustomLoss:
    def _make_rev_datum(
        self,
        N: int,
        K: int,
        teacher_log_renorm_NK: torch.Tensor,
        position_mask_N: torch.Tensor,
        k_valid_N: torch.Tensor,
    ) -> tinker.Datum:
        weights = _pack_weights(teacher_log_renorm_NK, position_mask_N, k_valid_N)
        return tinker.Datum(
            model_input=tinker.ModelInput.from_ints([0] * N),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(torch.zeros(N, K, dtype=torch.long)),
                "weights": tinker.TensorData.from_torch(weights),
            },
        )

    def test_zero_when_distributions_equal(self):
        """L_rev ≈ 0 when student (after renorm) matches teacher."""
        N, K = 4, 3
        raw = torch.tensor([0.2, -0.1, 0.5])
        teacher_log_renorm = (raw - torch.logsumexp(raw, dim=0)).unsqueeze(0).repeat(N, 1)
        position_mask = torch.tensor([0.0, 1.0, 1.0, 1.0])
        k_valid = torch.tensor([K, K, K, K], dtype=torch.long)

        student_logp = raw.unsqueeze(0).repeat(N, 1).clone().requires_grad_(True)

        datum = self._make_rev_datum(N, K, teacher_log_renorm, position_mask, k_valid)
        loss, metrics = reverse_kl_custom_loss([datum], [student_logp])

        assert abs(loss.item()) < 1e-6
        assert metrics["sdft/reverse_kl_mean"] < 1e-6
        assert metrics["sdft/reverse_kl_positions"] == 3.0

    def test_positive_kl_and_gradient(self):
        """L_rev > 0 when p ≠ q, and gradient flows through student logits."""
        N, K = 2, 3
        teacher_raw = torch.tensor([0.0, -2.0, -4.0])
        teacher_log_renorm = (teacher_raw - torch.logsumexp(teacher_raw, dim=0))
        teacher_log_renorm_NK = teacher_log_renorm.unsqueeze(0).repeat(N, 1)

        student_logp = torch.tensor([[-4.0, -2.0, 0.0], [-4.0, -2.0, 0.0]], requires_grad=True)

        position_mask = torch.tensor([1.0, 1.0])
        k_valid = torch.tensor([K, K], dtype=torch.long)
        datum = self._make_rev_datum(N, K, teacher_log_renorm_NK, position_mask, k_valid)

        loss, metrics = reverse_kl_custom_loss([datum], [student_logp])
        assert loss.item() > 0.1

        p_renorm = torch.log_softmax(student_logp, dim=-1).exp()
        kl_full = (
            p_renorm
            * (torch.log_softmax(student_logp, dim=-1) - teacher_log_renorm_NK)
        ).sum()
        assert abs(metrics["sdft/reverse_kl_mean"] * 2 - kl_full.item()) < 1e-4

        loss.backward()
        assert student_logp.grad is not None
        assert student_logp.grad.abs().sum().item() > 0.0

    def test_position_mask_zeros_contribution(self):
        """Position-mask=0 fully excludes a position from the loss and metrics."""
        N, K = 2, 2
        teacher_log_renorm_NK = torch.tensor(
            [[-0.1, -2.5], [-0.1, -2.5]], dtype=torch.float32
        )
        student_logp = torch.tensor([[2.0, -2.0], [2.0, -2.0]], requires_grad=True)
        position_mask = torch.tensor([0.0, 1.0])
        k_valid = torch.tensor([K, K], dtype=torch.long)
        datum = self._make_rev_datum(N, K, teacher_log_renorm_NK, position_mask, k_valid)

        loss_masked, metrics_masked = reverse_kl_custom_loss([datum], [student_logp])
        datum_all = self._make_rev_datum(
            N, K, teacher_log_renorm_NK, torch.tensor([1.0, 1.0]), k_valid
        )
        student_logp2 = torch.tensor([[2.0, -2.0], [2.0, -2.0]], requires_grad=True)
        loss_all, _ = reverse_kl_custom_loss([datum_all], [student_logp2])
        assert abs(loss_masked.item() * 2 - loss_all.item()) < 1e-5
        assert metrics_masked["sdft/reverse_kl_positions"] == 1.0
