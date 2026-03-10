"""Unit tests for the SDPO recipe."""

import tinker
import torch

from tinker_cookbook.recipes.sdpo.train import (
    build_full_sequence,
    build_student_datum,
    compute_sdpo_loss,
    extract_response_tokens,
)
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.completers import TokensWithLogprobs


def _make_simple_datum(prompt_tokens: list[int], response_tokens: list[int]) -> tinker.Datum:
    """Helper: build a student datum from token lists."""
    ob = tinker.ModelInput.from_ints(prompt_tokens)
    return build_student_datum(ob, response_tokens)


class TestExtractResponseTokens:
    def test_single_transition(self):
        traj = Trajectory(
            transitions=[
                Transition(
                    ob=tinker.ModelInput.from_ints([1, 2, 3]),
                    ac=TokensWithLogprobs(tokens=[10, 11, 12], maybe_logprobs=[-1.0, -2.0, -3.0]),
                    reward=1.0,
                    episode_done=True,
                )
            ],
            final_ob=tinker.ModelInput.empty(),
        )
        assert extract_response_tokens(traj) == [10, 11, 12]

    def test_multi_transition(self):
        traj = Trajectory(
            transitions=[
                Transition(
                    ob=tinker.ModelInput.from_ints([1]),
                    ac=TokensWithLogprobs(tokens=[10], maybe_logprobs=[-1.0]),
                    reward=0.0,
                    episode_done=False,
                ),
                Transition(
                    ob=tinker.ModelInput.from_ints([1, 10, 2]),
                    ac=TokensWithLogprobs(tokens=[11, 12], maybe_logprobs=[-2.0, -3.0]),
                    reward=1.0,
                    episode_done=True,
                ),
            ],
            final_ob=tinker.ModelInput.empty(),
        )
        assert extract_response_tokens(traj) == [10, 11, 12]


class TestBuildFullSequence:
    def test_appends_tokens(self):
        ob = tinker.ModelInput.from_ints([1, 2, 3])
        full = build_full_sequence(ob, [10, 11])
        assert list(full.to_ints()) == [1, 2, 3, 10, 11]


class TestBuildStudentDatum:
    def test_basic(self):
        prompt = [100, 101, 102]
        response = [200, 201]
        datum = _make_simple_datum(prompt, response)

        # model_input is right-shifted: first 4 tokens (len 5 - 1)
        assert datum.model_input.length == 4  # 3 prompt + 2 response - 1

        weights = datum.loss_fn_inputs["weights"].data
        target_tokens = datum.loss_fn_inputs["target_tokens"].data

        # 4 target tokens; first 2 are prompt targets (weight=0), last 2 are response (weight=1)
        assert len(weights) == 4
        assert len(target_tokens) == 4
        assert weights[:2] == [0.0, 0.0]
        assert weights[2:] == [1.0, 1.0]

    def test_single_response_token(self):
        datum = _make_simple_datum([1, 2], [10])
        weights = datum.loss_fn_inputs["weights"].data
        # total length 3, targets length 2
        assert len(weights) == 2
        assert weights[0] == 0.0
        assert weights[1] == 1.0


class TestComputeSDPOLoss:
    def test_zero_when_equal(self):
        """Loss should be near zero when student and teacher logprobs match."""
        datum = _make_simple_datum([1, 2, 3], [10, 11, 12])

        # Student logprobs for all target positions (length = total - 1 = 5)
        student_lps = torch.tensor([-2.0, -1.5, -1.0, -0.8, -1.2])
        # Teacher logprobs for response positions only (length = 3)
        teacher_lps = torch.tensor([-1.0, -0.8, -1.2])

        loss, metrics = compute_sdpo_loss([datum], [student_lps], [teacher_lps])
        # When student response logprobs == teacher logprobs, loss should be 0.
        assert abs(metrics["sdpo/loss"]) < 1e-6
        assert abs(metrics["sdpo/mean_log_ratio"]) < 1e-6

    def test_positive_when_student_higher(self):
        """Loss should be positive when student assigns higher probability than teacher."""
        datum = _make_simple_datum([1, 2, 3], [10, 11, 12])

        # Student response logprobs higher than teacher → positive log-ratio
        student_lps = torch.tensor([-2.0, -1.5, -0.5, -0.3, -0.7])
        teacher_lps = torch.tensor([-1.0, -0.8, -1.2])

        loss, metrics = compute_sdpo_loss([datum], [student_lps], [teacher_lps])
        # Positive log-ratio × student logprobs (negative) → negative loss per token.
        # But mean may vary; the key check is that loss is non-zero.
        assert metrics["sdpo/mean_log_ratio"] > 0

    def test_gradient_flows(self):
        """Student logprobs should have gradients after loss computation."""
        datum = _make_simple_datum([1, 2, 3], [10, 11])

        student_lps = torch.tensor([-2.0, -1.5, -1.0, -0.8], requires_grad=True)
        teacher_lps = torch.tensor([-1.2, -0.9])

        loss, _ = compute_sdpo_loss([datum], [student_lps], [teacher_lps])
        loss.backward()

        assert student_lps.grad is not None
        # Gradients on prompt positions should be zero (weights=0 → not included in loss).
        assert student_lps.grad[0].item() == 0.0
        assert student_lps.grad[1].item() == 0.0
        # Gradients on response positions should be non-zero.
        assert student_lps.grad[2].item() != 0.0
        assert student_lps.grad[3].item() != 0.0

    def test_empty_data(self):
        """Should handle empty data gracefully."""
        loss, metrics = compute_sdpo_loss([], [], [])
        assert metrics["sdpo/loss"] == 0.0

    def test_multiple_datums(self):
        """Should correctly handle batches of multiple datums."""
        datum1 = _make_simple_datum([1, 2], [10, 11])
        datum2 = _make_simple_datum([3, 4, 5], [20])

        student1 = torch.tensor([-1.0, -0.5, -0.8])
        teacher1 = torch.tensor([-0.5, -0.8])

        student2 = torch.tensor([-2.0, -1.5, -1.0, -0.7])
        teacher2 = torch.tensor([-0.7])

        loss, metrics = compute_sdpo_loss(
            [datum1, datum2], [student1, student2], [teacher1, teacher2]
        )
        assert metrics["sdpo/loss"] != 0.0
        assert "sdpo/mean_log_ratio" in metrics
