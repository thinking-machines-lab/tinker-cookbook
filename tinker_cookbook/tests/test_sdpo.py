"""Unit tests for the SDPO implementation."""

import tinker
import torch

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.recipes.sdpo.sciknoweval_env import (
    _extract_answer,
    _format_choices,
)
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.sdpo.data import (
    build_full_sequence,
    build_sdpo_datum,
    extract_response_logprobs,
    extract_response_tokens,
)
from tinker_cookbook.sdpo.teacher import strip_thinking_blocks


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
        assert extract_response_logprobs(traj) == [-1.0, -2.0, -3.0]

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
        assert extract_response_logprobs(traj) == [-1.0, -2.0, -3.0]


class TestBuildFullSequence:
    def test_appends_tokens(self):
        ob = tinker.ModelInput.from_ints([1, 2, 3])
        full = build_full_sequence(ob, [10, 11])
        assert list(full.to_ints()) == [1, 2, 3, 10, 11]


class TestBuildSDPODatum:
    def test_basic_structure(self):
        """Datum should have target_tokens, logprobs, and advantages."""
        ob = tinker.ModelInput.from_ints([100, 101, 102])
        response = [200, 201]
        sampled_lps = [-1.0, -2.0]
        teacher_lps = torch.tensor([-0.5, -1.5])

        datum = build_sdpo_datum(ob, response, sampled_lps, teacher_lps)

        # model_input is right-shifted: length = 3 + 2 - 1 = 4
        assert datum.model_input.length == 4

        target_tokens = datum.loss_fn_inputs["target_tokens"].data
        logprobs = datum.loss_fn_inputs["logprobs"].data
        advantages = datum.loss_fn_inputs["advantages"].data

        assert len(target_tokens) == 4
        assert len(logprobs) == 4
        assert len(advantages) == 4

    def test_logprobs_alignment(self):
        """Sampled logprobs should appear at response positions, 0 at prompt."""
        ob = tinker.ModelInput.from_ints([1, 2, 3])
        response = [10, 11]
        sampled_lps = [-1.0, -2.0]
        teacher_lps = torch.tensor([-0.5, -1.5])

        datum = build_sdpo_datum(ob, response, sampled_lps, teacher_lps)
        logprobs = datum.loss_fn_inputs["logprobs"].data

        # Prompt positions (indices 0, 1) should be 0
        assert logprobs[0] == 0.0
        assert logprobs[1] == 0.0
        # Response positions should have sampled logprobs
        assert logprobs[2] == -1.0
        assert logprobs[3] == -2.0

    def test_advantages_are_teacher_minus_student(self):
        """Advantages should be teacher_lp - student_lp at response positions."""
        ob = tinker.ModelInput.from_ints([1, 2])
        response = [10, 11]
        sampled_lps = [-1.0, -2.0]
        teacher_lps = torch.tensor([-0.5, -1.0])

        datum = build_sdpo_datum(ob, response, sampled_lps, teacher_lps)
        advantages = datum.loss_fn_inputs["advantages"].data

        # Prompt position
        assert advantages[0] == 0.0
        # Response positions: teacher - student
        assert abs(advantages[1] - (-0.5 - (-1.0))) < 1e-6  # 0.5
        assert abs(advantages[2] - (-1.0 - (-2.0))) < 1e-6  # 1.0

    def test_zero_advantages_when_equal(self):
        """Advantages should be 0 when teacher == student."""
        ob = tinker.ModelInput.from_ints([1, 2])
        response = [10]
        sampled_lps = [-1.0]
        teacher_lps = torch.tensor([-1.0])

        datum = build_sdpo_datum(ob, response, sampled_lps, teacher_lps)
        advantages = datum.loss_fn_inputs["advantages"].data

        assert advantages[0] == 0.0  # prompt
        assert abs(advantages[1]) < 1e-6  # response: 0


class TestStripThinkingBlocks:
    def test_removes_thinking(self):
        assert strip_thinking_blocks("Hello <think>reasoning</think> world") == "Hello  world"

    def test_multiline_thinking(self):
        assert strip_thinking_blocks("Start <think>\nline1\n</think> end") == "Start  end"

    def test_no_thinking(self):
        assert strip_thinking_blocks("No thinking here") == "No thinking here"

    def test_multiple_blocks(self):
        assert strip_thinking_blocks("<think>a</think> mid <think>b</think> end") == "mid  end"


class TestSciKnowEvalGrading:
    def test_extract_answer_basic(self):
        assert _extract_answer("<answer>A</answer>") == "A"
        assert _extract_answer("<answer>B</answer>") == "B"
        assert _extract_answer("<answer> C </answer>") == "C"
        assert _extract_answer("<answer>D</answer>") == "D"

    def test_extract_answer_in_context(self):
        text = "<reasoning>blah blah</reasoning>\n<answer>B</answer>"
        assert _extract_answer(text) == "B"

    def test_extract_answer_lowercase(self):
        assert _extract_answer("<answer>a</answer>") == "A"

    def test_extract_answer_invalid(self):
        assert _extract_answer("no answer here") is None
        assert _extract_answer("<answer>E</answer>") is None
        assert _extract_answer("<answer>AB</answer>") is None

    def test_format_choices(self):
        choices = {"label": ["A", "B", "C"], "text": ["cat", "dog", "fish"]}
        result = _format_choices(choices)
        assert result == "A: cat\nB: dog\nC: fish"

    def test_format_choices_empty(self):
        assert _format_choices({}) == ""
        assert _format_choices({"label": [], "text": []}) == ""
