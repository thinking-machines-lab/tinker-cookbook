"""Tests for the IFEval instruction verifier.

Covers the silent free-credit paths reported in the repeat_prompt and
english_capital issues: an instruction id with no branch fell through to
``return True`` at DEBUG, and the all-capital check accepted title case.
"""

import pytest

from tinker_cookbook.eval.benchmarks._ifeval_verify import (
    verify_all_instructions,
    verify_instruction,
)

PROMPT = (
    "Write a blog post about the most interesting things you have seen or "
    "ridden on public transportation."
)


class TestRepeatPrompt:
    def test_violating_response_fails(self):
        assert not verify_instruction(
            "combination:repeat_prompt",
            "i will not repeat anything",
            {"prompt_to_repeat": PROMPT},
        )

    def test_complying_response_passes(self):
        response = PROMPT + " Here is the post: trams are underrated."
        assert verify_instruction(
            "combination:repeat_prompt", response, {"prompt_to_repeat": PROMPT}
        )

    def test_case_and_whitespace_insensitive_like_reference(self):
        response = "  " + PROMPT.upper() + " and now the answer."
        assert verify_instruction(
            "combination:repeat_prompt", response, {"prompt_to_repeat": PROMPT}
        )

    def test_missing_kwargs_fails_closed(self):
        assert not verify_instruction("combination:repeat_prompt", "anything", {})

    def test_solo_constraint_prompt_no_longer_scores_any_response_correct(self):
        # google/IFEval key=1480: repeat_prompt is the only constraint.
        fraction, results = verify_all_instructions(
            "garbage response",
            ["combination:repeat_prompt"],
            [{"prompt_to_repeat": PROMPT}],
        )
        assert fraction < 1.0
        assert results["combination:repeat_prompt"] is False


class TestEnglishCapital:
    def test_all_caps_passes(self):
        assert verify_instruction("change_case:english_capital", "THIS IS ALL CAPS.", {})

    def test_title_case_fails(self):
        assert not verify_instruction(
            "change_case:english_capital", "This Response Is Title Case", {}
        )

    def test_mixed_case_fails(self):
        assert not verify_instruction("change_case:english_capital", "THIS IS MOSTLY caps", {})


class TestUnhandledInstructionId:
    def test_unknown_id_fails_closed(self):
        assert not verify_instruction("no_such:instruction", "any response", {})

    @pytest.mark.parametrize(
        "iid",
        ["detectable_format:constrained_response", "count:counting_composition"],
    )
    def test_documented_approximations_still_pass(self, iid):
        assert verify_instruction(iid, "any response", {})
