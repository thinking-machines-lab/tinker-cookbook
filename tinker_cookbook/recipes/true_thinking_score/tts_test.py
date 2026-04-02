"""Unit tests for TTS computation utilities (no API needed)."""

import random

from tinker_cookbook.recipes.true_thinking_score.tts import (
    is_self_verification_step,
    perturb_numbers,
    segment_cot_steps,
)


class TestSegmentCotSteps:
    def test_simple_steps(self):
        cot = (
            "First, I need to find the sum of 1 to 10.\n"
            "We know the formula is n*(n+1)/2.\n"
            "So the answer is 10*11/2 = 55.\n"
            "Therefore the sum is 55."
        )
        steps = segment_cot_steps(cot)
        assert len(steps) >= 2
        assert any("formula" in s for s in steps)

    def test_self_verification_step(self):
        cot = (
            "Let me calculate 5 * 12 = 60.\n"
            "Wait, let me re-check that. 5 * 12 = 60. Yes, that's correct.\n"
            "Therefore the area is 60."
        )
        steps = segment_cot_steps(cot)
        assert len(steps) >= 2
        # At least one step should be identified as self-verification
        sv_steps = [s for s in steps if is_self_verification_step(s)]
        assert len(sv_steps) >= 1

    def test_empty_input(self):
        assert segment_cot_steps("") == []
        assert segment_cot_steps("   ") == []

    def test_single_sentence(self):
        steps = segment_cot_steps("The answer is 42.")
        assert len(steps) == 1

    def test_merge_short_fragments(self):
        cot = "Step 1: x = 5.\nOK.\nStep 2: y = x + 3 = 8."
        steps = segment_cot_steps(cot, min_step_chars=20)
        # "OK." should be merged with the previous step
        assert not any(s.strip() == "OK." for s in steps)


class TestPerturbNumbers:
    def test_basic_perturbation(self):
        rng = random.Random(42)
        text = "The sum is 55 and the product is 120."
        perturbed = perturb_numbers(text, rng)
        assert perturbed != text
        # Should still have numbers, just different
        assert "sum is" in perturbed
        assert "product is" in perturbed

    def test_preserves_non_numeric_text(self):
        rng = random.Random(42)
        text = "Step 1: calculate the area"
        perturbed = perturb_numbers(text, rng)
        # "1" after "Step " should not be perturbed (preceded by letter/space boundary)
        # Actually our regex matches standalone numbers, so "1" after "Step " would match
        # Let's just verify the text structure is preserved
        assert "Step" in perturbed
        assert "calculate the area" in perturbed

    def test_preserves_decimal_format(self):
        rng = random.Random(42)
        text = "The value is 3.14"
        perturbed = perturb_numbers(text, rng)
        # Should contain a decimal number
        import re

        assert re.search(r"\d+\.\d+", perturbed)

    def test_zero_handling(self):
        rng = random.Random(42)
        text = "The value is 0"
        perturbed = perturb_numbers(text, rng)
        # Should not remain 0
        assert "is 0" not in perturbed

    def test_reproducibility(self):
        text = "x = 10, y = 20, z = 30"
        p1 = perturb_numbers(text, random.Random(42))
        p2 = perturb_numbers(text, random.Random(42))
        assert p1 == p2

    def test_does_not_perturb_variable_names(self):
        rng = random.Random(42)
        text = "variable x1 equals y2"
        perturbed = perturb_numbers(text, rng)
        # x1 and y2 should be left alone (preceded by letters)
        assert "x1" in perturbed
        assert "y2" in perturbed


class TestIsSelfVerification:
    def test_wait_pattern(self):
        assert is_self_verification_step("Wait, that doesn't look right")
        assert is_self_verification_step("Wait, let me re-check this")

    def test_hmm_pattern(self):
        assert is_self_verification_step("Hmm, I think I made an error here")

    def test_actually_pattern(self):
        assert is_self_verification_step("Actually, the correct formula is...")

    def test_let_me_verify(self):
        assert is_self_verification_step("Let me verify: 5 * 12 = 60")
        assert is_self_verification_step("Let me double-check the calculation")

    def test_non_verification(self):
        assert not is_self_verification_step("The sum of 1 to 10 is 55")
        assert not is_self_verification_step("Using the formula n*(n+1)/2")
