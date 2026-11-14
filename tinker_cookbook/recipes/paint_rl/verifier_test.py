import random

import pytest

from tinker_cookbook.recipes.paint_rl.verifier import (
    effort_preset,
    extract_letter_score,
    round_robin_pairs,
)


def test_round_robin_covers_every_pair_once():
    pairs = round_robin_pairs(5, random.Random(0))
    assert len(pairs) == 10
    assert {frozenset(p) for p in pairs} == {
        frozenset((a, b)) for a in range(5) for b in range(a + 1, 5)
    }


def test_effort_preset():
    assert effort_preset(0.2) == "low"
    assert effort_preset(0.99) == "xhigh"
    with pytest.raises(ValueError):
        effort_preset(0.3)


def test_letter_grade_maps_a_to_j_onto_unit_interval():
    # A-J is a 1-10 scale: A (10) is 1.0, J (1) is 0.0, evenly spaced.
    assert extract_letter_score("<score> A </score>") == 1.0
    assert extract_letter_score("<score>J</score>") == 0.0
    e = extract_letter_score("<score> E </score>")
    assert e is not None and abs(e - 5 / 9) < 1e-12
    assert extract_letter_score("<score> K </score>") is None  # off the scale
    assert extract_letter_score("no verdict") is None


def test_transient_errors_are_retried():
    import httpx
    import openai

    from tinker_cookbook.recipes.paint_rl.verifier import _is_transient

    request = httpx.Request("POST", "https://example.invalid")

    def status(code: int) -> openai.APIStatusError:
        return openai.APIStatusError("x", response=httpx.Response(code, request=request), body=None)

    assert _is_transient(status(502)) and _is_transient(status(429))
    assert not _is_transient(status(400))
    assert _is_transient(openai.APIConnectionError(request=request))
    assert not _is_transient(RuntimeError("no answer logprobs"))


def test_top_is_tied():
    from tinker_cookbook.recipes.paint_rl.verifier import top_is_tied

    assert top_is_tied([8 / 9, 8 / 9, 0.5])
    assert not top_is_tied([8 / 9, 7 / 9, 7 / 9])  # a tie below the top is not counted
    assert not top_is_tied([0.7])


def test_mean_per_painting_skips_unusable_verdicts():
    from tinker_cookbook.recipes.paint_rl.verifier import Verdict, _mean_per_painting

    def v(i: int, score: float) -> Verdict:
        return Verdict("c", 0, i, None, score, None, "", False)

    verdicts = [v(0, 1.0), v(0, 0.0), v(1, 0.2), v(2, 0.9)]
    assert _mean_per_painting(verdicts, 3, [True, False, True, False]) == [1.0, 0.2, 0.5]


def test_reward_config_follows_reward_mode():
    from tinker_cookbook.recipes.paint_rl.env import PaintRLDatasetBuilder, RewardWeights
    from tinker_cookbook.recipes.paint_rl.verifier import JudgeConfig, RewardMode, VerifierConfig

    def builder(mode: RewardMode) -> PaintRLDatasetBuilder:
        return PaintRLDatasetBuilder(
            model_name_for_tokenizer="m", renderer_name="r", batch_size=1, group_size=2,
            n_batches=1, policy_effort=0.7, reward_mode=mode, verifier_config=VerifierConfig(),
            judge_config=JudgeConfig(), evaluator_config=None, weights=RewardWeights(),
            canvas_size=512, render_concurrency=1,
            render_timeout_s=1.0, n_train_prompts=1, n_test_prompts=0, test_group_size=1, seed=0,
        )  # fmt: skip

    assert isinstance(builder("verifier").reward_config, VerifierConfig)
    assert isinstance(builder("judge").reward_config, JudgeConfig)
    # The two arms are matched on everything but the scoring.
    v, j = VerifierConfig(), JudgeConfig()
    assert (v.model_name, v.effort, v.max_tokens, v.max_concurrency) == (
        j.model_name, j.effort, j.max_tokens, j.max_concurrency,
    )  # fmt: skip
    assert effort_preset(v.effort) == "low" and v.n_evaluations == 4


def test_absolute_prompt_asks_for_one_criterion_a_to_j():
    from tinker_cookbook.recipes.paint_rl.verifier import (
        EvaluatorConfig,
        JudgeConfig,
        VerifierConfig,
        absolute_prompt,
        resolve_criteria,
    )

    for criterion in resolve_criteria(()):
        prompt = absolute_prompt("Paint a teal mailbox.", criterion)
        assert criterion[0] in prompt and "<score> LETTER_A_TO_J </score>" in prompt
        others = [name for name, _ in resolve_criteria(()) if name != criterion[0]]
        assert not any(name in prompt for name in others)
    # The judge and the held-out evaluator both score the verifier's three criteria.
    assert resolve_criteria(EvaluatorConfig().criteria) == resolve_criteria(())
    assert resolve_criteria(JudgeConfig().criteria) == resolve_criteria(())
    assert len(resolve_criteria(VerifierConfig().criteria)) == 3
    with pytest.raises(ValueError):
        resolve_criteria(("Overall Quality",))


def test_letter_scale_can_be_narrowed_per_process():
    # use_letter_scale changes llm_verifier for the whole process, so it is
    # checked in a fresh interpreter to leave the other tests on A-T.
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        from llm_verifier import fine_grained_reward as fgr
        from tinker_cookbook.recipes.paint_rl.verifier import pairwise_note, use_letter_scale
        use_letter_scale(5)
        use_letter_scale(5)  # idempotent
        prompt = fgr.build_prompt("p", "a", "b", {"name": "n", "description": "d"}, pairwise_note(5))
        assert "letter A-E" in prompt and "LETTER_A_TO_E" in prompt and "A-T" not in prompt, prompt
        assert "E means a complete failure" in prompt
        assert fgr.extract_score("<score_A> A </score_A>", None, None, "<score_A>") == 1.0
        assert fgr.extract_score("<score_A> B </score_A>", None, None, "<score_A>") == 0.75
        assert fgr.extract_score("<score_A> E </score_A>", None, None, "<score_A>") == 0.0
        lp = [[("A", -0.6931), ("C", -0.6931)]]  # 50/50 between A (5) and C (3)
        assert abs(fgr.extract_score("", ["<score_A>"], [[("x", 0.0)]] + lp, "<score_A>") - 0.75) < 1e-3
        try:
            use_letter_scale(20)
        except ValueError:
            pass
        else:
            raise AssertionError("a second scale in one process must be refused")
        from tinker_cookbook.recipes.paint_rl.verifier import use_letter_logprobs
        use_letter_logprobs(False)
        # The written letter wins over a logprob distribution that disagrees.
        text = "<score_A> B </score_A>"
        assert fgr.extract_score(text, ["<score_A>"], [[("x", 0.0)]] + lp, "<score_A>") == 0.75
        assert fgr.extract_score("no verdict", None, None, "<score_A>") == 0.5
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]


def test_judge_can_grade_with_letters():
    from tinker_cookbook.recipes.paint_rl.verifier import (
        absolute_prompt,
        extract_letter_score,
        resolve_criteria,
    )

    prompt = absolute_prompt("Paint a teal mailbox.", resolve_criteria(())[0], letter_grades=20)
    assert "<score> LETTER_A_TO_T </score>" in prompt and "A-J" not in prompt
    assert extract_letter_score("... <score> A </score>", 20) == 1.0
    assert extract_letter_score("<score>T</score>", 20) == 0.0
    c = extract_letter_score("<score> c </score>", 20)
    assert c is not None and abs(c - (1 - 2 / 19)) < 1e-12
    assert (
        extract_letter_score("draft <score> T </score> final <score> B </score>", 20) == 1 - 1 / 19
    )
    assert extract_letter_score("<score> Z </score>", 20) is None
    assert extract_letter_score("<score> 7 </score>", 20) is None
    assert extract_letter_score("<score> J\n\nThat is my grade.", 20) == 1 - 9 / 19
    assert extract_letter_score("<score> **B** </score>", 20) == 1 - 1 / 19
    assert extract_letter_score("<score> LETTER_A_TO_T </score>", 20) is None


def test_judge_and_heldout_evaluator_grade_a_to_j_by_default():
    from tinker_cookbook.recipes.paint_rl.verifier import (
        EvaluatorConfig,
        JudgeConfig,
        absolute_prompt,
        resolve_criteria,
    )

    for cfg in (EvaluatorConfig(), JudgeConfig()):
        assert cfg.letter_grades == 10 and len(resolve_criteria(cfg.criteria)) == 3
        prompt = absolute_prompt("p", resolve_criteria(cfg.criteria)[0], cfg.letter_grades)
        assert "<score> LETTER_A_TO_J </score>" in prompt and "A-J" in prompt
