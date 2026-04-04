"""LLM-as-judge rewards.

Provides a rubric-based scoring system where a judge LLM evaluates model
responses against configurable criteria. Extracted from
``tinker_cookbook.recipes.rubric`` so any recipe can use LLM grading.

Includes telemetry via ``tinker_cookbook.utils.trace`` (Perfetto spans),
``tinker_cookbook.utils.logtree`` (HTML reports), and metric computation
helpers for logging reward statistics.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from tinker_cookbook.renderers.base import Message
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.trace import scope_span

if TYPE_CHECKING:
    from tinker_cookbook.completers import MessageCompleter

logger = logging.getLogger(__name__)

Conversation: TypeAlias = list[Message]


@dataclass
class Rubric:
    """A single grading criterion with extraction logic.

    Attributes:
        rubric_str: Human-readable description of what the rubric measures.
        extraction_regex: Regex to extract the numeric score from the
            judge response.  Must contain one capture group.
        grader_output_format_instruction: Instructions appended to the
            grader prompt telling it how to format its score.
    """

    rubric_str: str
    extraction_regex: str = r"<score>(.*?)</score>"
    grader_output_format_instruction: str = (
        "Please output your score between 0 and 1 wrapped in <score> ... </score>"
    )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def get_grader_prompt(self, convo: Conversation) -> Conversation:
        """Build a single-turn prompt for the judge to grade *convo*.

        The last message in *convo* is treated as the completion to grade;
        all preceding messages provide context.
        """
        context = convo[:-1]
        completion = convo[-1]

        def _role_label(role: str) -> str:
            return "Human" if role in ("user", "system") else "Chatbot"

        context_str = (
            "\n\n".join(f"{_role_label(m['role'])}: {m['content']}" for m in context)
            if context
            else "(No prior context)"
        )
        lines = [
            "I will show you a conversation context, a chatbot completion to grade, and a rubric.",
            "Please grade the chatbot's completion based on the rubric.",
            "",
            "<context>",
            context_str,
            "</context>",
            "",
            "<completion_to_grade>",
            f"Chatbot: {completion['content']}",
            "</completion_to_grade>",
            "",
            "<rubric>",
            self.rubric_str,
            "</rubric>",
            "",
            f"Please grade the chatbot's completion based on the rubric. {self.grader_output_format_instruction}",
        ]
        return [{"role": "user", "content": "\n".join(lines)}]

    # ------------------------------------------------------------------
    # Score extraction
    # ------------------------------------------------------------------

    def extract_score(self, response: str) -> float:
        """Parse a numeric score from the judge response.

        Returns ``0.0`` if the score cannot be extracted.
        """
        match = re.search(self.extraction_regex, response, re.DOTALL)
        if match is not None:
            try:
                return float(match.group(1))
            except ValueError:
                return 0.0
        return 0.0


async def score_with_rubric(
    convo: Conversation,
    rubric: Rubric,
    grader_llm: MessageCompleter,
    *,
    default_on_error: float = 0.0,
) -> tuple[float, str]:
    """Grade a conversation against a single rubric using the judge LLM.

    Args:
        convo: The full conversation including the assistant response
            to grade as the last message.
        rubric: The grading criterion.
        grader_llm: An async callable that takes a conversation and
            returns an assistant message.
        default_on_error: Score returned when the API call fails.
            Defaults to ``0.0``.

    Returns:
        Tuple of ``(score, grader_response_text)``.  On API failure the
        score is *default_on_error* and the response text describes the
        error.
    """
    grader_prompt = rubric.get_grader_prompt(convo)
    try:
        grader_response = await grader_llm(grader_prompt)
    except Exception as exc:
        logger.warning("LLM judge API call failed: %s", exc)
        return default_on_error, f"[error] {exc}"
    content = grader_response["content"]
    assert isinstance(content, str), "Grader response content must be a string"
    score = rubric.extract_score(content)
    return score, content


async def score_with_rubrics(
    convo: Conversation,
    rubrics: Sequence[Rubric],
    grader_llm: MessageCompleter,
) -> list[tuple[float, str]]:
    """Grade a conversation against multiple rubrics concurrently.

    Returns one ``(score, grader_response_text)`` per rubric, in order.
    """
    return list(
        await asyncio.gather(
            *[score_with_rubric(convo, rubric, grader_llm) for rubric in rubrics]
        )
    )


# ======================================================================
# Telemetry-instrumented LLM judge grading
# ======================================================================


async def score_with_rubric_traced(
    convo: Conversation,
    rubric: Rubric,
    grader_llm: MessageCompleter,
    *,
    reward_name: str = "llm_judge",
    log_to_logtree: bool = True,
) -> tuple[float, str, dict[str, float]]:
    """Grade a conversation with tracing, logtree logging, and metrics.

    Wraps :func:`grade_with_rubric` with telemetry:

    - An async ``scope_span`` named ``"compute_{reward_name}_reward"``
    - Logtree table with grading details
    - Metrics dict with computation time

    Args:
        convo: The full conversation including the assistant response.
        rubric: The grading criterion.
        grader_llm: An async callable that takes a conversation and returns
            an assistant message.
        reward_name: Name for the reward (used in span names and metric keys).
        log_to_logtree: Whether to emit logtree output.

    Returns:
        Tuple of ``(score, grader_response_text, metrics_dict)``.
    """
    t_start = time.perf_counter()

    async with scope_span(f"compute_{reward_name}_reward"):
        score, grader_response = await score_with_rubric(convo, rubric, grader_llm)

    elapsed = time.perf_counter() - t_start

    metrics = {
        f"reward/{reward_name}/computation_time": elapsed,
    }

    if log_to_logtree:
        rubric_preview = rubric.rubric_str[:100] + "..." if len(rubric.rubric_str) > 100 else rubric.rubric_str
        with logtree.scope_header("Reward Computation"):
            logtree.table_from_dict({
                "reward_type": "llm_judge",
                "rubric": rubric_preview,
                "score": score,
                "computation_time": f"{elapsed:.4f}s",
            })

    return score, grader_response, metrics


def compute_llm_judge_metrics(
    scores: Sequence[float],
    *,
    reward_name: str = "llm_judge",
) -> dict[str, float]:
    """Compute aggregate metrics for a batch of LLM judge scores.

    Thin wrapper around :func:`~tinker_cookbook.rewards._metrics.compute_reward_metrics`
    with ``reward_name`` defaulting to ``"llm_judge"``.
    """
    from tinker_cookbook.rewards._metrics import compute_reward_metrics

    return compute_reward_metrics(scores, reward_name)


# ======================================================================
# Deprecated aliases (backward compatibility)
# ======================================================================

grade_with_rubric = score_with_rubric
grade_with_rubric_traced = score_with_rubric_traced
grade_with_rubrics = score_with_rubrics
