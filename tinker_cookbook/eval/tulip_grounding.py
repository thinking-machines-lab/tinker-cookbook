"""Optional GSAR groundedness scoring for tau2-bench trajectories.

**Off by default, fully optional, self-contained.** Nothing else in this
repository imports this module or depends on ``tulip-agents``. Import it
yourself and score a completed tau2-bench run's ``trajectories.jsonl``.

## The gap this fills

tau2-bench's own grading (``_check_actions`` in
``eval/benchmarks/_tau2_bench.py``) is action-matching: did the agent call
the right tools with the right arguments. It says nothing about whether the
agent's *spoken reply to the customer* actually reflects what those tools
returned. An agent that calls ``get_reservation_details`` correctly and then
tells the customer a fabricated cabin class scores identically to one that
reports it correctly — both called the right tool.

Verified live, not asserted: given the same real tool call and the same
real tool result, an honest summary and a hallucinated one (fabricated
cabin, fabricated baggage count) score **identical action_score (1.0)** but
diverge sharply under GSAR — grounding_score 1.00 (resolved) vs. 0.40
(abstain, contradicted claims flagged). See this module's test file for the
full reproduction.

## What GSAR actually does

[GSAR](https://tulipagents.ai/concepts/gsar/) (arXiv:2604.23366) partitions
a synthesis into atomic claims and labels each against evidence:
``grounded`` / ``ungrounded`` / ``contradicted`` / ``complementary``, then
computes a contradiction-penalized score. Here, the "synthesis" is the
agent's final message to the customer; the "evidence" is the real tool call
results tau2-bench already executed during the episode — this module adds
no new tool calls and makes no claims about ground truth beyond what the
episode's own tools returned.

## Requires

- ``pip install tulip-agents`` (not a dependency of ``tinker-cookbook``
  itself — only of this module, if you choose to import it).
- ``_tau2_bench.py``'s ``ToolBackend.execute`` logging the real result
  alongside each call (not just name/arguments) — already the case as of
  this change; ``logs["agent_final_message"]`` is populated the same way.
- A judge satisfying ``tulip.reasoning.gsar_judge.BaseGSARJudge`` — any
  ``tulip.models.base.BaseModel``-compatible chat model driven through
  ``StructuredOutputGSARJudge``. Deliberately not forced through Tinker's
  own ``SamplingClient``: that's a low-level, token-based sampling API with
  no native structured-output mode, and building a reliable JSON-extraction
  adapter on top of it is unproven — using a model tulip already drives
  reliably (OpenAI, Anthropic, or any OpenAI-compatible endpoint) is the
  honest choice for a first version, not a limitation hidden from the
  reader.

## Example

    from tulip.models.native.openai import OpenAIModel
    from tulip.reasoning.gsar_judge import StructuredOutputGSARJudge
    from tinker_cookbook.eval.tulip_grounding import score_trajectories_file

    judge = StructuredOutputGSARJudge(
        model=OpenAIModel(model="gpt-4o-mini"),  # any tulip-compatible model
    )
    verdicts = await score_trajectories_file(
        "evals/step500/tau2_bench/trajectories.jsonl", judge
    )
    for example_id, verdict in verdicts.items():
        print(example_id, verdict.grounding_score, verdict.decision_status)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tulip.reasoning.gsar_judge import BaseGSARJudge, JudgeOutput


def _build_evidence_corpus(predicted_actions_json: str) -> str:
    """Render the real tool calls + results (already logged by
    ``ToolBackend.execute``) as the evidence corpus GSAR scores the agent's
    final message against."""
    try:
        calls: list[dict[str, Any]] = json.loads(predicted_actions_json)
    except json.JSONDecodeError:
        return predicted_actions_json
    lines = []
    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        result = call.get("result", "<result not logged>")
        lines.append(f"Tool {name}({args}) returned: {result}")
    return "\n".join(lines) if lines else "No tool calls were made this episode."


async def score_trajectory_grounding(logs: dict[str, Any], judge: BaseGSARJudge) -> JudgeOutput:
    """Score one tau2-bench trajectory's ``logs`` dict for groundedness.

    Args:
        logs: A ``StoredTrajectory.logs`` dict, as saved in tau2-bench's
            ``trajectories.jsonl`` — must contain ``predicted_actions``
            (JSON string of ``{name, arguments, result}`` dicts) and
            ``agent_final_message`` (the agent's last turn to the customer).
        judge: Any ``tulip.reasoning.gsar_judge.BaseGSARJudge`` — typically
            a ``StructuredOutputGSARJudge`` wrapping a tulip model.

    Returns:
        The judge's ``JudgeOutput`` — ``.grounding_score``, ``.is_grounded``,
        ``.decision_status``, and the four-way claim partition.
    """
    evidence = _build_evidence_corpus(logs.get("predicted_actions", "[]"))
    final_message = logs.get("agent_final_message", "")
    return await judge.judge(report_synthesis=final_message, evidence_corpus=evidence)


async def score_trajectories_file(path: str, judge: BaseGSARJudge) -> dict[str, JudgeOutput]:
    """Score every line of a local ``trajectories.jsonl`` file.

    Local files only — for a cloud ``save_dir`` (e.g. ``s3://...``), read
    the file yourself (this repo's own ``Storage`` abstraction handles
    that) and call :func:`score_trajectory_grounding` per row instead.

    Args:
        path: Local path to a tau2-bench ``trajectories.jsonl`` file.
        judge: Any ``tulip.reasoning.gsar_judge.BaseGSARJudge``.

    Returns:
        Mapping of ``example_id`` to that trajectory's ``JudgeOutput``.
        Rows with no ``agent_final_message`` in ``logs`` (e.g. an episode
        that ended mid tool-call sequence) are skipped, not scored as 0.
    """
    results: dict[str, JudgeOutput] = {}
    with open(path, encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            logs = row.get("logs", {})
            if not logs.get("agent_final_message"):
                continue
            example_id = row.get("example_id") or f"line_{line_num}"
            results[example_id] = await score_trajectory_grounding(logs, judge)
    return results
