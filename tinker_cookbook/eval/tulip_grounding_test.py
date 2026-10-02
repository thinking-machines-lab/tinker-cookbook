"""Tests for tulip_grounding.

Uses tau2-bench's own real ToolBackend/_check_actions (imported, not
reimplemented) to prove the actual gap this module fills: two transcripts
with the identical real tool call and the identical real tool result --
one honest, one hallucinated -- score identically under tau2-bench's own
action_score, and diverge under GSAR.

The live-judge tests are skipped without an API key -- real, billed calls,
not part of the default suite for that reason. Not gated behind a Tinker
key: the judge is any tulip-compatible model, deliberately not routed
through Tinker's own low-level SamplingClient (see the module docstring).
"""

import json
import os

import pytest

from tinker_cookbook.eval.benchmarks._tau2_bench import ToolBackend, _check_actions

tulip = pytest.importorskip("tulip", reason="tulip-agents is not installed")

from tinker_cookbook.eval.tulip_grounding import (  # noqa: E402
    _build_evidence_corpus,
    score_trajectory_grounding,
)

_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_reservation_details",
            "description": "Get the details of a reservation by reservation id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reservation_id": {
                        "type": "string",
                        "description": "The unique id of the reservation",
                    },
                },
                "required": ["reservation_id"],
            },
        },
    }
]

_DB = {
    "reservations": {
        "ZFA04Y": {
            "reservation_id": "ZFA04Y",
            "passenger_name": "Mei Chen",
            "origin": "SFO",
            "destination": "JFK",
            "status": "confirmed",
            "baggage_allowance": 1,
            "cabin": "economy",
        }
    }
}

_EXPECTED_ACTIONS = [{"name": "get_reservation_details", "arguments": {"reservation_id": "ZFA04Y"}}]

_HONEST = (
    "I found your reservation ZFA04Y — economy cabin, SFO to JFK, "
    "confirmed, with 1 checked bag included."
)
_HALLUCINATED = (
    "I found your reservation ZFA04Y — you're booked in business class, "
    "SFO to JFK, confirmed, with 3 checked bags included at no charge."
)


def _real_call_log() -> list[dict]:
    """Runs the real ToolBackend, exactly the way Tau2MessageEnv does."""
    backend = ToolBackend(_DB, _TOOL_DEFINITIONS)
    backend.execute("get_reservation_details", {"reservation_id": "ZFA04Y"})
    return backend.call_log


def test_call_log_now_includes_the_real_result():
    """The logging change this module depends on: ToolBackend.call_log
    entries carry `result`, not just name/arguments."""
    [entry] = _real_call_log()
    assert entry["name"] == "get_reservation_details"
    assert "result" in entry
    assert "ZFA04Y" in entry["result"]
    assert "economy" in entry["result"]


def test_tau2_action_score_cannot_distinguish_honest_from_hallucinated():
    """The actual gap, demonstrated with tau2-bench's own real grading
    function -- both transcripts make the identical real tool call, so
    both get a perfect action_score regardless of what the agent then
    told the customer."""
    calls = _real_call_log()
    honest_score, _ = _check_actions(calls, _EXPECTED_ACTIONS)
    hallucinated_score, _ = _check_actions(calls, _EXPECTED_ACTIONS)
    assert honest_score == 1.0
    assert hallucinated_score == 1.0
    assert honest_score == hallucinated_score  # identical -- the actual gap


def test_evidence_corpus_renders_the_real_tool_result():
    calls = _real_call_log()
    corpus = _build_evidence_corpus(json.dumps(calls))
    assert "get_reservation_details" in corpus
    assert "economy" in corpus  # the real DB value, not the agent's claim


class TestLiveGrounding:
    """Live-judge tests -- real tulip.reasoning.gsar_judge, real API call."""

    pytestmark = pytest.mark.skipif(
        not (os.environ.get("OPENAI_API_KEY") or os.environ.get("TOGETHER_API_KEY")),
        reason="needs a live OPENAI_API_KEY or TOGETHER_API_KEY -- makes real, billed API calls",
    )

    def _judge(self):
        # tulip-agents isn't installed in this repo's CI (it's an optional
        # dependency of this one test module, not of tinker-cookbook) -- the
        # class this test skips on (pytestmark above) never actually runs
        # there, but pyright still resolves imports regardless of runtime
        # skip markers, so these need an explicit, narrow ignore rather than
        # a real type error.
        import tulip.reasoning.gsar_judge as _gsar_judge  # pyright: ignore[reportMissingImports]

        StructuredOutputGSARJudge = _gsar_judge.StructuredOutputGSARJudge

        if os.environ.get("OPENAI_API_KEY"):
            import tulip.models.native.openai as _openai_model  # pyright: ignore[reportMissingImports]

            OpenAIModel = _openai_model.OpenAIModel
            model = OpenAIModel(model="gpt-4o-mini")
        else:
            import tulip.models.native.openai as _openai_model  # pyright: ignore[reportMissingImports]

            OpenAIModel = _openai_model.OpenAIModel

            model = OpenAIModel(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                api_key=os.environ["TOGETHER_API_KEY"],
                base_url="https://api.together.xyz/v1",
            )
        return StructuredOutputGSARJudge(model=model, strict=False)

    @pytest.mark.asyncio
    async def test_honest_transcript_scores_grounded(self):
        calls = _real_call_log()
        logs = {
            "predicted_actions": json.dumps(calls),
            "agent_final_message": _HONEST,
        }
        verdict = await score_trajectory_grounding(logs, self._judge())
        assert verdict.grounding_score >= 0.8
        assert verdict.decision_status == "resolved"

    @pytest.mark.asyncio
    async def test_hallucinated_transcript_scores_ungrounded_despite_identical_action_score(self):
        """The actual proof: same real tool call, same real result,
        different final message -- and GSAR catches what action_score
        structurally cannot."""
        calls = _real_call_log()
        logs = {
            "predicted_actions": json.dumps(calls),
            "agent_final_message": _HALLUCINATED,
        }
        verdict = await score_trajectory_grounding(logs, self._judge())
        assert verdict.grounding_score < 0.8

    @pytest.mark.asyncio
    async def test_score_trajectories_file_skips_rows_without_a_final_message(self, tmp_path):
        """A row from an episode that ended mid tool-call sequence (no
        agent_final_message) is skipped, not silently scored as 0."""
        from tinker_cookbook.eval.tulip_grounding import score_trajectories_file

        path = tmp_path / "trajectories.jsonl"
        calls = _real_call_log()
        rows = [
            {
                "example_id": "ex1",
                "logs": {"predicted_actions": json.dumps(calls), "agent_final_message": _HONEST},
            },
            {
                "example_id": "ex2",
                "logs": {"predicted_actions": json.dumps(calls)},
            },  # no final message
        ]
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        results = await score_trajectories_file(str(path), self._judge())
        assert set(results.keys()) == {"ex1"}
