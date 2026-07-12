"""Tests for code_rl token DB capture wiring (builder metadata, grading capture).

These exercise pure logic with the sandbox call stubbed out; no network or
Docker dependencies.
"""

import asyncio
import json
from typing import Any

from tinker_cookbook.recipes.code_rl.code_env import DeepcoderEnvGroupBuilder
from tinker_cookbook.recipes.code_rl.deepcoder_tool import (
    DeepcoderReward,
    DeepcoderTask,
    _grading_details_summary,
)

_TESTS: list[dict[str, Any]] = [
    {"input": "1\n", "output": "2\n", "testtype": "stdin_stdout", "metadata": {"func_name": None}},
    {"input": "2\n", "output": "3\n", "testtype": "stdin_stdout", "metadata": {"func_name": None}},
]


class TestBuilderMetadata:
    def _builder(self, source: str | None) -> DeepcoderEnvGroupBuilder:
        return DeepcoderEnvGroupBuilder(
            task=DeepcoderTask(problem="p", tests=_TESTS, source=source),
            model_name="m",
            renderer_name=None,
            max_turns=1,
            group_size=1,
            sandbox_backend=None,
        )

    def test_metadata_with_source(self):
        assert self._builder("taco").metadata() == {"dataset": "deepcoder", "source": "taco"}

    def test_metadata_without_source(self):
        assert self._builder(None).metadata() == {"dataset": "deepcoder"}


class TestGradingDetailsSummary:
    def test_keeps_verdict_error_and_timing_drops_output(self):
        details = {
            "status": "Failed",
            "run_result": {
                "status": "Finished",
                "execution_time": 0.5,
                "return_code": 255,
                "stdout": "huge test harness output",
                "stderr": "traceback ...",
            },
        }
        summary = _grading_details_summary(details)
        assert summary["status"] == "Failed"
        assert summary["run_result"] == {
            "status": "Finished",
            "execution_time": 0.5,
            "return_code": 255,
        }

    def test_error_payload(self):
        assert _grading_details_summary({"error": "boom"}) == {"error": "boom"}


class TestDeepcoderRewardCapture:
    def _grade(
        self,
        monkeypatch,
        response: str,
        sandbox_result: tuple[bool, dict[str, Any]] | None = None,
        sandbox_error: Exception | None = None,
    ):
        async def fake_check(tests, code, timeout=6, backend=None):
            if sandbox_error is not None:
                raise sandbox_error
            assert sandbox_result is not None
            return sandbox_result

        monkeypatch.setattr(
            "tinker_cookbook.recipes.code_rl.deepcoder_tool.sandbox_check_correctness",
            fake_check,
        )
        reward_fn = DeepcoderReward(task=DeepcoderTask(problem="p", tests=_TESTS))
        return asyncio.run(reward_fn([{"role": "assistant", "content": response}]))

    def test_passed_metrics_and_grading_logs(self, monkeypatch):
        reward, metrics, logs = self._grade(
            monkeypatch,
            "```python\nprint(2)\n```",
            sandbox_result=(
                True,
                {
                    "status": "Success",
                    "run_result": {
                        "status": "Finished",
                        "execution_time": 0.25,
                        "return_code": 0,
                        "stdout": "should be dropped",
                    },
                },
            ),
        )
        assert reward == 1.0
        assert metrics == {"format": 1.0, "correct": 1.0, "tests_total": 2.0}
        assert logs["grading/verdict"] == "passed"
        details = json.loads(str(logs["grading/details"]))
        assert details["status"] == "Success"
        assert details["run_result"]["execution_time"] == 0.25
        assert "stdout" not in details["run_result"]

    def test_failed_run_verdict(self, monkeypatch):
        reward, metrics, logs = self._grade(
            monkeypatch,
            "```python\nprint(1)\n```",
            sandbox_result=(False, {"status": "Failed"}),
        )
        assert metrics["correct"] == 0.0
        assert metrics["tests_total"] == 2.0
        assert logs["grading/verdict"] == "failed"

    def test_no_code_block(self, monkeypatch):
        reward, metrics, logs = self._grade(monkeypatch, "no code here")
        assert reward == -0.1
        assert metrics == {"format": 0.0, "correct": 0.0, "tests_total": 2.0}
        assert logs["grading/verdict"] == "no_code_block"

    def test_sandbox_error_records_error_class(self, monkeypatch):
        reward, metrics, logs = self._grade(
            monkeypatch,
            "```python\nprint(1)\n```",
            sandbox_error=RuntimeError("sandbox down"),
        )
        assert metrics["correct"] == 0.0
        assert logs["grading/verdict"] == "error"
        assert logs["grading/error_type"] == "RuntimeError"
