import logging
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import patch

from . import ml_log
from .ml_log import configure_logging_module


def _flush_root_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_configure_logging_module_logs_invocation_and_appends(tmp_path):
    log_path = tmp_path / "logs.log"

    argv_first = ["python", "train.py", "--log-path", str(tmp_path), "--run-name", "first run"]
    with patch.object(sys, "argv", argv_first):
        root_logger = configure_logging_module(str(log_path))
        root_logger.info("first message")
        _flush_root_handlers()

    first_contents = log_path.read_text()
    first_invocation = shlex.join(argv_first)
    assert f"Command line invocation: {first_invocation}" in first_contents
    assert "first message" in first_contents
    assert first_contents.index(first_invocation) < first_contents.index("first message")

    argv_second = ["python", "train.py", "--resume", "--run-name", "second run"]
    with patch.object(sys, "argv", argv_second):
        root_logger = configure_logging_module(str(log_path))
        root_logger.info("second message")
        _flush_root_handlers()

    final_contents = log_path.read_text()
    second_invocation = shlex.join(argv_second)
    assert "first message" in final_contents
    assert "second message" in final_contents
    assert f"Command line invocation: {second_invocation}" in final_contents
    assert final_contents.count("Command line invocation:") == 2
    assert final_contents.index("first message") < final_contents.index(second_invocation)
    assert final_contents.index(second_invocation) < final_contents.index("second message")


def test_wandb_logger_log_rollouts_uses_weave_lazily(monkeypatch):
    init_projects = []
    contexts = []
    attrs_seen = []
    records_seen = []

    class FakeClient:
        def set_wandb_run_context(self, *, run_id, step):
            contexts.append((run_id, step))

        def clear_wandb_run_context(self):
            contexts.append("cleared")

    class FakeAttributes:
        def __init__(self, attrs):
            self.attrs = attrs

        def __enter__(self):
            attrs_seen.append(self.attrs)

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeWeave:
        @staticmethod
        def init(project_name):
            init_projects.append(project_name)
            return FakeClient()

        @staticmethod
        def attributes(attrs):
            return FakeAttributes(attrs)

    def fake_log_rollout(record):
        records_seen.append(record)

    logger = object.__new__(ml_log.WandbLogger)
    logger.run = SimpleNamespace(id="abc123")
    logger._weave_client = None
    logger._weave_project_name = "entity/project"
    logger._warned_weave_unavailable = False

    monkeypatch.setattr(ml_log, "_weave_available", True)
    monkeypatch.setattr(ml_log, "weave", FakeWeave)
    monkeypatch.setattr(ml_log, "_weave_log_rollout", fake_log_rollout)

    record = {
        "split": "train",
        "iteration": 2,
        "group_idx": 3,
        "traj_idx": 4,
        "tags": ["unit-test"],
        "steps": [{"ac_len": 2, "ac_text": "hello"}],
    }
    logger.log_rollouts([record], step=2)

    assert init_projects == ["entity/project"]
    assert contexts == [("abc123", 2), "cleared"]
    assert attrs_seen == [
        {
            "split": "train",
            "iteration": 2,
            "group_idx": 3,
            "traj_idx": 4,
            "tags": ["unit-test"],
        }
    ]
    assert records_seen == [record]


def test_rollout_for_weave_preserves_full_trajectory():
    record = {
        "schema_version": 1,
        "split": "train",
        "iteration": 0,
        "group_idx": 1,
        "traj_idx": 2,
        "tags": ["unit-test"],
        "sampling_client_step": 0,
        "total_reward": 2.0,
        "final_reward": 0.5,
        "trajectory_metrics": {"success": 1},
        "steps": [
            {
                "step_idx": 0,
                "ob_len": 3,
                "ac_len": 2,
                "reward": 0.5,
                "episode_done": False,
                "metrics": {"score": 0.5},
                "logs": {"tool": "none"},
                "ob_text": "User: first question",
                "ac_text": "Assistant: first answer",
                "stop_reason": "stop",
            },
            {
                "step_idx": 1,
                "ob_len": 7,
                "ac_len": 4,
                "reward": 1.0,
                "episode_done": True,
                "metrics": {"score": 1.0},
                "logs": {"tool": "calculator"},
                "ob_text": "User: follow-up",
                "ac_text": "Assistant: final answer",
                "stop_reason": "length",
            },
        ],
        "final_ob_len": 11,
    }

    view = ml_log._rollout_for_weave(record)

    assert view["total_reward"] == 2.0
    assert view["turns"] == record["steps"]
    assert sorted(view) == [
        "final_ob_len",
        "final_reward",
        "group_idx",
        "iteration",
        "sampling_client_step",
        "schema_version",
        "split",
        "tags",
        "total_reward",
        "traj_idx",
        "trajectory_metrics",
        "turns",
    ]
