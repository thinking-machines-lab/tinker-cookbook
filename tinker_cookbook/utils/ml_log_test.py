import logging
import shlex
import sys
from unittest.mock import patch

import pytest

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


def test_wandb_log_summary_updates_the_summary_not_the_step_history():
    """Summary values between steps must not open or close a W&B step."""
    from unittest.mock import MagicMock

    from . import ml_log

    fake_wandb = MagicMock()
    with patch.object(ml_log, "wandb", fake_wandb), patch.object(ml_log, "_wandb_available", True):
        logger = ml_log.WandbLogger(project="p")
        logger.log_metrics({"loss": 1.0}, step=3)
        logger.log_summary({"tokens/cumulative/sampled": 10})
        logger.log_summary({"tokens/cumulative/sampled": 20})
        logger.log_metrics({"loss": 0.5}, step=3)

    assert fake_wandb.log.call_args_list == [
        (({"loss": 1.0},), {"step": 3}),
        (({"loss": 0.5},), {"step": 3}),
    ]
    assert fake_wandb.init.return_value.summary.update.call_args_list == [
        (({"tokens/cumulative/sampled": 10},), {}),
        (({"tokens/cumulative/sampled": 20},), {}),
    ]


def test_json_logger_writes_summary_rows_without_a_step(tmp_path):
    import json

    from . import ml_log

    logger = ml_log.JsonLogger(tmp_path)
    logger.log_metrics({"loss": 1.0}, step=3)
    logger.log_summary({"tokens/cumulative/sampled": 10})
    with open(tmp_path / "metrics.jsonl") as f:
        rows = [json.loads(line) for line in f]
    assert rows == [{"step": 3, "loss": 1.0}, {"tokens/cumulative/sampled": 10}]
    ml_log.MultiplexLogger([logger, ml_log.PrettyPrintLogger()]).log_summary({"a": 1})


def test_append_events_keep_late_eval_in_history_and_preserve_json_steps(tmp_path):
    import json
    from unittest.mock import MagicMock, call

    from . import ml_log

    fake_wandb = MagicMock()
    with patch.object(ml_log, "wandb", fake_wandb), patch.object(ml_log, "_wandb_available", True):
        wandb_logger = ml_log.WandbLogger(project="p")
        logger = ml_log.MultiplexLogger([wandb_logger, ml_log.JsonLogger(tmp_path)])
        logger.enable_event_steps()
        logger.log_metrics({"loss": 0.4}, step=61)
        fake_wandb.log.assert_called_once_with({"loss": 0.4, "training/step": 61}, commit=True)
        logger.log_metrics_event(
            {"grader_eval/reward": 0.8}, step=0, step_metric="eval/snapshot/training_step"
        )
        logger.log_metrics({"loss": 0.3}, step=62)
        logger.close()
        assert fake_wandb.log.call_args_list == [
            call({"loss": 0.4, "training/step": 61}, commit=True),
            call({"grader_eval/reward": 0.8, "eval/snapshot/training_step": 0}, commit=True),
            call({"loss": 0.3, "training/step": 62}, commit=True),
        ]
        fake_wandb.init.return_value.summary.update.assert_not_called()
        fake_wandb.init.return_value.define_metric.assert_any_call(
            "grader_eval/reward", step_metric="eval/snapshot/training_step", step_sync=False
        )
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [61, 0, 62]
    assert rows[1]["eval/snapshot/training_step"] == 0


@pytest.fixture(autouse=True)
def _clear_wandb_identity_environment(monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)


def test_wandb_event_history_continues_across_restart(tmp_path):
    from unittest.mock import MagicMock, call

    from . import ml_log

    fake_wandb = MagicMock()
    fake_wandb.init.return_value.id = "run-one"
    fake_wandb.init.return_value.project = "p"
    fake_wandb.init.return_value.entity = "team"
    fake_wandb.init.return_value.settings.mode = "online"
    with patch.object(ml_log, "wandb", fake_wandb), patch.object(ml_log, "_wandb_available", True):
        first = ml_log.setup_logging(
            str(tmp_path),
            wandb_project="p",
            do_configure_logging_module=False,
        )
        first.enable_event_steps()
        first.log_metrics({"loss": 0.4}, step=40)
        first.log_metrics({"loss": 0.3}, step=41)
        first.close()
        second = ml_log.setup_logging(
            str(tmp_path),
            wandb_project="p",
            do_configure_logging_module=False,
        )
        second.log_metrics({"loss": 0.35}, step=40)
        second.log_metrics_event(
            {
                "eval/reward": 0.8,
                "eval/by_cost/reward": 0.8,
                "eval/by_time/reward": 0.8,
                "eval/cost": 2.5,
                "eval/time": 7.0,
            },
            step=39,
            step_metric="eval/step",
            metric_axes={"eval/by_cost/reward": "eval/cost", "eval/by_time/reward": "eval/time"},
        )
        second.close()

    assert fake_wandb.init.call_args.kwargs == {
        "project": "p",
        "config": None,
        "dir": str(tmp_path),
        "name": None,
        "id": "run-one",
        "entity": "team",
        "resume": "allow",
    }
    assert fake_wandb.log.call_args_list[:3] == [
        call({"loss": 0.4, "training/step": 40}, commit=True),
        call({"loss": 0.3, "training/step": 41}, commit=True),
        call({"loss": 0.35, "training/step": 40}, commit=True),
    ]
    assert fake_wandb.log.call_args.kwargs == {"commit": True}
    run = fake_wandb.init.return_value
    for metric, coordinate in [
        ("*", "training/step"),
        ("eval/reward", "eval/step"),
        ("eval/by_cost/reward", "eval/cost"),
        ("eval/by_time/reward", "eval/time"),
    ]:
        run.define_metric.assert_any_call(metric, step_metric=coordinate, step_sync=False)


@pytest.mark.parametrize("changed", ["project", "entity", "id"])
def test_wandb_resume_rejects_conflicting_identity(tmp_path, monkeypatch, changed):
    import json
    from unittest.mock import MagicMock

    from tinker_cookbook.exceptions import ConfigurationError

    from . import ml_log

    (tmp_path / "wandb_run.json").write_text(
        json.dumps({"id": "original", "project": "p", "entity": "team"})
    )
    if changed != "project":
        monkeypatch.setenv("WANDB_ENTITY" if changed == "entity" else "WANDB_RUN_ID", "different")
    fake_wandb = MagicMock()
    with patch.object(ml_log, "wandb", fake_wandb), patch.object(ml_log, "_wandb_available", True):
        with pytest.raises(ConfigurationError, match=r"W&B.*identity"):
            ml_log.setup_logging(
                str(tmp_path),
                wandb_project="different" if changed == "project" else "p",
                do_configure_logging_module=False,
            )
    fake_wandb.init.assert_not_called()


@pytest.mark.parametrize("state", ["{", "[]", "{}", '{"id": "", "project": "p", "entity": "team"}'])
def test_wandb_resume_rejects_invalid_identity(tmp_path, state):
    from unittest.mock import MagicMock

    from tinker_cookbook.exceptions import ConfigurationError

    from . import ml_log

    (tmp_path / "wandb_run.json").write_text(state)
    fake_wandb = MagicMock()
    with patch.object(ml_log, "wandb", fake_wandb), patch.object(ml_log, "_wandb_available", True):
        with pytest.raises(ConfigurationError, match="Invalid W&B run identity"):
            ml_log.setup_logging(str(tmp_path), wandb_project="p")
    fake_wandb.init.assert_not_called()


@pytest.mark.parametrize("mode", ["offline", "disabled"])
def test_wandb_does_not_save_unresumable_identity(tmp_path, mode):
    from unittest.mock import MagicMock

    from . import ml_log

    fake_wandb = MagicMock()
    fake_wandb.init.return_value.settings.mode = mode
    with patch.object(ml_log, "wandb", fake_wandb), patch.object(ml_log, "_wandb_available", True):
        for _ in range(2):
            logger = ml_log.setup_logging(
                str(tmp_path),
                wandb_project="p",
                do_configure_logging_module=False,
            )
            logger.enable_event_steps()
            logger.close()
    assert not (tmp_path / "wandb_run.json").exists()
    assert fake_wandb.init.call_args.kwargs["id"] is None
    assert fake_wandb.init.call_args.kwargs["resume"] is None


def test_non_event_logging_keeps_explicit_steps_without_saving_identity(tmp_path):
    from unittest.mock import MagicMock, call

    from . import ml_log

    state = tmp_path / "wandb_run.json"
    fake_wandb = MagicMock()
    with patch.object(ml_log, "wandb", fake_wandb), patch.object(ml_log, "_wandb_available", True):
        for step in (41, 40):
            logger = ml_log.setup_logging(
                str(tmp_path),
                wandb_project="p",
                do_configure_logging_module=False,
            )
            logger.log_metrics({"loss": 0.3}, step=step)
            logger.close()
    assert fake_wandb.init.call_args.kwargs["id"] is None
    assert fake_wandb.init.call_args.kwargs["resume"] is None
    assert fake_wandb.log.call_args_list == [
        call({"loss": 0.3}, step=41),
        call({"loss": 0.3}, step=40),
    ]
    assert not state.exists()


def test_resumed_hparams_can_update_real_wandb_config(tmp_path):
    import json
    from unittest.mock import MagicMock

    pytest.importorskip("wandb")
    from wandb.sdk.wandb_config import Config

    from . import ml_log

    (tmp_path / "wandb_run.json").write_text(
        json.dumps({"id": "original", "project": "p", "entity": "team"})
    )
    fake_wandb = MagicMock()
    fake_wandb.config = Config()
    fake_wandb.config.update({"learning_rate": 1e-4})
    with patch.object(ml_log, "wandb", fake_wandb), patch.object(ml_log, "_wandb_available", True):
        logger = ml_log.setup_logging(
            str(tmp_path),
            wandb_project="p",
            config={"learning_rate": 2e-4},
            do_configure_logging_module=False,
        )
        logger.close()
    assert fake_wandb.config["learning_rate"] == 2e-4


def test_resume_failure_explains_identity_recovery(tmp_path):
    import json
    from unittest.mock import MagicMock

    from tinker_cookbook.exceptions import ConfigurationError

    from . import ml_log

    state = json.dumps({"id": "original", "project": "p", "entity": "team"})
    (tmp_path / "wandb_run.json").write_text(state)
    fake_wandb = MagicMock()
    fake_wandb.init.side_effect = RuntimeError("run was deleted")
    with patch.object(ml_log, "wandb", fake_wandb), patch.object(ml_log, "_wandb_available", True):
        with pytest.raises(
            ConfigurationError, match=r"Failed to resume W&B run team/p/original.*"
        ) as exc:
            ml_log.setup_logging(str(tmp_path), wandb_project="p")
    assert "training checkpoints are unchanged" in str(exc.value)
    assert (tmp_path / "wandb_run.json").read_text() == state


def test_wandb_identity_persists_through_cloud_storage():
    from unittest.mock import MagicMock
    from uuid import uuid4

    pytest.importorskip("fsspec")
    from . import ml_log

    log_dir = f"memory://wandb-resume/{uuid4().hex}"
    fake_wandb = MagicMock()
    fake_wandb.init.return_value.id = "original"
    fake_wandb.init.return_value.project = "p"
    fake_wandb.init.return_value.entity = "team"
    fake_wandb.init.return_value.settings.mode = "online"
    with patch.object(ml_log, "wandb", fake_wandb), patch.object(ml_log, "_wandb_available", True):
        for _ in range(2):
            logger = ml_log.setup_logging(
                log_dir,
                wandb_project="p",
                do_configure_logging_module=False,
            )
            logger.enable_event_steps()
            logger.close()
    assert fake_wandb.init.call_args.kwargs["id"] == "original"
    assert fake_wandb.init.call_args.kwargs["resume"] == "allow"
    assert fake_wandb.init.call_args.kwargs["dir"] is None
