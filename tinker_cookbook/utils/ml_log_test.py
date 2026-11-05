import logging
import shlex
import sys
from unittest.mock import patch

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
