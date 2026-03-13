import logging
import runpy
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

configure_logging_module = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "utils" / "ml_log.py")
)["configure_logging_module"]


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
