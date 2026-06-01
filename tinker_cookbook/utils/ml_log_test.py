import logging
import shlex
import sys
from unittest.mock import patch

from tinker_cookbook.stores.storage import LocalStorage
from tinker_cookbook.stores.training_store import TrainingRunStore

from .ml_log import JsonLogger, configure_logging_module


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


def test_json_logger_close_flushes_store(tmp_path):
    class FlushCountingStorage(LocalStorage):
        def __init__(self, root):
            super().__init__(root)
            self.flush_count = 0

        def flush(self) -> None:
            self.flush_count += 1

    storage = FlushCountingStorage(tmp_path)
    logger = JsonLogger(tmp_path, store=TrainingRunStore(storage))

    logger.log_metrics({"loss": 1.0}, step=0)
    logger.sync()
    logger.close()

    assert storage.flush_count == 2
