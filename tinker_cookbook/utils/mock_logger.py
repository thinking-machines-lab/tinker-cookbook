"""Test utilities for tinker cookbook tests."""

import json
import os
from typing import Any
from unittest.mock import MagicMock


def create_mock_logger_with_jsonl(
    log_path: str,
    interrupt_at_step: int | None = None,
    interrupt_exception_class: type[Exception] | None = None,
    metrics_filename: str = "metrics.jsonl",
) -> MagicMock:
    """Create a mock logger that writes metrics to JSONL and optionally interrupts at a specific step.

    Args:
        log_path: Directory to write metrics file to
        interrupt_at_step: If provided, raise exception at this step
        interrupt_exception_class: Exception class to raise (required if interrupt_at_step is set)
        metrics_filename: Name of the metrics file to write

    Returns:
        Mock logger object
    """
    mock_logger = MagicMock()

    def log_metrics(metrics: dict[str, Any], step: int):
        # Write to JSONL file
        jsonl_path = os.path.join(log_path, metrics_filename)
        with open(jsonl_path, "a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
            print(f"Step {step} metrics: {metrics}")

        # Interrupt if requested
        if interrupt_at_step is not None and step == interrupt_at_step:
            if interrupt_exception_class is None:
                raise ValueError(
                    "interrupt_exception_class must be provided if interrupt_at_step is set"
                )
            raise interrupt_exception_class(f"Interrupting at step {step}")

    mock_logger.log_metrics = log_metrics
    mock_logger.close = MagicMock()
    mock_logger.get_logger_url = MagicMock(return_value=None)

    return mock_logger
