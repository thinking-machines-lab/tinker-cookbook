"""Token DB capture configuration for training loops.

This module is intentionally lightweight (no pyarrow import) so training
configs can reference :class:`TokenDbConfig` without the ``tokendb`` extra
installed; :func:`check_token_db_dependencies` is the config-time gate that
turns a missing dependency into an actionable error before training starts.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import chz

from tinker_cookbook.exceptions import ConfigurationError

if TYPE_CHECKING:
    from tinker_cookbook.stores.storage import Storage
    from tinker_cookbook.tokendb.writer import TokenDbWriter


@chz.chz
class TokenDbConfig:
    """Configuration for token DB capture during training.

    Attributes:
        store_text (bool): Store decoded ``ob_text`` / ``ac_text`` columns
            alongside the canonical token IDs.
        flush_interval_s (float): Background flush period for the writer.
        buffer_rows (int): Flush to a new segment when the buffer reaches
            this many rows.
        capture_filtered (bool): Also capture trajectory groups dropped by
            the rollout pipeline (constant reward, rollout errors) as
            ``source="filtered"`` rows. Note: with a cross-process rollout
            executor, filtered groups are dropped inside worker processes
            and are not captured (known v1 gap).
    """

    store_text: bool = True
    flush_interval_s: float = 5.0
    buffer_rows: int = 2048
    capture_filtered: bool = True


def check_token_db_dependencies() -> None:
    """Raise :class:`ConfigurationError` if the token DB write path can't run.

    Called at config/startup time (never mid-run) so a missing optional
    dependency fails fast with an actionable message.
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise ConfigurationError(
            "token_db is enabled but pyarrow is not installed. "
            "Install the token DB extra with: pip install 'tinker-cookbook[tokendb]'"
        ) from e


def build_token_db_writer(
    config: TokenDbConfig,
    storage_or_log_path: Storage | str | Path,
    *,
    model_name: str | None = None,
    recipe_name: str | None = None,
    extra_context: dict[str, Any] | None = None,
) -> TokenDbWriter:
    """Construct a :class:`TokenDbWriter` for a training run.

    Run identity (``run_id`` from the resolved log path + start timestamp,
    ``run_attempt`` incremented on resume) is handled by the writer itself;
    this helper records provenance (``model_name``, ``recipe_name``, capture
    config) into ``run.json`` via the writer context.

    Args:
        config: Capture configuration (buffer/flush knobs recorded in
            ``run.json``).
        storage_or_log_path: The run's ``Storage`` backend or ``log_path``.
        model_name: Model being trained (for tokenizer recovery by readers).
        recipe_name: Recipe/config provenance.
        extra_context: Additional metadata to record in ``run.json``.
    """
    from tinker_cookbook.tokendb.writer import TokenDbWriter

    context: dict[str, Any] = {
        "model_name": model_name,
        "recipe_name": recipe_name,
        "capture": {
            "store_text": config.store_text,
            "flush_interval_s": config.flush_interval_s,
            "buffer_rows": config.buffer_rows,
            "capture_filtered": config.capture_filtered,
        },
    }
    if extra_context:
        context.update(extra_context)
    return TokenDbWriter(
        storage_or_log_path,
        context=context,
        buffer_rows=config.buffer_rows,
        flush_interval_s=config.flush_interval_s,
    )
