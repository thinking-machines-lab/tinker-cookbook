"""Token DB: persist every raw token exchanged during rollouts.

Parquet segments written through the ``Storage`` protocol, behind the
storage-agnostic :class:`TokenStoreBackend` protocol. See ``schema.py`` for
the row model and ``writer.py`` for the segment/manifest layout.
"""

from tinker_cookbook.tokendb.capture import (
    ActiveCapture,
    CaptureContext,
    active_capture_filtered_sink,
    get_active_capture,
    get_capture_context,
    get_filtered_group_sink,
    record_groups,
    record_groups_to_active_capture,
    set_active_capture,
    set_capture_context,
    set_filtered_group_sink,
)
from tinker_cookbook.tokendb.config import (
    TokenDbConfig,
    build_token_db_writer,
    check_token_db_dependencies,
)
from tinker_cookbook.tokendb.interface import TokenStoreBackend, TokenWriter
from tinker_cookbook.tokendb.reader import (
    ParquetSegmentReader,
    TokenDB,
    reconstruct_full_ob,
)
from tinker_cookbook.tokendb.schema import (
    SCHEMA_VERSION,
    TokenRow,
    arrow_schema,
    compute_ob_delta,
)
from tinker_cookbook.tokendb.writer import ParquetSegmentBackend, TokenDbWriter, make_writer_id

__all__ = [
    "SCHEMA_VERSION",
    "ActiveCapture",
    "CaptureContext",
    "ParquetSegmentBackend",
    "ParquetSegmentReader",
    "TokenDB",
    "TokenDbConfig",
    "TokenDbWriter",
    "TokenRow",
    "TokenStoreBackend",
    "TokenWriter",
    "active_capture_filtered_sink",
    "arrow_schema",
    "build_token_db_writer",
    "check_token_db_dependencies",
    "compute_ob_delta",
    "get_active_capture",
    "get_capture_context",
    "get_filtered_group_sink",
    "make_writer_id",
    "reconstruct_full_ob",
    "record_groups",
    "record_groups_to_active_capture",
    "set_active_capture",
    "set_capture_context",
    "set_filtered_group_sink",
]
