"""Token DB: persist every raw token exchanged during rollouts.

Parquet segments written through the ``Storage`` protocol, behind the
storage-agnostic :class:`TokenStoreBackend` protocol. See ``schema.py`` for
the row model and ``writer.py`` for the segment/manifest layout.
"""

from tinker_cookbook.tokendb.interface import TokenStoreBackend, TokenWriter
from tinker_cookbook.tokendb.schema import (
    SCHEMA_VERSION,
    TokenRow,
    arrow_schema,
    compute_ob_delta,
)
from tinker_cookbook.tokendb.writer import ParquetSegmentBackend, TokenDbWriter, make_writer_id

__all__ = [
    "SCHEMA_VERSION",
    "ParquetSegmentBackend",
    "TokenDbWriter",
    "TokenRow",
    "TokenStoreBackend",
    "TokenWriter",
    "arrow_schema",
    "compute_ob_delta",
    "make_writer_id",
]
