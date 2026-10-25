"""Local capture store daemon: ingest, query, and SSE streaming.

``tinker_cookbook.stores`` remains the byte/run layer; this daemon is the
service boundary over capture data. See ``store/README.md``.
"""

from tinker_cookbook.capture.store.client import StoreSink, capture_to_store, ensure_daemon
from tinker_cookbook.capture.store.db import CaptureDB

__all__ = ["CaptureDB", "StoreSink", "capture_to_store", "ensure_daemon"]
