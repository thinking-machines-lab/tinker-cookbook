"""Transparent local / GCS file I/O.

All functions accept either a local path or a ``gs://bucket/key`` URI.
The scheme is detected automatically -- callers never branch on storage type.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlparse


def _is_gcs(path: str) -> bool:
    return urlparse(path).scheme == "gs"


def _gcs_bucket_blob(path: str):
    from google.cloud import storage as gcs_storage

    parsed = urlparse(path)
    client = gcs_storage.Client()
    bucket = client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip("/"))
    return bucket, blob


# -- core primitives -----------------------------------------------------------


def read_bytes(path: str) -> bytes:
    """Read entire file. Returns ``b""`` if the file does not exist."""
    if _is_gcs(path):
        _, blob = _gcs_bucket_blob(path)
        if blob.exists():
            return blob.download_as_bytes()
        return b""
    if not os.path.exists(path):
        return b""
    with open(path, "rb") as f:
        return f.read()


def write_bytes(path: str, data: bytes) -> None:
    """Atomically overwrite a file (local: tmp+rename, GCS: upload)."""
    if _is_gcs(path):
        _, blob = _gcs_bucket_blob(path)
        blob.upload_from_string(data)
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def append_line(path: str, line: str) -> None:
    """Append a single text line (caller supplies trailing newline)."""
    if _is_gcs(path):
        existing = read_bytes(path)
        write_bytes(path, existing + line.encode())
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(line)
            f.flush()


def read_text(path: str) -> str:
    """Read entire file as UTF-8 text. Returns ``""`` if missing."""
    return read_bytes(path).decode("utf-8", errors="replace")


def makedirs(path: str) -> None:
    """Ensure directory exists. No-op for GCS paths."""
    if not _is_gcs(path):
        os.makedirs(path, exist_ok=True)


# -- JSONL helpers -------------------------------------------------------------


def write_json(path: str, data: dict[str, Any]) -> None:
    """Atomically write a single JSON object to *path*."""
    payload = json.dumps(data, separators=(",", ":")) + "\n"
    write_bytes(path, payload.encode())


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    """Append one JSON record as a JSONL line."""
    line = json.dumps(record, separators=(",", ":")) + "\n"
    append_line(path, line)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Read all JSONL records from *path*. Returns ``[]`` if missing."""
    text = read_text(path)
    if not text:
        return []
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# -- path helpers --------------------------------------------------------------


def join(base: str, *parts: str) -> str:
    """Join path segments. Works for both local and GCS paths."""
    if _is_gcs(base):
        return base.rstrip("/") + "/" + "/".join(parts)
    return os.path.join(base, *parts)
