"""Download checkpoint weights from Tinker storage."""

from __future__ import annotations

import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def download(*, tinker_path: str, output_dir: str) -> str:
    """Download a checkpoint from Tinker storage to local disk.

    Fetches a signed URL via the Tinker SDK, downloads the archive, and
    extracts it with security validation (rejects symlinks and path
    traversal).

    Args:
        tinker_path: Tinker checkpoint path, e.g.
            ``"tinker://<run_id>/sampler_weights/final"``.
        output_dir: Local directory where the checkpoint will be extracted.

    Returns:
        Path to the extracted checkpoint directory.
    """
    import tinker

    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    response = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(response.url, tmp_path)
        _safe_extract_tar(tmp_path, Path(output_dir))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return output_dir


def _safe_extract_tar(archive_path: str, extract_dir: Path) -> None:
    """Extract a tar archive with security validation.

    Rejects archives containing symlinks, hardlinks, or paths that escape
    the extraction directory (path traversal).
    """
    base = extract_dir.resolve()
    with tarfile.open(archive_path, "r") as tar:
        members = tar.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise ValueError(
                    "Unsafe symlink or hardlink found in tar archive. "
                    "Archive may be corrupted or malicious."
                )
            member_path = (extract_dir / member.name).resolve()
            if not str(member_path).startswith(str(base)):
                raise ValueError(
                    "Unsafe path found in tar archive (path traversal). "
                    "Archive may be corrupted or malicious."
                )
        tar.extractall(path=extract_dir)
