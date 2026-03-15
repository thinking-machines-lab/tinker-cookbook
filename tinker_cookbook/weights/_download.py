"""Download checkpoint weights from Tinker storage."""

from __future__ import annotations

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

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tmp_path = Path(tempfile.mktemp(suffix=".tar"))
    try:
        urllib.request.urlretrieve(response.url, str(tmp_path))
        _safe_extract_tar(tmp_path, out)
    finally:
        tmp_path.unlink(missing_ok=True)

    return output_dir


def _safe_extract_tar(archive_path: Path, extract_dir: Path) -> None:
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
            if not member_path.is_relative_to(base):
                raise ValueError(
                    "Unsafe path found in tar archive (path traversal). "
                    "Archive may be corrupted or malicious."
                )
        tar.extractall(path=extract_dir)
