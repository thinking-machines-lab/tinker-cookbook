"""Download checkpoint weights from Tinker storage."""

from __future__ import annotations

import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import tinker


def download(*, tinker_path: str, output_dir: str, base_url: str | None = None) -> str:
    """Download a checkpoint from Tinker storage to local disk.

    Fetches a signed URL via the Tinker SDK, downloads the archive, and
    extracts it with security validation (rejects symlinks and path
    traversal).

    Args:
        tinker_path: Tinker checkpoint path, e.g.
            ``"tinker://<run_id>/sampler_weights/final"``.
        output_dir: Local directory where the checkpoint will be extracted.
        base_url: Custom Tinker service URL. If ``None`` (default), uses
            the default Tinker service endpoint (or ``TINKER_BASE_URL``
            environment variable if set).

    Returns:
        Path to the extracted checkpoint directory.

    Raises:
        ValueError: If the archive contains unsafe entries.
        urllib.error.URLError: If the download fails.

    Example::

        from tinker_cookbook import weights

        # Download from default Tinker service
        adapter_dir = weights.download(
            tinker_path="tinker://run-id/sampler_weights/final",
            output_dir="./adapter",
        )

        # Download from a custom Tinker deployment
        adapter_dir = weights.download(
            tinker_path="tinker://run-id/sampler_weights/final",
            output_dir="./adapter",
            base_url="https://tinker.my-company.com",
        )
    """
    kwargs: dict = {}
    if base_url is not None:
        kwargs["base_url"] = base_url
    try:
        sc = tinker.ServiceClient(**kwargs)
        rc = sc.create_rest_client()
    except Exception as e:
        raise RuntimeError(
            "Failed to connect to Tinker service. "
            "Ensure TINKER_API_KEY is set and the service is reachable."
        ) from e

    try:
        response = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()
    except Exception as e:
        raise RuntimeError(
            f"Failed to get download URL for {tinker_path!r}. "
            f"Check that the checkpoint path is valid and the checkpoint has not expired."
        ) from e

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        try:
            urllib.request.urlretrieve(response.url, str(tmp_path))
        except urllib.error.URLError as e:
            raise RuntimeError(
                "Failed to download checkpoint archive from signed URL. "
                "The URL may have expired — try downloading again."
            ) from e
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
