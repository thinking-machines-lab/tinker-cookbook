"""Download checkpoint weights from Tinker storage."""

from __future__ import annotations

import datetime
import logging
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pyrage
import tinker

from tinker_cookbook.exceptions import WeightsDownloadError

if TYPE_CHECKING:
    from tinker.lib.public_interfaces.rest_client import RestClient
    from tinker.types import ExternalWeightsUrlsResponse

logger = logging.getLogger(__name__)

# The extension module ships no type information.
_pyrage: Any = pyrage
_AGE_SUFFIX = ".age"

_URL_REFRESH_MARGIN = datetime.timedelta(minutes=5)

_TEMPORARY_EXPORT_TTL_SECONDS = 3600


def download(*, tinker_path: str, output_dir: str, base_url: str | None = None) -> str:
    """Download a checkpoint from Tinker storage to local disk.

    The path type selects how the files are fetched:

    - ``tinker://<run_id>/external_weights/<name>``, as returned by
      ``TrainingClient.save_weights_external()``: each file is downloaded
      from its own signed URL into ``output_dir``. Files of an
      age-encrypted export keep their ``.age`` suffix; decrypt them with
      ``age -d`` before you use them.
    - ``tinker://<run_id>/weights/<name>``, as returned by
      ``TrainingClient.save_state()``: the checkpoint is loaded into a new
      training run and saved with ``save_weights_external()``, encrypted to a
      key generated for this download. The files are downloaded and decrypted,
      and then the export is deleted. This starts a training session on the
      checkpoint's base model.

    Sampler checkpoints (``tinker://<run_id>/sampler_weights/<name>``) cannot
    be downloaded. Save a training checkpoint or external weights instead.

    File paths are validated before they are written (rejects path traversal).

    Args:
        tinker_path: Tinker checkpoint path (see above).
        output_dir: Local directory where the checkpoint files are written.
        base_url: Custom Tinker service URL. If ``None`` (default), uses
            the default Tinker service endpoint (or ``TINKER_BASE_URL``
            environment variable if set).

    Returns:
        ``output_dir``.

    Raises:
        WeightsDownloadError: If the path is invalid, the service cannot be
            reached, a checkpoint cannot be exported, a file fails to download or decrypt, or a
            file path is unsafe.

    Example::

        from tinker_cookbook import weights

        tinker_path = training_client.save_weights_external("final").result().path
        adapter_dir = weights.download(tinker_path=tinker_path, output_dir="./adapter")
    """
    try:
        parsed_path = tinker.ParsedCheckpointTinkerPath.from_tinker_path(tinker_path)
    except ValueError as e:
        raise WeightsDownloadError(f"Invalid tinker path: {tinker_path!r}") from e
    if parsed_path.checkpoint_type == "sampler":
        raise WeightsDownloadError(
            f"Sampler checkpoints cannot be downloaded: {tinker_path!r}. Download a training "
            "checkpoint (tinker://<run_id>/weights/<name>) or external weights saved with "
            "TrainingClient.save_weights_external() instead."
        )

    kwargs: dict = {}
    if base_url is not None:
        kwargs["base_url"] = base_url
    try:
        sc = tinker.ServiceClient(**kwargs)
        rc = sc.create_rest_client()
    except Exception as e:
        raise WeightsDownloadError(
            "Failed to connect to Tinker service. "
            "Ensure TINKER_API_KEY is set and the service is reachable."
        ) from e

    out = Path(output_dir)
    status: Literal["success", "errored"] = "errored"
    try:
        if parsed_path.checkpoint_type == "external":
            _download_external_weights(rc, tinker_path, out)
        else:
            _download_training_weights(sc, rc, tinker_path, out)
        status = "success"
    finally:
        try:
            sc.close(status).result()
        except Exception:
            logger.warning("Failed to close the Tinker client.", exc_info=True)
    return output_dir


def _download_training_weights(
    sc: tinker.ServiceClient, rc: RestClient, tinker_path: str, output_dir: Path
) -> None:
    """Download a training checkpoint through a temporary external weights export.

    ``save_weights_external()`` needs a training client, so the checkpoint is
    loaded into a new training run.
    """
    identity = _pyrage.x25519.Identity.generate()
    try:
        training_client = sc.create_training_client_from_state(
            tinker_path, user_metadata={"created_by": "tinker_cookbook.weights.download"}
        )
        export_path = (
            training_client.save_weights_external(
                "download",
                ttl_seconds=_TEMPORARY_EXPORT_TTL_SECONDS,
                age_encryption_recipients=[str(identity.to_public())],
            )
            .result()
            .path
        )
    except Exception as e:
        raise WeightsDownloadError(
            f"Failed to export {tinker_path!r} as external weights: {e}"
        ) from e

    try:
        _download_external_weights(
            rc,
            export_path,
            output_dir,
            renew_ttl_seconds=_TEMPORARY_EXPORT_TTL_SECONDS,
            decrypt_key=str(identity),
        )
    finally:
        try:
            rc.delete_checkpoint_from_tinker_path(export_path).result()
        except Exception:
            logger.warning(
                "Failed to delete the temporary export %s. It expires in %d seconds.",
                export_path,
                _TEMPORARY_EXPORT_TTL_SECONDS,
                exc_info=True,
            )


def _download_external_weights(
    rc: RestClient,
    tinker_path: str,
    output_dir: Path,
    *,
    renew_ttl_seconds: int | None = None,
    decrypt_key: str | None = None,
) -> None:
    """Download every file of an external weights checkpoint into ``output_dir``.

    Signs new URLs whenever the current ones are about to expire, so an export of
    many large files can take longer than one URL lifetime. If ``renew_ttl_seconds``
    is set, the checkpoint's TTL is reset to it each time URLs are signed. If
    ``decrypt_key`` is set, every file must be age-encrypted to it and is decrypted.
    """
    response = _get_external_weights_urls(rc, tinker_path, renew_ttl_seconds)
    if not response.urls:
        raise WeightsDownloadError(f"External weights checkpoint {tinker_path!r} has no files.")

    base = output_dir.resolve()
    destinations: dict[str, Path] = {}
    for name in response.urls:
        output_name = name
        if decrypt_key is not None:
            if not name.endswith(_AGE_SUFFIX):
                raise WeightsDownloadError(f"{name!r} in {tinker_path!r} is not encrypted.")
            output_name = name.removesuffix(_AGE_SUFFIX)
        dest = (output_dir / output_name).resolve()
        if not dest.is_relative_to(base):
            raise WeightsDownloadError(
                f"Unsafe file path {name!r} in external weights checkpoint (path traversal)."
            )
        destinations[name] = dest

    # A fresh private directory, so a failed transfer leaves output_dir unchanged.
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir, prefix=".download-"))
    try:
        for name, dest in sorted(destinations.items()):
            if datetime.datetime.now(datetime.UTC) >= response.expires - _URL_REFRESH_MARGIN:
                response = _get_external_weights_urls(rc, tinker_path, renew_ttl_seconds)
            url = response.urls.get(name)
            if url is None:
                raise WeightsDownloadError(
                    f"{name!r} is missing from the re-signed URLs for {tinker_path!r}."
                )
            staged = staging / dest.relative_to(base)
            staged.parent.mkdir(parents=True, exist_ok=True)
            fetched = staged if decrypt_key is None else staging / name
            try:
                urllib.request.urlretrieve(url, str(fetched))
            except urllib.error.URLError as e:
                raise WeightsDownloadError(
                    f"Failed to download {name!r} from its signed URL."
                ) from e
            if decrypt_key is not None:
                try:
                    identity = _pyrage.x25519.Identity.from_str(decrypt_key)
                    _pyrage.decrypt_file(str(fetched), str(staged), [identity])
                except _pyrage.DecryptError as e:
                    raise WeightsDownloadError(f"Failed to decrypt {name!r}: {e}") from e
                fetched.unlink()

        for dest in destinations.values():
            dest.parent.mkdir(parents=True, exist_ok=True)
            (staging / dest.relative_to(base)).replace(dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _get_external_weights_urls(
    rc: RestClient, tinker_path: str, renew_ttl_seconds: int | None
) -> ExternalWeightsUrlsResponse:
    try:
        response = rc.get_external_weights_urls(tinker_path).result()
    except Exception as e:
        raise WeightsDownloadError(f"Failed to get download URLs for {tinker_path!r}: {e}") from e
    if renew_ttl_seconds is not None:
        # After signing, so the checkpoint outlives these URLs.
        try:
            rc.set_checkpoint_ttl_from_tinker_path(tinker_path, renew_ttl_seconds).result()
        except Exception:
            logger.warning("Failed to renew the TTL of %s.", tinker_path, exc_info=True)
    return response
