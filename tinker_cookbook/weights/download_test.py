"""Tests for the download function."""

import datetime
import shutil
import tarfile
import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from tinker.types import ExternalWeightsUrlsResponse, SaveWeightsExternalResponse

from tinker_cookbook.exceptions import WeightsDownloadError
from tinker_cookbook.weights._download import _pyrage, _safe_extract_tar, download


class TestSafeExtractTar:
    """Security validation for tar extraction."""

    def test_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "bad.tar"
            extract_dir = root / "extract"
            extract_dir.mkdir()

            target = root / "target.txt"
            target.write_text("target")
            link = root / "link"
            link.symlink_to(target)

            with tarfile.open(archive_path, "w") as tar:
                tar.add(link, arcname="link")

            with pytest.raises(WeightsDownloadError, match="symlink"):
                _safe_extract_tar(archive_path, extract_dir)

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "bad.tar"
            extract_dir = root / "extract"
            extract_dir.mkdir()

            normal_file = root / "normal.txt"
            normal_file.write_text("content")

            with tarfile.open(archive_path, "w") as tar:
                tar.add(normal_file, arcname="../../../etc/passwd")

            with pytest.raises(WeightsDownloadError, match="path traversal"):
                _safe_extract_tar(archive_path, extract_dir)

    def test_extracts_safe_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "good.tar"
            extract_dir = root / "extract"
            extract_dir.mkdir()

            content_file = root / "data.txt"
            content_file.write_text("hello")

            with tarfile.open(archive_path, "w") as tar:
                tar.add(content_file, arcname="data.txt")

            _safe_extract_tar(archive_path, extract_dir)
            assert (extract_dir / "data.txt").exists()


def _future(*, result: object = None, error: Exception | None = None) -> MagicMock:
    future = MagicMock()
    if error is not None:
        future.result.side_effect = error
    else:
        future.result.return_value = result
    return future


def _patch_sdk(rest_client: MagicMock, service_client: MagicMock | None = None):
    service_client = service_client or MagicMock()
    service_client.create_rest_client.return_value = rest_client
    return patch(
        "tinker_cookbook.weights._download.tinker.ServiceClient",
        return_value=service_client,
    )


def _urls_response(
    urls: dict[str, str], expires_in: datetime.timedelta = datetime.timedelta(hours=1)
) -> ExternalWeightsUrlsResponse:
    return ExternalWeightsUrlsResponse(
        urls=urls, expires=datetime.datetime.now(datetime.UTC) + expires_in
    )


def _copy_file_url(url: str, dest: str) -> None:
    shutil.copy2(url.removeprefix("file://"), dest)


_EXPORT_PATH = "tinker://export-run/external_weights/download"


def _export_urls(export_dir: Path, files: dict[str, str]) -> dict[str, str]:
    return {f"{name}.age": f"file://{export_dir / name}.age" for name in files}


def _training_round_trip_clients(
    export_dir: Path, files: dict[str, str]
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return a service client, rest client, and training client for a ``weights/`` download.

    Like the server, the export encrypts ``files`` into ``export_dir`` to the given recipients.
    """

    def save_weights_external(
        name: str, *, ttl_seconds: int, age_encryption_recipients: list[str]
    ) -> MagicMock:
        recipients = [_pyrage.x25519.Recipient.from_str(r) for r in age_encryption_recipients]
        export_dir.mkdir(parents=True, exist_ok=True)
        for file_name, content in files.items():
            encrypted = _pyrage.encrypt(content.encode(), recipients)
            (export_dir / f"{file_name}.age").write_bytes(encrypted)
        return _future(result=SaveWeightsExternalResponse(path=_EXPORT_PATH))

    training_client = MagicMock()
    training_client.save_weights_external.side_effect = save_weights_external
    service_client = MagicMock()
    service_client.create_training_client_from_state.return_value = training_client
    rest_client = MagicMock()
    rest_client.get_external_weights_urls.return_value = _future(
        result=_urls_response(_export_urls(export_dir, files))
    )
    rest_client.delete_checkpoint_from_tinker_path.return_value = _future()
    return service_client, rest_client, training_client


class TestDownload:
    """Tests for the download function with mocked Tinker SDK."""

    def test_sampler_path_downloads_and_extracts_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive_path = root / "archive.tar"
            content_dir = root / "content"
            content_dir.mkdir()
            (content_dir / "adapter_model.safetensors").write_text("fake")
            (content_dir / "adapter_config.json").write_text("{}")

            with tarfile.open(archive_path, "w") as tar:
                tar.add(
                    content_dir / "adapter_model.safetensors", arcname="adapter_model.safetensors"
                )
                tar.add(content_dir / "adapter_config.json", arcname="adapter_config.json")

            output_dir = root / "output"

            archive_response = MagicMock()
            archive_response.url = f"file://{archive_path}"
            rest_client = MagicMock()
            rest_client.get_checkpoint_archive_url_from_tinker_path.return_value = _future(
                result=archive_response
            )

            service_client = MagicMock()
            with (
                _patch_sdk(rest_client, service_client),
                patch(
                    "tinker_cookbook.weights._download.urllib.request.urlretrieve",
                    _copy_file_url,
                ),
            ):
                result = download(
                    tinker_path="tinker://fake-run/sampler_weights/final",
                    output_dir=str(output_dir),
                )

            assert result == str(output_dir)
            assert (output_dir / "adapter_model.safetensors").exists()
            assert (output_dir / "adapter_config.json").exists()
            rest_client.get_external_weights_urls.assert_not_called()
            service_client.close.assert_called_once_with("success")

    def test_external_path_downloads_every_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            source_dir.mkdir()
            files = {
                "adapter_model.safetensors": "weights",
                "adapter_config.json": "{}",
                "nested/extra.json": "{}",
            }
            for name, content in files.items():
                (source_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (source_dir / name).write_text(content)

            tinker_path = "tinker://fake-run/external_weights/final"
            urls_response = _urls_response({name: f"file://{source_dir / name}" for name in files})
            rest_client = MagicMock()
            rest_client.get_external_weights_urls.return_value = _future(result=urls_response)

            output_dir = root / "output"
            with (
                _patch_sdk(rest_client),
                patch(
                    "tinker_cookbook.weights._download.urllib.request.urlretrieve",
                    _copy_file_url,
                ),
            ):
                result = download(tinker_path=tinker_path, output_dir=str(output_dir))

            assert result == str(output_dir)
            for name, content in files.items():
                assert (output_dir / name).read_text() == content
            rest_client.get_external_weights_urls.assert_called_once_with(tinker_path)
            rest_client.get_checkpoint_archive_url_from_tinker_path.assert_not_called()
            rest_client.set_checkpoint_ttl_from_tinker_path.assert_not_called()

    def test_external_download_failure_keeps_the_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            (output_dir / "adapter_config.json").write_text("previous config")
            (output_dir / "adapter_model.safetensors").write_text("previous checkpoint")
            urls_response = _urls_response(
                {
                    "adapter_config.json": "https://example.invalid/config",
                    "adapter_model.safetensors": "https://example.invalid/weights",
                }
            )
            rest_client = MagicMock()
            rest_client.get_external_weights_urls.return_value = _future(result=urls_response)

            def fail_midway(url: str, dest: str) -> None:
                Path(dest).write_text("new or truncated")
                if url.endswith("/weights"):
                    raise urllib.error.URLError("connection reset")

            with (
                _patch_sdk(rest_client),
                patch("tinker_cookbook.weights._download.urllib.request.urlretrieve", fail_midway),
                pytest.raises(WeightsDownloadError, match="Failed to download"),
            ):
                download(
                    tinker_path="tinker://fake-run/external_weights/final",
                    output_dir=str(output_dir),
                )

            assert (output_dir / "adapter_config.json").read_text() == "previous config"
            assert (output_dir / "adapter_model.safetensors").read_text() == "previous checkpoint"
            assert sorted(p.name for p in output_dir.iterdir()) == [
                "adapter_config.json",
                "adapter_model.safetensors",
            ]

    def test_external_path_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            urls_response = _urls_response({"../escape.json": "file:///unused"})
            rest_client = MagicMock()
            rest_client.get_external_weights_urls.return_value = _future(result=urls_response)
            urlretrieve = MagicMock()

            with (
                _patch_sdk(rest_client),
                patch("tinker_cookbook.weights._download.urllib.request.urlretrieve", urlretrieve),
                pytest.raises(WeightsDownloadError, match="path traversal"),
            ):
                download(
                    tinker_path="tinker://fake-run/external_weights/final",
                    output_dir=str(output_dir),
                )
            urlretrieve.assert_not_called()

    def test_training_path_downloads_a_temporary_export_and_deletes_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            files = {
                "adapter_config.json": "{}",
                "model-1.safetensors": "weights 1",
                "model-2.safetensors": "weights 2",
            }
            service_client, rest_client, training_client = _training_round_trip_clients(
                root / "export", files
            )
            urls = _export_urls(root / "export", files)
            rest_client.get_external_weights_urls.side_effect = [
                _future(result=_urls_response(urls, expires_in=datetime.timedelta(minutes=1))),
                _future(result=_urls_response(urls)),
            ]

            output_dir = root / "output"
            with (
                _patch_sdk(rest_client, service_client),
                patch(
                    "tinker_cookbook.weights._download.urllib.request.urlretrieve",
                    _copy_file_url,
                ),
            ):
                download(tinker_path="tinker://fake-run/weights/final", output_dir=str(output_dir))

            assert {p.name: p.read_text() for p in output_dir.iterdir()} == files
            service_client.create_training_client_from_state.assert_called_once_with(
                "tinker://fake-run/weights/final",
                user_metadata={"created_by": "tinker_cookbook.weights.download"},
            )
            training_client.save_weights_external.assert_called_once()
            save_args = training_client.save_weights_external.call_args
            assert save_args.args == ("download",)
            assert save_args.kwargs["ttl_seconds"] == 3600
            assert len(save_args.kwargs["age_encryption_recipients"]) == 1
            assert rest_client.get_external_weights_urls.call_args_list == [call(_EXPORT_PATH)] * 2
            assert (
                rest_client.set_checkpoint_ttl_from_tinker_path.call_args_list
                == [call(_EXPORT_PATH, 3600)] * 2
            )
            rest_client.delete_checkpoint_from_tinker_path.assert_called_once_with(_EXPORT_PATH)
            service_client.close.assert_called_once_with("success")
            rest_client.get_checkpoint_archive_url_from_tinker_path.assert_not_called()

    def test_training_path_deletes_the_export_when_the_download_fails(self):
        def fail(url: str, dest: str) -> None:
            raise urllib.error.URLError("connection reset")

        with tempfile.TemporaryDirectory() as tmpdir:
            service_client, rest_client, _ = _training_round_trip_clients(
                Path(tmpdir) / "export", {"adapter_config.json": "{}"}
            )
            with (
                _patch_sdk(rest_client, service_client),
                patch("tinker_cookbook.weights._download.urllib.request.urlretrieve", fail),
                pytest.raises(WeightsDownloadError, match="Failed to download"),
            ):
                download(
                    tinker_path="tinker://fake-run/weights/final",
                    output_dir=str(Path(tmpdir) / "output"),
                )
        rest_client.delete_checkpoint_from_tinker_path.assert_called_once_with(_EXPORT_PATH)
        service_client.close.assert_called_once_with("errored")

    def test_training_path_rejects_files_it_cannot_decrypt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            service_client, rest_client, _ = _training_round_trip_clients(
                root / "export", {"adapter_config.json": "{}"}
            )
            other_key = _pyrage.x25519.Identity.generate().to_public()
            (root / "foreign.json.age").write_bytes(_pyrage.encrypt(b"{}", [other_key]))
            (root / "plain.json").write_text("{}")
            output_dir = root / "output"

            for name, error in [
                ("foreign.json.age", "Failed to decrypt"),
                ("plain.json", "is not encrypted"),
            ]:
                rest_client.get_external_weights_urls.return_value = _future(
                    result=_urls_response({name: f"file://{root / name}"})
                )
                with (
                    _patch_sdk(rest_client, service_client),
                    patch(
                        "tinker_cookbook.weights._download.urllib.request.urlretrieve",
                        _copy_file_url,
                    ),
                    pytest.raises(WeightsDownloadError, match=error),
                ):
                    download(
                        tinker_path="tinker://fake-run/weights/final", output_dir=str(output_dir)
                    )
                assert not output_dir.exists() or list(output_dir.iterdir()) == []

    def test_training_path_succeeds_when_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            service_client, rest_client, _ = _training_round_trip_clients(
                root / "export", {"adapter_config.json": "{}"}
            )
            rest_client.delete_checkpoint_from_tinker_path.return_value = _future(
                error=RuntimeError("delete failed")
            )
            service_client.close.return_value = _future(error=RuntimeError("close failed"))

            output_dir = root / "output"
            with (
                _patch_sdk(rest_client, service_client),
                patch(
                    "tinker_cookbook.weights._download.urllib.request.urlretrieve",
                    _copy_file_url,
                ),
            ):
                download(tinker_path="tinker://fake-run/weights/final", output_dir=str(output_dir))

            assert (output_dir / "adapter_config.json").read_text() == "{}"
