"""Tests for the download function."""

import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tinker_cookbook.exceptions import WeightsDownloadError
from tinker_cookbook.weights._download import _safe_extract_tar, download


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


class TestDownload:
    """Tests for the download function with mocked Tinker SDK."""

    def test_downloads_and_extracts(self):
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

            mock_response = MagicMock()
            mock_response.url = f"file://{archive_path}"

            mock_future = MagicMock()
            mock_future.result.return_value = mock_response

            mock_rest_client = MagicMock()
            mock_rest_client.get_checkpoint_archive_url_from_tinker_path.return_value = mock_future

            mock_service_client = MagicMock()
            mock_service_client.create_rest_client.return_value = mock_rest_client

            def fake_urlretrieve(url: str, dest: str) -> None:
                import shutil

                shutil.copy2(str(archive_path), dest)

            with (
                patch(
                    "tinker_cookbook.weights._download.tinker.ServiceClient",
                    return_value=mock_service_client,
                ),
                patch(
                    "tinker_cookbook.weights._download.urllib.request.urlretrieve",
                    fake_urlretrieve,
                ),
            ):
                result = download(
                    tinker_path="tinker://fake-run/sampler_weights/final",
                    output_dir=str(output_dir),
                )

            assert result == str(output_dir)
            assert (output_dir / "adapter_model.safetensors").exists()
            assert (output_dir / "adapter_config.json").exists()
