"""Tests for the download function."""

import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tinker_cookbook.weights._download import _safe_extract_tar, download


class TestSafeExtractTar:
    """Security validation for tar extraction."""

    def test_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, "bad.tar")
            extract_dir = os.path.join(tmpdir, "extract")
            os.makedirs(extract_dir)

            # Create a tar with a symlink
            target = os.path.join(tmpdir, "target.txt")
            with open(target, "w") as f:
                f.write("target")
            link = os.path.join(tmpdir, "link")
            os.symlink(target, link)

            with tarfile.open(archive_path, "w") as tar:
                tar.add(link, arcname="link")

            with pytest.raises(ValueError, match="symlink"):
                _safe_extract_tar(archive_path, Path(extract_dir))

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, "bad.tar")
            extract_dir = os.path.join(tmpdir, "extract")
            os.makedirs(extract_dir)

            # Create a tar with a path traversal entry
            normal_file = os.path.join(tmpdir, "normal.txt")
            with open(normal_file, "w") as f:
                f.write("content")

            with tarfile.open(archive_path, "w") as tar:
                tar.add(normal_file, arcname="../../../etc/passwd")

            with pytest.raises(ValueError, match="path traversal"):
                _safe_extract_tar(archive_path, Path(extract_dir))

    def test_extracts_safe_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, "good.tar")
            extract_dir = os.path.join(tmpdir, "extract")
            os.makedirs(extract_dir)

            # Create a normal tar
            content_file = os.path.join(tmpdir, "data.txt")
            with open(content_file, "w") as f:
                f.write("hello")

            with tarfile.open(archive_path, "w") as tar:
                tar.add(content_file, arcname="data.txt")

            _safe_extract_tar(archive_path, Path(extract_dir))
            assert os.path.exists(os.path.join(extract_dir, "data.txt"))


class TestDownload:
    """Tests for the download function with mocked Tinker SDK."""

    def test_downloads_and_extracts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake archive
            archive_path = os.path.join(tmpdir, "archive.tar")
            content_dir = os.path.join(tmpdir, "content")
            os.makedirs(content_dir)
            with open(os.path.join(content_dir, "adapter_model.safetensors"), "w") as f:
                f.write("fake")
            with open(os.path.join(content_dir, "adapter_config.json"), "w") as f:
                f.write("{}")

            with tarfile.open(archive_path, "w") as tar:
                tar.add(
                    os.path.join(content_dir, "adapter_model.safetensors"),
                    arcname="adapter_model.safetensors",
                )
                tar.add(
                    os.path.join(content_dir, "adapter_config.json"),
                    arcname="adapter_config.json",
                )

            output_dir = os.path.join(tmpdir, "output")

            # Mock the Tinker SDK and urllib
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

                # Copy the archive to the destination (simulating download)
                shutil.copy2(archive_path, dest)

            with (
                patch("tinker.ServiceClient", return_value=mock_service_client),
                patch(
                    "tinker_cookbook.weights._download.urllib.request.urlretrieve", fake_urlretrieve
                ),
            ):
                result = download(
                    tinker_path="tinker://fake-run/sampler_weights/final",
                    output_dir=output_dir,
                )

            assert result == output_dir
            assert os.path.exists(os.path.join(output_dir, "adapter_model.safetensors"))
            assert os.path.exists(os.path.join(output_dir, "adapter_config.json"))
