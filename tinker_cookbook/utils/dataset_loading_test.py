"""Tests for tinker_cookbook.utils.dataset_loading."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from tinker_cookbook.utils.dataset_loading import (
    _infer_format,
    is_cloud_uri,
    load_dataset,
    load_from_disk,
)

# ---------------------------------------------------------------------------
# is_cloud_uri
# ---------------------------------------------------------------------------


class TestIsCloudUri:
    @pytest.mark.parametrize(
        "uri",
        [
            "s3://bucket/path/data.parquet",
            "gs://bucket/path/",
            "gcs://bucket/data",
            "az://container/path",
            "abfs://container/path",
            "abfss://container/path",
        ],
    )
    def test_positive(self, uri: str) -> None:
        assert is_cloud_uri(uri) is True

    @pytest.mark.parametrize(
        "uri",
        [
            "openai/gsm8k",
            "allenai/tulu-3-sft-mixture",
            "/local/path/to/data",
            "./relative/path",
            "file:///local/path",
            "https://huggingface.co/datasets/openai/gsm8k",
        ],
    )
    def test_negative(self, uri: str) -> None:
        assert is_cloud_uri(uri) is False


# ---------------------------------------------------------------------------
# _infer_format
# ---------------------------------------------------------------------------


class TestInferFormat:
    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("s3://bucket/data.parquet", "parquet"),
            ("gs://bucket/train.jsonl", "json"),
            ("az://container/data.json", "json"),
            ("s3://bucket/data.csv", "csv"),
            ("s3://bucket/data.tsv", "csv"),
            ("s3://bucket/data.arrow", "arrow"),
            ("s3://bucket/data.txt", "text"),
        ],
    )
    def test_known_extensions(self, uri: str, expected: str) -> None:
        assert _infer_format(uri) == expected

    def test_glob_pattern(self) -> None:
        assert _infer_format("s3://bucket/data/*.parquet") == "parquet"
        assert _infer_format("gs://bucket/train/*.jsonl") == "json"

    def test_no_extension_returns_none(self) -> None:
        assert _infer_format("s3://bucket/my_dataset") is None
        assert _infer_format("s3://bucket/my_dataset/") is None

    def test_unknown_extension_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot infer dataset format"):
            _infer_format("s3://bucket/data.xyz")


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


class TestLoadDataset:
    @patch("tinker_cookbook.utils.dataset_loading._hf_datasets")
    def test_hub_path_delegates_to_hf(self, mock_hf: MagicMock) -> None:
        mock_hf.load_dataset.return_value = "mock_dataset"
        result = load_dataset("openai/gsm8k", name="main", split="test")
        mock_hf.load_dataset.assert_called_once_with("openai/gsm8k", name="main", split="test")
        assert result == "mock_dataset"

    @patch("tinker_cookbook.utils.dataset_loading._hf_datasets")
    def test_local_path_delegates_to_hf(self, mock_hf: MagicMock) -> None:
        mock_hf.load_dataset.return_value = "mock_dataset"
        result = load_dataset("/local/path/data.parquet", split="train")
        mock_hf.load_dataset.assert_called_once_with("/local/path/data.parquet", split="train")
        assert result == "mock_dataset"

    @patch("tinker_cookbook.utils.dataset_loading._hf_datasets")
    @patch("tinker_cookbook.utils.dataset_loading._check_cloud_deps")
    def test_cloud_parquet_uses_data_files(self, mock_check: MagicMock, mock_hf: MagicMock) -> None:
        mock_hf.load_dataset.return_value = "cloud_dataset"
        result = load_dataset("s3://bucket/data.parquet", split="train")
        mock_hf.load_dataset.assert_called_once_with(
            "parquet", data_files="s3://bucket/data.parquet", split="train"
        )
        assert result == "cloud_dataset"

    @patch("tinker_cookbook.utils.dataset_loading._hf_datasets")
    @patch("tinker_cookbook.utils.dataset_loading._check_cloud_deps")
    def test_cloud_jsonl_uses_data_files(self, mock_check: MagicMock, mock_hf: MagicMock) -> None:
        mock_hf.load_dataset.return_value = "cloud_dataset"
        result = load_dataset("gs://bucket/train.jsonl", split="train")
        mock_hf.load_dataset.assert_called_once_with(
            "json", data_files="gs://bucket/train.jsonl", split="train"
        )
        assert result == "cloud_dataset"

    @patch("tinker_cookbook.utils.dataset_loading.load_from_disk")
    @patch("tinker_cookbook.utils.dataset_loading._check_cloud_deps")
    def test_cloud_directory_uses_load_from_disk(
        self, mock_check: MagicMock, mock_load: MagicMock
    ) -> None:
        mock_load.return_value = "arrow_dataset"
        result = load_dataset("s3://bucket/my_dataset")
        mock_load.assert_called_once_with("s3://bucket/my_dataset", storage_options=None)
        assert result == "arrow_dataset"

    @patch("tinker_cookbook.utils.dataset_loading._hf_datasets")
    @patch("tinker_cookbook.utils.dataset_loading._check_cloud_deps")
    def test_storage_options_forwarded(self, mock_check: MagicMock, mock_hf: MagicMock) -> None:
        opts = {"anon": True}
        load_dataset("s3://bucket/data.parquet", split="train", storage_options=opts)
        mock_hf.load_dataset.assert_called_once_with(
            "parquet", data_files="s3://bucket/data.parquet", split="train", storage_options=opts
        )

    @patch("tinker_cookbook.utils.dataset_loading._hf_datasets")
    def test_extra_kwargs_forwarded(self, mock_hf: MagicMock) -> None:
        load_dataset("openai/gsm8k", split="test", streaming=True)
        mock_hf.load_dataset.assert_called_once_with("openai/gsm8k", split="test", streaming=True)


# ---------------------------------------------------------------------------
# load_from_disk
# ---------------------------------------------------------------------------


class TestLoadFromDisk:
    @patch("tinker_cookbook.utils.dataset_loading._hf_datasets")
    def test_local_path_delegates_to_hf(self, mock_hf: MagicMock) -> None:
        mock_hf.load_from_disk.return_value = "local_dataset"
        result = load_from_disk("/local/path/dataset")
        mock_hf.load_from_disk.assert_called_once_with("/local/path/dataset")
        assert result == "local_dataset"

    @patch("tinker_cookbook.utils.dataset_loading._cached_cloud_download")
    @patch("tinker_cookbook.utils.dataset_loading._hf_datasets")
    @patch("tinker_cookbook.utils.dataset_loading._check_cloud_deps")
    def test_cloud_path_caches_locally(
        self, mock_check: MagicMock, mock_hf: MagicMock, mock_download: MagicMock
    ) -> None:
        mock_download.return_value = "/tmp/cached/abc123"
        mock_hf.load_from_disk.return_value = "cached_dataset"
        result = load_from_disk("s3://bucket/my_dataset")
        mock_download.assert_called_once_with(
            "s3://bucket/my_dataset", cache_dir=None, storage_options=None
        )
        mock_hf.load_from_disk.assert_called_once_with("/tmp/cached/abc123")
        assert result == "cached_dataset"

    @patch("tinker_cookbook.utils.dataset_loading._cached_cloud_download")
    @patch("tinker_cookbook.utils.dataset_loading._hf_datasets")
    @patch("tinker_cookbook.utils.dataset_loading._check_cloud_deps")
    def test_custom_cache_dir(
        self, mock_check: MagicMock, mock_hf: MagicMock, mock_download: MagicMock
    ) -> None:
        mock_download.return_value = "/custom/cache/abc123"
        load_from_disk("gs://bucket/dataset", cache_dir="/custom/cache")
        mock_download.assert_called_once_with(
            "gs://bucket/dataset", cache_dir="/custom/cache", storage_options=None
        )


# ---------------------------------------------------------------------------
# _check_cloud_deps
# ---------------------------------------------------------------------------


class TestCheckCloudDeps:
    def test_fsspec_available(self) -> None:
        # Should not raise — fsspec is installed in this environment.
        from tinker_cookbook.utils.dataset_loading import _check_cloud_deps

        _check_cloud_deps("s3://bucket/data")

    @patch.dict("sys.modules", {"fsspec": None})
    def test_fsspec_missing_raises(self) -> None:
        from tinker_cookbook.utils.dataset_loading import _check_cloud_deps

        with pytest.raises(ImportError, match="fsspec is required"):
            _check_cloud_deps("s3://bucket/data")


# ---------------------------------------------------------------------------
# _cached_cloud_download
# ---------------------------------------------------------------------------


class TestCachedCloudDownload:
    def test_downloads_and_creates_marker(self) -> None:
        """Integration-style test using a mock fsspec filesystem."""
        import tempfile

        from tinker_cookbook.utils.dataset_loading import _cached_cloud_download

        cache_dir = tempfile.mkdtemp()
        mock_fs = MagicMock()

        with patch("fsspec.core.url_to_fs", return_value=(mock_fs, "bucket/dataset")):
            local_path = _cached_cloud_download("s3://bucket/dataset", cache_dir=cache_dir)

        # Marker file should exist.
        marker = os.path.join(local_path, ".cloud_source")
        assert os.path.isfile(marker)
        with open(marker) as f:
            assert f.read().strip() == "s3://bucket/dataset"

        # fs.get should have been called once with recursive=True.
        mock_fs.get.assert_called_once_with("bucket/dataset", local_path, recursive=True)

    def test_reuses_cache(self) -> None:
        """Second call for same URI should skip download."""
        import tempfile

        from tinker_cookbook.utils.dataset_loading import _cached_cloud_download

        cache_dir = tempfile.mkdtemp()
        mock_fs = MagicMock()

        with patch("fsspec.core.url_to_fs", return_value=(mock_fs, "bucket/dataset")):
            path1 = _cached_cloud_download("s3://bucket/dataset", cache_dir=cache_dir)
            mock_fs.get.reset_mock()
            path2 = _cached_cloud_download("s3://bucket/dataset", cache_dir=cache_dir)

        assert path1 == path2
        mock_fs.get.assert_not_called()  # Second call should not download.

    def test_cleans_up_on_failure(self) -> None:
        """Partial download should be cleaned up on failure."""
        import tempfile

        from tinker_cookbook.utils.dataset_loading import _cached_cloud_download

        cache_dir = tempfile.mkdtemp()
        mock_fs = MagicMock()
        mock_fs.get.side_effect = OSError("download failed")

        with (
            patch("fsspec.core.url_to_fs", return_value=(mock_fs, "bucket/dataset")),
            pytest.raises(OSError, match="download failed"),
        ):
            _cached_cloud_download("s3://bucket/dataset", cache_dir=cache_dir)

        # Cache directory should have been cleaned up.
        import hashlib as _hashlib

        cache_key = _hashlib.sha256(b"s3://bucket/dataset").hexdigest()[:16]
        assert not os.path.exists(os.path.join(cache_dir, cache_key))
