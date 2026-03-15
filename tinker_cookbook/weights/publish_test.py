"""Tests for publish_to_hf_hub."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tinker_cookbook.weights import publish_to_hf_hub


class TestPublishToHfHub:
    def test_raises_on_nonexistent_path(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            publish_to_hf_hub(
                model_path="/nonexistent/path",
                repo_id="user/model",
            )

    def test_calls_hf_api_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_api = MagicMock()

            with patch("huggingface_hub.HfApi", return_value=mock_api):
                url = publish_to_hf_hub(
                    model_path=tmpdir,
                    repo_id="user/my-model",
                    private=True,
                )

            mock_api.create_repo.assert_called_once_with(
                repo_id="user/my-model",
                repo_type="model",
                private=True,
                exist_ok=True,
            )
            mock_api.upload_folder.assert_called_once_with(
                folder_path=tmpdir,
                repo_id="user/my-model",
                repo_type="model",
            )
            assert url == "https://huggingface.co/user/my-model"

    def test_public_repo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_api = MagicMock()

            with patch("huggingface_hub.HfApi", return_value=mock_api):
                publish_to_hf_hub(
                    model_path=tmpdir,
                    repo_id="org/public-model",
                    private=False,
                )

            mock_api.create_repo.assert_called_once_with(
                repo_id="org/public-model",
                repo_type="model",
                private=False,
                exist_ok=True,
            )
