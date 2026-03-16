"""Integration test for weights.publish_to_hf_hub().

Requires HF authentication (HF_TOKEN env var or `hf auth login`).
Skipped otherwise.

Creates a temporary private repo, uploads a tiny dummy model, verifies
the upload, and cleans up the repo regardless of test outcome.
"""

import json
import tempfile
import uuid
from pathlib import Path

import pytest

from tinker_cookbook.weights import publish_to_hf_hub


def _hf_username() -> str:
    """Get the authenticated HF username, or skip the test."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.whoami()
        return info["name"]
    except Exception:
        pytest.skip("HF authentication required (set HF_TOKEN or run `hf auth login`)")
        return ""  # unreachable, keeps type checker happy


def _create_dummy_model_dir(path: Path) -> None:
    """Create a minimal directory that looks like an HF model."""
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "test"}))
    (path / "README.md").write_text("Test model for tinker_cookbook integration test")


@pytest.mark.integration
class TestPublishToHfHubIntegration:
    def test_upload_and_verify(self):
        username = _hf_username()
        repo_id = f"{username}/tinker-cookbook-test-{uuid.uuid4().hex[:8]}"

        from huggingface_hub import HfApi

        api = HfApi()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "model"
                _create_dummy_model_dir(model_path)

                url = publish_to_hf_hub(
                    model_path=str(model_path),
                    repo_id=repo_id,
                    private=True,
                )

                assert url == f"https://huggingface.co/{repo_id}"

                # Verify the repo exists and has our files
                files = api.list_repo_files(repo_id=repo_id, repo_type="model")
                assert "config.json" in files
                assert "README.md" in files
        finally:
            # Always clean up, even if test fails
            try:
                api.delete_repo(repo_id=repo_id, repo_type="model")
            except Exception:
                pass
