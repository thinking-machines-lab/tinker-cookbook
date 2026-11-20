"""Integration test for weights.publish_to_hf_hub().

Publishes to a fixed private repo, which the HF token needs write access to.
Skipped when no HF token is configured. CI runs these tests one at a time, since
they share the repo.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from huggingface_hub import CommitOperationDelete, HfApi, get_token, hf_hub_download

from tinker_cookbook.weights import ModelCardConfig, publish_to_hf_hub

_TEST_REPO_ID = "thinkingmachines/tinker-cookbook-integration-tests"


def _create_dummy_model_dir(path: Path) -> None:
    """Create a minimal directory that looks like an HF model."""
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "test"}))
    (path / "README.md").write_text("Test model for tinker_cookbook integration test")


def _create_dummy_adapter_dir(path: Path) -> None:
    """Create a minimal directory that looks like a PEFT adapter."""
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 32, "base_model_name_or_path": "Qwen/Qwen3-8B"})
    )
    (path / "adapter_model.safetensors").write_bytes(b"dummy")


def _download_readme(repo_id: str) -> str:
    """Download README.md from a HF repo and return its content."""
    path = hf_hub_download(repo_id=repo_id, filename="README.md")
    return Path(path).read_text()


def _clean_test_repo() -> tuple[str, HfApi]:
    """Return the shared test repo, emptied of previous runs' files, or skip without a token."""
    if get_token() is None:
        pytest.skip("HF authentication required (set HF_TOKEN or run `hf auth login`)")
    api = HfApi()
    stale = [f for f in api.list_repo_files(_TEST_REPO_ID) if f != ".gitattributes"]
    if stale:
        api.create_commit(
            repo_id=_TEST_REPO_ID,
            operations=[CommitOperationDelete(path_in_repo=f) for f in stale],
            commit_message="Clear previous test run",
        )
    return _TEST_REPO_ID, api


@pytest.mark.integration
class TestPublishToHfHubIntegration:
    def test_upload_and_verify(self):
        repo_id, api = _clean_test_repo()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            _create_dummy_model_dir(model_path)

            url = publish_to_hf_hub(
                model_path=str(model_path),
                repo_id=repo_id,
                private=True,
            )

            assert url == f"https://huggingface.co/{repo_id}"

            files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            assert "config.json" in files
            assert "README.md" in files

    def test_model_card_merged_model(self):
        """Publish a merged model with model_card and verify the generated README."""
        repo_id, api = _clean_test_repo()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model_path.mkdir(parents=True)
            (model_path / "config.json").write_text(json.dumps({"model_type": "test"}))

            publish_to_hf_hub(
                model_path=str(model_path),
                repo_id=repo_id,
                private=True,
                model_card=ModelCardConfig(
                    base_model="Qwen/Qwen3-8B",
                    datasets=["my-org/my-dataset"],
                    tags=["sft"],
                    license="apache-2.0",
                ),
            )

            files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            assert "README.md" in files

            content = _download_readme(repo_id)
            assert "base_model: Qwen/Qwen3-8B" in content
            assert "library_name: transformers" in content
            assert "- tinker" in content
            assert "- sft" in content
            assert "license: apache-2.0" in content
            assert "- my-org/my-dataset" in content
            assert "Merged model" in content

    def test_model_card_adapter(self):
        """Publish an adapter with model_card and verify auto-detected peft card."""
        repo_id, api = _clean_test_repo()
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter"
            _create_dummy_adapter_dir(adapter_path)

            publish_to_hf_hub(
                model_path=str(adapter_path),
                repo_id=repo_id,
                private=True,
                model_card=ModelCardConfig(base_model="Qwen/Qwen3-8B"),
            )

            files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            assert "README.md" in files
            assert "adapter_config.json" in files

            content = _download_readme(repo_id)
            assert "base_model: Qwen/Qwen3-8B" in content
            assert "library_name: peft" in content
            assert "- peft" in content
            assert "- lora" in content
            assert "LoRA adapter (PEFT)" in content
            assert "PeftModel" in content
