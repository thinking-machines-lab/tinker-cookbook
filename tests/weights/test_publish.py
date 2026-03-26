"""Integration test for weights.publish_to_hf_hub().

Requires HF authentication (HF_TOKEN env var or `hf auth login`).
Skipped otherwise.

Creates a temporary private repo, uploads a tiny dummy model, verifies
the upload, and cleans up the repo regardless of test outcome.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from huggingface_hub import HfApi, hf_hub_download

from tinker_cookbook.weights import ModelCardConfig, publish_to_hf_hub


def _hf_username() -> str:
    """Get the authenticated HF username, or skip the test."""
    try:
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


@contextlib.contextmanager
def _managed_hf_repo() -> Generator[tuple[str, HfApi]]:
    """Create a temporary HF repo ID with automatic cleanup."""
    username = _hf_username()
    repo_id = f"{username}/tinker-cookbook-test-{uuid.uuid4().hex[:8]}"
    api = HfApi()
    try:
        yield repo_id, api
    finally:
        with contextlib.suppress(Exception):
            api.delete_repo(repo_id=repo_id, repo_type="model")


@pytest.mark.integration
class TestPublishToHfHubIntegration:
    def test_upload_and_verify(self):
        with _managed_hf_repo() as (repo_id, api):
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
        with _managed_hf_repo() as (repo_id, api):
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
        with _managed_hf_repo() as (repo_id, api):
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
