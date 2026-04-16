"""Tests for model card generation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from tinker_cookbook.weights._model_card import (
    ModelCardConfig,
    generate_model_card,
)
from tinker_cookbook.weights._publish import publish_to_hf_hub

# =============================================================================
# generate_model_card tests
# =============================================================================


class TestGenerateModelCard:
    def test_minimal_config(self) -> None:
        """Minimal config (just base_model) produces valid card with frontmatter."""
        card = generate_model_card(config=ModelCardConfig(base_model="Qwen/Qwen3-8B"))
        text = str(card)

        assert "base_model: Qwen/Qwen3-8B" in text
        assert "library_name: transformers" in text
        assert "pipeline_tag: text-generation" in text
        assert "- tinker" in text
        assert "- tinker-cookbook" in text
        assert "## Usage" in text
        assert "## Framework versions" in text

    def test_adapter_auto_detection(self) -> None:
        """adapter_config.json in model_path triggers adapter mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "adapter_config.json").write_text("{}")

            card = generate_model_card(
                config=ModelCardConfig(base_model="Qwen/Qwen3-8B"),
                model_path=tmpdir,
            )
            text = str(card)

            assert "library_name: peft" in text
            assert "- peft" in text
            assert "- lora" in text
            assert "LoRA adapter (PEFT)" in text
            assert "PeftModel" in text

    def test_merged_model_detection(self) -> None:
        """model_path without adapter_config.json produces merged model card."""
        with tempfile.TemporaryDirectory() as tmpdir:
            card = generate_model_card(
                config=ModelCardConfig(base_model="Qwen/Qwen3-8B"),
                model_path=tmpdir,
            )
            text = str(card)

            assert "library_name: transformers" in text
            assert "peft" not in text.split("---")[1]  # not in YAML frontmatter
            assert "Merged model" in text
            assert "AutoModelForCausalLM.from_pretrained" in text

    def test_no_model_path_defaults_to_merged(self) -> None:
        """No model_path defaults to merged model mode."""
        card = generate_model_card(config=ModelCardConfig(base_model="Qwen/Qwen3-8B"))
        text = str(card)

        assert "library_name: transformers" in text
        assert "Merged model" in text

    def test_custom_tags_appended(self) -> None:
        """User tags are appended to defaults, not replacing them."""
        card = generate_model_card(
            config=ModelCardConfig(base_model="Qwen/Qwen3-8B", tags=["sft", "my-project"]),
        )
        text = str(card)

        assert "- tinker\n" in text
        assert "- tinker-cookbook\n" in text
        assert "- sft\n" in text
        assert "- my-project\n" in text

    def test_datasets_in_frontmatter(self) -> None:
        """datasets field appears in YAML frontmatter."""
        card = generate_model_card(
            config=ModelCardConfig(base_model="Qwen/Qwen3-8B", datasets=["my-org/my-dataset"]),
        )
        text = str(card)

        assert "datasets:" in text
        assert "- my-org/my-dataset" in text

    def test_license_in_frontmatter(self) -> None:
        """license field appears in YAML frontmatter."""
        card = generate_model_card(
            config=ModelCardConfig(base_model="Qwen/Qwen3-8B", license="apache-2.0"),
        )
        text = str(card)

        assert "license: apache-2.0" in text

    def test_language_in_frontmatter(self) -> None:
        """language field appears in YAML frontmatter."""
        card = generate_model_card(
            config=ModelCardConfig(base_model="Qwen/Qwen3-8B", language=["en", "zh"]),
        )
        text = str(card)

        assert "language:" in text
        assert "- en" in text
        assert "- zh" in text

    def test_repo_id_in_title_and_snippet(self) -> None:
        """repo_id is used in the card title and usage snippet."""
        card = generate_model_card(
            config=ModelCardConfig(base_model="Qwen/Qwen3-8B"),
            repo_id="user/my-finetuned-model",
        )
        text = str(card)

        assert "# user/my-finetuned-model" in text
        assert '"user/my-finetuned-model"' in text

    def test_adapter_usage_snippet(self) -> None:
        """Adapter card shows PeftModel loading code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "adapter_config.json").write_text("{}")
            card = generate_model_card(
                config=ModelCardConfig(base_model="Qwen/Qwen3-8B"),
                repo_id="user/my-adapter",
                model_path=tmpdir,
            )
            text = str(card)

            assert "from peft import PeftModel" in text
            assert '"Qwen/Qwen3-8B"' in text
            assert '"user/my-adapter"' in text


# =============================================================================
# publish_to_hf_hub integration with model_card
# =============================================================================


class TestPublishWithModelCard:
    def test_model_card_writes_readme(self) -> None:
        """model_card param causes README.md to be written before upload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_api = MagicMock()

            with patch("tinker_cookbook.weights._publish.HfApi", return_value=mock_api):
                publish_to_hf_hub(
                    model_path=tmpdir,
                    repo_id="user/my-model",
                    model_card=ModelCardConfig(base_model="Qwen/Qwen3-8B"),
                )

            readme = Path(tmpdir) / "README.md"
            assert readme.exists()
            content = readme.read_text()
            assert "base_model: Qwen/Qwen3-8B" in content
            assert "- tinker" in content

            mock_api.upload_folder.assert_called_once()

    def test_existing_readme_not_overwritten(self) -> None:
        """Existing README.md is preserved when model_card is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            readme = Path(tmpdir) / "README.md"
            readme.write_text("My custom README")

            mock_api = MagicMock()

            with patch("tinker_cookbook.weights._publish.HfApi", return_value=mock_api):
                publish_to_hf_hub(
                    model_path=tmpdir,
                    repo_id="user/my-model",
                    model_card=ModelCardConfig(base_model="Qwen/Qwen3-8B"),
                )

            assert readme.read_text() == "My custom README"

    def test_no_model_card_no_readme(self) -> None:
        """Without model_card, no README.md is generated (backward compat)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_api = MagicMock()

            with patch("tinker_cookbook.weights._publish.HfApi", return_value=mock_api):
                publish_to_hf_hub(
                    model_path=tmpdir,
                    repo_id="user/my-model",
                )

            assert not (Path(tmpdir) / "README.md").exists()

    def test_adapter_auto_detected_in_publish(self) -> None:
        """publish_to_hf_hub auto-detects adapter and generates peft card."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "adapter_config.json").write_text(
                json.dumps({"peft_type": "LORA", "r": 32})
            )

            mock_api = MagicMock()

            with patch("tinker_cookbook.weights._publish.HfApi", return_value=mock_api):
                publish_to_hf_hub(
                    model_path=tmpdir,
                    repo_id="user/my-adapter",
                    model_card=ModelCardConfig(base_model="Qwen/Qwen3-8B"),
                )

            content = (Path(tmpdir) / "README.md").read_text()
            assert "library_name: peft" in content
            assert "- lora" in content
            assert "PeftModel" in content
