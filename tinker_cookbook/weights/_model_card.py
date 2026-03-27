"""Model card generation for HuggingFace Hub publishing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData


@dataclass
class ModelCardConfig:
    """Configuration for auto-generating a HuggingFace model card.

    Contains only standard HuggingFace metadata fields. List fields default
    to empty (omitted from the card when empty). Additional tags are appended
    to the auto-generated defaults (``tinker``, ``tinker-cookbook``, and
    ``peft``/``lora`` for adapters).

    Example::

        config = ModelCardConfig(
            base_model="Qwen/Qwen3-8B",
            datasets=["my-org/my-dataset"],
            tags=["sft"],
            license="apache-2.0",
        )
    """

    base_model: str
    """HuggingFace model ID used as the base (e.g. ``"Qwen/Qwen3-8B"``)."""

    datasets: list[str] = field(default_factory=list)
    """HuggingFace dataset IDs used for training."""

    tags: list[str] = field(default_factory=list)
    """Additional tags to include (appended to auto-generated defaults)."""

    license: str | None = None
    """SPDX license identifier (e.g. ``"apache-2.0"``)."""

    language: list[str] = field(default_factory=list)
    """Language codes (e.g. ``["en"]``)."""


def generate_model_card(
    *,
    config: ModelCardConfig,
    repo_id: str | None = None,
    model_path: str | None = None,
) -> ModelCard:
    """Generate a HuggingFace model card from config.

    The returned :class:`~huggingface_hub.ModelCard` can be previewed with
    ``str(card)`` or saved with ``card.save("README.md")``.

    Args:
        config: Model card metadata.
        repo_id: HuggingFace repo ID (e.g. ``"user/my-model"``). Used for
            the card title and usage snippet. If ``None``, a generic title
            is used.
        model_path: Local path to the model directory. If provided,
            auto-detects whether the output is a LoRA adapter (by checking
            for ``adapter_config.json``) or a merged model, and adjusts the
            card content accordingly.

    Returns:
        A :class:`~huggingface_hub.ModelCard` with YAML frontmatter and
        markdown body.
    """
    is_adapter = _detect_adapter(model_path)
    tags = _build_tags(config, is_adapter)
    library_name = "peft" if is_adapter else "transformers"

    card_data = ModelCardData(
        base_model=config.base_model,
        library_name=library_name,
        pipeline_tag="text-generation",
        tags=tags,
    )
    if config.datasets:
        card_data.datasets = config.datasets
    if config.license:
        card_data.license = config.license
    if config.language:
        card_data.language = config.language

    title = repo_id or config.base_model
    description = (
        f"This model was fine-tuned from "
        f"[{config.base_model}](https://huggingface.co/{config.base_model}) "
        f"using [Tinker](https://thinkingmachines.ai/tinker) and "
        f"[tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)."
    )
    format_label = "LoRA adapter (PEFT)" if is_adapter else "Merged model"
    usage_snippet = _build_usage_snippet(config.base_model, repo_id or "your-repo-id", is_adapter)

    return ModelCard.from_template(
        card_data,
        template_str=_TEMPLATE,
        title=title,
        description=description,
        base_model=config.base_model,
        format_label=format_label,
        usage_snippet=usage_snippet,
        tinker_cookbook_version=_get_version("tinker_cookbook"),
        transformers_version=_get_version("transformers"),
        torch_version=_get_version("torch"),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _detect_adapter(model_path: str | None) -> bool:
    """Check if model_path contains a PEFT adapter."""
    if model_path is None:
        return False
    return (Path(model_path) / "adapter_config.json").exists()


def _build_tags(config: ModelCardConfig, is_adapter: bool) -> list[str]:
    """Build the complete tag list."""
    tags = ["tinker", "tinker-cookbook"]
    if is_adapter:
        tags.extend(["peft", "lora"])
    tags.extend(config.tags)
    return tags


def _build_usage_snippet(base_model: str, repo_id: str, is_adapter: bool) -> str:
    """Build the Python usage code snippet."""
    if is_adapter:
        return (
            f"from peft import PeftModel\n"
            f"from transformers import AutoModelForCausalLM\n"
            f"\n"
            f'base = AutoModelForCausalLM.from_pretrained("{base_model}")\n'
            f'model = PeftModel.from_pretrained(base, "{repo_id}")'
        )
    return (
        f"from transformers import AutoModelForCausalLM\n"
        f"\n"
        f'model = AutoModelForCausalLM.from_pretrained("{repo_id}")'
    )


def _get_version(package: str) -> str:
    """Get package version, returning 'unknown' on failure."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


_TEMPLATE = """\
---
{{ card_data }}
---

# {{ title }}

{{ description }}

## Model details

- **Base model:** [{{ base_model }}](https://huggingface.co/{{ base_model }})
- **Format:** {{ format_label }}

## Usage

```python
{{ usage_snippet }}
```

## Framework versions

- tinker-cookbook: {{ tinker_cookbook_version }}
- transformers: {{ transformers_version }}
- torch: {{ torch_version }}
"""
