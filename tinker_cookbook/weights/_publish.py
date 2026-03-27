"""Publish model weights to HuggingFace Hub."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import HfApi

from tinker_cookbook.weights._model_card import ModelCardConfig, generate_model_card

logger = logging.getLogger(__name__)


def publish_to_hf_hub(
    *,
    model_path: str,
    repo_id: str,
    private: bool = True,
    token: str | None = None,
    model_card: ModelCardConfig | None = None,
) -> str:
    """Push a model or adapter directory to HuggingFace Hub.

    Works with outputs from :func:`build_hf_model`, :func:`build_lora_adapter`,
    or any HuggingFace-compatible model directory.

    Args:
        model_path: Local path to the model or adapter directory to upload.
        repo_id: HuggingFace Hub repository ID (e.g. ``"user/my-model"``).
        private: Whether the repository should be private. Defaults to
            ``True`` for safety.
        token: HuggingFace API token. If ``None`` (default), uses the
            ``HF_TOKEN`` environment variable or cached login from
            ``hf auth login``.
        model_card: Optional model card configuration. When provided, a
            ``README.md`` is auto-generated and included in the upload.
            If the directory already contains a ``README.md``, the existing
            file is preserved and a warning is logged.

    Returns:
        URL of the published repository.
    """
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"model_path does not exist or is not a directory: {model_path}")

    if model_card is not None:
        readme_path = path / "README.md"
        if readme_path.exists():
            logger.warning(
                "model_path already contains README.md; skipping model card generation. "
                "Remove the existing README.md to auto-generate a model card."
            )
        else:
            card = generate_model_card(config=model_card, repo_id=repo_id, model_path=model_path)
            card.save(str(readme_path))

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(path), repo_id=repo_id, repo_type="model")

    return f"https://huggingface.co/{repo_id}"
