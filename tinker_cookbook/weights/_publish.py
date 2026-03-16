"""Publish model weights to HuggingFace Hub."""

from __future__ import annotations

from pathlib import Path


def publish_to_hf_hub(
    *,
    model_path: str,
    repo_id: str,
    private: bool = True,
    token: str | None = None,
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

    Returns:
        URL of the published repository.
    """
    from huggingface_hub import HfApi

    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"model_path does not exist or is not a directory: {model_path}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(path), repo_id=repo_id, repo_type="model")

    return f"https://huggingface.co/{repo_id}"
