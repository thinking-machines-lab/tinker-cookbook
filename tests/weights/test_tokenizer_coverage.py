"""Verify that the tokenizer file allowlist covers all supported models.

Downloads only tokenizer files (not weights) from HuggingFace for one
representative model per family, copies them using our allowlist, and
confirms ``AutoTokenizer.from_pretrained`` succeeds on the result.

Requires network access. Skipped when TINKER_API_KEY is not set (same
gate as other integration tests).
"""

import shutil
import tempfile
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

from tinker_cookbook.weights._export import _TOKENIZER_AND_PROCESSOR_FILES

# One representative per model family — covers distinct tokenizer layouts.
_REPRESENTATIVE_MODELS = (
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "deepseek-ai/DeepSeek-V3.1",
    "openai/gpt-oss-20b",
    "moonshotai/Kimi-K2-Thinking",
    "moonshotai/Kimi-K2.5",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
)


def _download_tokenizer_files(model: str, dest: Path) -> list[str]:
    """Download allowlisted tokenizer files + *.py modules into *dest*."""
    copied = []
    for name in _TOKENIZER_AND_PROCESSOR_FILES:
        try:
            local = hf_hub_download(model, name)
            shutil.copyfile(local, dest / name)
            copied.append(name)
        except Exception:
            pass

    # *.py modules (copied by copy_model_code_files in the real pipeline)
    for f in list_repo_files(model):
        if f.endswith(".py") and "/" not in f:
            local = hf_hub_download(model, f)
            shutil.copyfile(local, dest / f)
            copied.append(f)

    return copied


@pytest.mark.integration
@pytest.mark.timeout(120)
@pytest.mark.parametrize("model", _REPRESENTATIVE_MODELS, ids=lambda m: m.split("/")[-1])
def test_tokenizer_loadable_from_allowlist(model: str) -> None:
    """Tokenizer loads from just the allowlisted files for each model family."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir)
        copied = _download_tokenizer_files(model, dest)
        assert copied, f"No tokenizer files found for {model}"

        tok = AutoTokenizer.from_pretrained(str(dest), trust_remote_code=True)
        tokens = tok.encode("Hello world")
        assert len(tokens) > 0, f"Tokenizer for {model} produced empty encoding"


@pytest.mark.integration
@pytest.mark.timeout(120)
@pytest.mark.parametrize("model", _REPRESENTATIVE_MODELS, ids=lambda m: m.split("/")[-1])
def test_tokenizer_class_not_corrupted(model: str) -> None:
    """tokenizer_class in our copied config matches the HF source."""
    import json

    src_path = hf_hub_download(model, "tokenizer_config.json")
    expected = json.loads(Path(src_path).read_text()).get("tokenizer_class")

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir)
        _download_tokenizer_files(model, dest)
        actual = json.loads((dest / "tokenizer_config.json").read_text()).get("tokenizer_class")

    assert actual == expected, f"{model}: tokenizer_class={actual!r}, expected {expected!r}"
