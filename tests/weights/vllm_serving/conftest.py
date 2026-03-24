"""Shared helpers for vLLM adapter serving tests.

These tests require GPU(s) and vLLM installed in an isolated venv
(vLLM pins its own torch/transformers which conflict with the main project).

Setup:
    bash tests/weights/vllm_serving/setup_env.sh

Run:
    /tmp/vllm-test-env/bin/python -m pytest tests/weights/vllm_serving/ -v -s

See requirements.txt for dependencies and README.md for full instructions.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch


def _vllm_available() -> bool:
    try:
        importlib.import_module("vllm")
        return True
    except ImportError:
        return False


# Skip the entire directory at collection time if vllm or GPU is unavailable.
# This prevents pytest from importing test files that have top-level vllm imports.
collect_ignore_glob: list[str] = []
if not _vllm_available() or not torch.cuda.is_available():
    collect_ignore_glob = ["test_*.py"]

# All runtime imports are conditional — only loaded when tests will actually run.
if _vllm_available() and torch.cuda.is_available():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from tinker_cookbook.weights import build_lora_adapter

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.lora.request import LoRARequest

LORA_RANK = 8
LORA_ALPHA = 16
PROMPT = "The capital of France is"


def save_tinker_adapter(
    adapter_path: Path,
    weights: dict[str, torch.Tensor],
    rank: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
) -> None:
    """Save weights in Tinker adapter format."""
    adapter_path.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(adapter_path / "adapter_model.safetensors"))
    (adapter_path / "adapter_config.json").write_text(
        json.dumps({"lora_alpha": lora_alpha, "r": rank})
    )


def convert_and_load(
    model_name: str,
    adapter_path: Path,
    peft_path: Path,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Run build_lora_adapter and return (peft_weights, peft_config)."""
    build_lora_adapter(
        base_model=model_name,
        adapter_path=str(adapter_path),
        output_path=str(peft_path),
    )
    peft_weights = load_file(str(peft_path / "adapter_model.safetensors"))
    peft_config = json.loads((peft_path / "adapter_config.json").read_text())
    return peft_weights, peft_config


def generate(llm: LLM, prompt: str, lora_request: LoRARequest | None = None) -> str:
    """Generate text, optionally with a LoRA adapter."""
    params = SamplingParams(max_tokens=20, temperature=0.0)
    outputs = llm.generate([prompt], sampling_params=params, lora_request=lora_request)
    return outputs[0].outputs[0].text


def load_hf_config_dict(model_name: str) -> dict:
    """Load a model's config.json as a plain dict.

    Avoids transformers version dependency — useful when vLLM pins an older
    transformers that doesn't recognize newer model_types (e.g. qwen3_5).
    """
    config_path = hf_hub_download(model_name, "config.json")
    return json.loads(Path(config_path).read_text())
