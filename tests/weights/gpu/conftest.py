"""GPU weight tests — require GPU + TINKER_API_KEY + model weights on NFS.

These tests exercise the full weight export pipeline with real models:
train 1 step via Tinker API → download adapter → merge/export/quantize.

Setup:
    1. Ensure TINKER_API_KEY is set
    2. Ensure HF model weights are cached on NFS:
       export HF_HUB_CACHE=~/huggingface/hub

Run:
    HF_HUB_CACHE=~/huggingface/hub pytest tests/weights/gpu/ -v --timeout=3600

    # Single model family:
    HF_HUB_CACHE=~/huggingface/hub pytest tests/weights/gpu/test_qwen3.py -v

    # Only vLLM tests (requires vLLM venv):
    /tmp/vllm-test-env/bin/python -m pytest tests/weights/gpu/ -v -k vllm
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import cast

import datasets
import pytest
import tinker
import torch
from safetensors.torch import load_file

from tinker_cookbook import renderers
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.weights import download

# ---------------------------------------------------------------------------
# Directory-level skip: no GPU → skip all tests here
# ---------------------------------------------------------------------------

collect_ignore_glob: list[str] = []
if not torch.cuda.is_available():
    collect_ignore_glob = ["test_*.py"]

# Some models (Nemotron) require trust_remote_code for config loading.
# Set this globally so AutoConfig.from_pretrained doesn't prompt for stdin.
os.environ.setdefault("HF_TRUST_REMOTE_CODE", "1")


# ---------------------------------------------------------------------------
# vLLM availability
# ---------------------------------------------------------------------------


def _vllm_available() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


skip_no_vllm = pytest.mark.skipif(not _vllm_available(), reason="vLLM not installed")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
MAX_LENGTH = 512
LORA_RANK = 8


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------


def train_one_step(
    model_name: str,
    renderer_name: str,
    checkpoint_name: str,
    *,
    train_unembed: bool = False,
) -> str:
    """Train for 1 step and return tinker:// path.

    Args:
        train_unembed: Whether to train the unembed (lm_head) LoRA.
            Default False because many models tie embeddings, making
            the unembed weight absent from the shard-by-shard state dict.
    """
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    ds = datasets.load_dataset("allenai/tulu-3-sft-mixture")
    ds = cast(datasets.DatasetDict, ds)
    train_ds = ds["train"].take(BATCH_SIZE)

    def map_fn(row: dict) -> tinker.Datum:
        return conversation_to_datum(row["messages"], renderer, MAX_LENGTH)

    sft_dataset = SupervisedDatasetFromHFDataset(train_ds, batch_size=BATCH_SIZE, map_fn=map_fn)

    async def _run() -> str:
        sc = tinker.ServiceClient()
        tc = await sc.create_lora_training_client_async(
            base_model=model_name,
            rank=LORA_RANK,
            train_unembed=train_unembed,
        )
        batch = sft_dataset.get_batch(0)
        fwd_bwd = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
        await fwd_bwd.result_async()
        optim = await tc.optim_step_async({"learning_rate": 1e-4})
        await optim.result_async()
        sampler_resp = await tc.save_weights_for_sampler_async(checkpoint_name)
        result = await sampler_resp.result_async()
        return result.path

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Download / verification helpers
# ---------------------------------------------------------------------------


def download_adapter(tinker_path: str, output_dir: Path) -> Path:
    """Download adapter and verify files exist."""
    adapter_dir = Path(download(tinker_path=tinker_path, output_dir=str(output_dir)))
    assert (adapter_dir / "adapter_model.safetensors").exists()
    assert (adapter_dir / "adapter_config.json").exists()
    return adapter_dir


def verify_merged_model(output_path: Path, *, expect_config_key: str | None = None) -> None:
    """Verify output looks like a valid HF model directory."""
    assert (output_path / "config.json").exists(), "config.json missing"
    assert any(output_path.glob("*.safetensors")), "No safetensors files"
    # Check tokenizer
    has_tokenizer = (
        (output_path / "tokenizer.json").exists()
        or (output_path / "tokenizer_config.json").exists()
        or (output_path / "tiktoken.model").exists()  # Kimi uses tiktoken
    )
    assert has_tokenizer, "Tokenizer files missing"

    if expect_config_key:
        config = json.loads((output_path / "config.json").read_text())
        assert expect_config_key in config, f"{expect_config_key} missing from config.json"


def load_all_tensors(output_path: Path) -> dict[str, torch.Tensor]:
    """Load all safetensors shards from output directory."""
    tensors: dict[str, torch.Tensor] = {}
    for sf in sorted(output_path.glob("*.safetensors")):
        tensors.update(load_file(str(sf)))
    return tensors


def verify_fp8_output(output_path: Path) -> None:
    """Verify FP8 quantized output: routed experts in FP8, rest in BF16."""
    tensors = load_all_tensors(output_path)

    expert_weights = [
        k
        for k in tensors
        if ".experts." in k and ".shared_experts." not in k and k.endswith(".weight")
    ]
    assert len(expert_weights) > 0, "No routed expert weights found"

    for key in expert_weights:
        assert tensors[key].dtype == torch.float8_e4m3fn, (
            f"Expected FP8 for routed expert {key}, got {tensors[key].dtype}"
        )
        scale_key = key.removesuffix(".weight") + ".weight_scale"
        assert scale_key in tensors, f"Missing scale for {key}"
        assert tensors[scale_key].dtype == torch.float32

    dense_weights = [k for k in tensors if k.endswith(".weight") and ".experts." not in k]
    for key in dense_weights:
        assert tensors[key].dtype != torch.float8_e4m3fn, f"Dense weight {key} should not be FP8"

    config = json.loads((output_path / "config.json").read_text())
    comp = config["compression_config"]
    assert comp["quant_method"] == "compressed-tensors"
    assert comp["format"] == "float-quantized"
