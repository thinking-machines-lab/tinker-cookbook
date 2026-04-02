"""End-to-end weight export tests across model families and quantization formats.

Tests the full lifecycle: train 1 step → save → download → export via
different paths (full merge, LoRA adapter, quantized merge). Covers
multiple model families to exercise model-specific merge profiles and
quantization formats.

Test matrix:
    Dense (Qwen3):     merge, adapter export
    MoE (Qwen3.5):    merge, adapter export, FP8 experts merge
    Quantized (DeepSeek V3.1): FP8 experts merge (CPU + GPU)

Requires TINKER_API_KEY and network access. Skipped otherwise.
DeepSeek tests also require significant disk space for model download.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import cast

import datasets
import pytest
import tinker
import torch
from safetensors.torch import load_file

from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.weights import build_hf_model, build_lora_adapter, download

BATCH_SIZE = 4
MAX_LENGTH = 512
LORA_RANK = 8


# ---------------------------------------------------------------------------
# Shared training helper
# ---------------------------------------------------------------------------


def _make_sft_dataset(
    model_name: str, renderer_name: str
) -> SupervisedDatasetFromHFDataset:
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    ds = datasets.load_dataset("allenai/tulu-3-sft-mixture")
    ds = cast(datasets.DatasetDict, ds)
    train_ds = ds["train"].take(BATCH_SIZE)

    def map_fn(row: dict) -> tinker.Datum:
        return conversation_to_datum(row["messages"], renderer, MAX_LENGTH)

    return SupervisedDatasetFromHFDataset(train_ds, batch_size=BATCH_SIZE, map_fn=map_fn)


def _train_one_step(
    model_name: str,
    renderer_name: str,
    checkpoint_name: str,
) -> str:
    """Train for 1 step and return tinker:// path."""
    sft_dataset = _make_sft_dataset(model_name, renderer_name)

    async def _run() -> str:
        sc = tinker.ServiceClient()
        tc = await sc.create_lora_training_client_async(
            base_model=model_name,
            rank=LORA_RANK,
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


def _download_adapter(tinker_path: str, output_dir: Path) -> Path:
    """Download adapter and verify files exist."""
    adapter_dir = Path(download(tinker_path=tinker_path, output_dir=str(output_dir)))
    assert (adapter_dir / "adapter_model.safetensors").exists()
    assert (adapter_dir / "adapter_config.json").exists()
    return adapter_dir


def _verify_merged_model(output_path: Path, *, expect_config_key: str | None = None) -> None:
    """Verify output looks like a valid HF model directory."""
    assert (output_path / "config.json").exists(), "config.json missing"
    assert any(output_path.glob("*.safetensors")), "No safetensors files"
    assert (output_path / "tokenizer.json").exists() or (
        output_path / "tokenizer_config.json"
    ).exists(), "Tokenizer files missing"

    if expect_config_key:
        config = json.loads((output_path / "config.json").read_text())
        assert expect_config_key in config, f"{expect_config_key} missing from config.json"


def _load_all_tensors(output_path: Path) -> dict[str, torch.Tensor]:
    """Load all safetensors shards from output directory."""
    tensors: dict[str, torch.Tensor] = {}
    for sf in sorted(output_path.glob("*.safetensors")):
        tensors.update(load_file(str(sf)))
    return tensors


# ---------------------------------------------------------------------------
# Dense model tests (Qwen3)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(900)
class TestDenseQwen3:
    """Train → download → merge / adapter export for dense Qwen3."""

    MODEL = "Qwen/Qwen3-4B-Instruct-2507"
    RENDERER = "qwen3_instruct"

    @pytest.fixture(scope="class")
    def adapter_dir(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("dense_qwen3")
        tinker_path = _train_one_step(self.MODEL, self.RENDERER, "dense_qwen3_e2e")
        return _download_adapter(tinker_path, root / "adapter")

    def test_shard_merge(self, adapter_dir, tmp_path):
        output = tmp_path / "merged"
        build_hf_model(
            base_model=self.MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            merge_strategy="shard",
        )
        _verify_merged_model(output)

        # All weights should be BF16 (no quantization)
        tensors = _load_all_tensors(output)
        weight_keys = [k for k in tensors if k.endswith(".weight")]
        assert all(
            tensors[k].dtype in (torch.bfloat16, torch.float32) for k in weight_keys
        ), "Expected all weights to be BF16/FP32"

    def test_lora_adapter_export(self, adapter_dir, tmp_path):
        output = tmp_path / "peft"
        build_lora_adapter(
            base_model=self.MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
        )
        # PEFT adapter should have adapter_config.json with PEFT fields
        assert (output / "adapter_config.json").exists()
        config = json.loads((output / "adapter_config.json").read_text())
        assert "peft_type" in config, "PEFT config missing peft_type"
        assert config.get("r") == LORA_RANK, f"Expected rank {LORA_RANK}"


# ---------------------------------------------------------------------------
# MoE model tests (Qwen3.5)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(900)
class TestMoEQwen35:
    """Train → download → merge / adapter export for MoE Qwen3.5."""

    MODEL = "Qwen/Qwen3.5-35B-A3B"
    RENDERER = "qwen3_5"

    @pytest.fixture(scope="class")
    def adapter_dir(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("moe_qwen35")
        tinker_path = _train_one_step(self.MODEL, self.RENDERER, "moe_qwen35_e2e")
        return _download_adapter(tinker_path, root / "adapter")

    def test_shard_merge(self, adapter_dir, tmp_path):
        output = tmp_path / "merged"
        build_hf_model(
            base_model=self.MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            merge_strategy="shard",
        )
        _verify_merged_model(output)

        # MoE model should have expert weights
        tensors = _load_all_tensors(output)
        expert_keys = [k for k in tensors if ".experts." in k]
        assert len(expert_keys) > 0, "Expected expert weights in MoE model output"

    def test_lora_adapter_export(self, adapter_dir, tmp_path):
        output = tmp_path / "peft"
        build_lora_adapter(
            base_model=self.MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
        )
        assert (output / "adapter_config.json").exists()
        config = json.loads((output / "adapter_config.json").read_text())
        assert "peft_type" in config

    def test_fp8_quantized_merge(self, adapter_dir, tmp_path):
        """FP8 quantized merge for Qwen3.5 MoE (non-DeepSeek model)."""
        output = tmp_path / "merged_fp8"
        build_hf_model(
            base_model=self.MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            quantize="experts-fp8",
            serving_format="vllm",
        )
        _verify_merged_model(output, expect_config_key="compression_config")

        # Verify routed experts are FP8, rest is BF16
        tensors = _load_all_tensors(output)
        expert_weights = [
            k for k in tensors
            if ".experts." in k
            and ".shared_experts." not in k
            and k.endswith(".weight")
        ]
        for key in expert_weights:
            assert tensors[key].dtype == torch.float8_e4m3fn, (
                f"Expected FP8 for routed expert {key}"
            )
            scale_key = key.removesuffix(".weight") + ".weight_scale"
            assert scale_key in tensors, f"Missing scale for {key}"

        # Check compression_config
        config = json.loads((output / "config.json").read_text())
        assert config["compression_config"]["quant_method"] == "compressed-tensors"


# ---------------------------------------------------------------------------
# Quantized export tests (DeepSeek V3.1 — FP8 experts)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(3600)  # Large model — allow 1 hour
class TestDeepSeekFP8:
    """Train → download → FP8 quantized merge for DeepSeek V3.1.

    Tests the FP8 blockwise quantization path with a real adapter.
    DeepSeek V3.1 has native FP8 weights that need dequant → merge → requant.
    """

    MODEL = "deepseek-ai/DeepSeek-V3.1"
    RENDERER = "deepseekv3"

    @pytest.fixture(scope="class")
    def adapter_dir(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("deepseek_fp8")
        tinker_path = _train_one_step(self.MODEL, self.RENDERER, "deepseek_fp8_e2e")
        return _download_adapter(tinker_path, root / "adapter")

    def test_fp8_quantized_merge_cpu(self, adapter_dir, tmp_path):
        """FP8 quantized merge on CPU — baseline correctness check."""
        output = tmp_path / "merged_cpu"
        build_hf_model(
            base_model=self.MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            quantize="experts-fp8",
            serving_format="vllm",
            device="cpu",
        )
        _verify_merged_model(output, expect_config_key="compression_config")
        self._verify_fp8_output(output)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_fp8_quantized_merge_gpu(self, adapter_dir, tmp_path):
        """FP8 quantized merge on GPU — verify GPU path produces same output."""
        output = tmp_path / "merged_gpu"
        build_hf_model(
            base_model=self.MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(output),
            quantize="experts-fp8",
            serving_format="vllm",
            device="cuda",
        )
        _verify_merged_model(output, expect_config_key="compression_config")
        self._verify_fp8_output(output)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
    def test_cpu_gpu_equivalence(self, adapter_dir, tmp_path):
        """CPU and GPU FP8 quantized merge produce identical output."""
        cpu_out = tmp_path / "cpu"
        gpu_out = tmp_path / "gpu"

        build_hf_model(
            base_model=self.MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(cpu_out),
            quantize="experts-fp8",
            serving_format="vllm",
            device="cpu",
        )
        build_hf_model(
            base_model=self.MODEL,
            adapter_path=str(adapter_dir),
            output_path=str(gpu_out),
            quantize="experts-fp8",
            serving_format="vllm",
            device="cuda",
        )

        cpu_tensors = _load_all_tensors(cpu_out)
        gpu_tensors = _load_all_tensors(gpu_out)

        assert set(cpu_tensors.keys()) == set(gpu_tensors.keys()), "Key mismatch"

        for key in cpu_tensors:
            cpu_t = cpu_tensors[key]
            gpu_t = gpu_tensors[key]
            assert cpu_t.shape == gpu_t.shape, f"Shape mismatch for {key}"
            assert cpu_t.dtype == gpu_t.dtype, f"Dtype mismatch for {key}"
            if cpu_t.dtype == torch.float8_e4m3fn:
                # FP8 tensors: compare as float
                assert torch.equal(
                    cpu_t.to(torch.float32), gpu_t.to(torch.float32)
                ), f"FP8 tensor mismatch for {key}"
            else:
                assert torch.equal(cpu_t, gpu_t), f"Tensor mismatch for {key}"

    @staticmethod
    def _verify_fp8_output(output: Path) -> None:
        """Verify FP8 quantized output structure."""
        tensors = _load_all_tensors(output)

        # Check routed expert weights are FP8
        expert_weights = [
            k for k in tensors
            if ".mlp.experts." in k
            and ".shared_experts." not in k
            and k.endswith(".weight")
        ]
        for key in expert_weights:
            assert tensors[key].dtype == torch.float8_e4m3fn, (
                f"Expected FP8 for routed expert {key}, got {tensors[key].dtype}"
            )
            # Should have corresponding scale
            scale_key = key.removesuffix(".weight") + ".weight_scale"
            assert scale_key in tensors, f"Missing scale for {key}"
            assert tensors[scale_key].dtype == torch.float32, (
                f"Expected float32 scale for {key}"
            )

        # Check dense weights are NOT FP8
        dense_weights = [
            k for k in tensors
            if k.endswith(".weight")
            and ".mlp.experts." not in k
        ]
        for key in dense_weights:
            assert tensors[key].dtype != torch.float8_e4m3fn, (
                f"Dense weight {key} should not be FP8"
            )

        # Check vLLM compression config
        config = json.loads((output / "config.json").read_text())
        comp = config["compression_config"]
        assert comp["quant_method"] == "compressed-tensors"
        assert comp["format"] == "float-quantized"
        # Verify ignore list doesn't include routed experts
        ignore = set(comp.get("ignore", []))
        for key in expert_weights:
            prefix = key.removesuffix(".weight")
            assert prefix not in ignore, f"Routed expert {prefix} in ignore list"
