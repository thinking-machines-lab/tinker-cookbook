"""E2E test: partial LoRA training → download → adapter export + full merge.

Trains models with partial LoRA configurations (attn-only, mlp-only) for
1 step, downloads the real adapter weights from Tinker, and verifies
build_lora_adapter and build_hf_model succeed with partial coverage.

Tests three architectures: dense (Qwen3-4B-Instruct-2507), dense Qwen3.5
(Qwen3.5-4B), and MoE (Qwen3.5-35B-A3B). All tests verify adapter export
and full weight merge.

Requires TINKER_API_KEY and network access. Skipped otherwise.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import cast

import datasets
import pytest
import tinker
from safetensors.torch import load_file

from tinker_cookbook import renderers
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.weights import build_hf_model, build_lora_adapter, download

BATCH_SIZE = 4
MAX_LENGTH = 512
LORA_RANK = 8


def _make_sft_dataset(model_name: str, renderer_name: str) -> "SupervisedDatasetFromHFDataset":
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    ds = datasets.load_dataset("allenai/tulu-3-sft-mixture")
    ds = cast(datasets.DatasetDict, ds)
    train_ds = ds["train"].take(BATCH_SIZE)

    def map_fn(row: dict) -> tinker.Datum:
        return conversation_to_datum(row["messages"], renderer, MAX_LENGTH)

    return SupervisedDatasetFromHFDataset(train_ds, batch_size=BATCH_SIZE, map_fn=map_fn)


def _train_one_step_and_save(
    *,
    model_name: str,
    renderer_name: str,
    train_attn: bool = True,
    train_mlp: bool = True,
    train_unembed: bool = True,
    checkpoint_name: str,
) -> str:
    """Train 1 step with the given LoRA flags and return the tinker:// path."""
    sft_dataset = _make_sft_dataset(model_name, renderer_name)

    async def _run() -> str:
        sc = tinker.ServiceClient()
        tc = await sc.create_lora_training_client_async(
            base_model=model_name,
            rank=LORA_RANK,
            train_attn=train_attn,
            train_mlp=train_mlp,
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


def _download_and_verify_adapter(tinker_path: str, output_dir: Path) -> Path:
    """Download adapter and verify basic structure."""
    adapter_dir = Path(download(tinker_path=tinker_path, output_dir=str(output_dir)))
    assert (adapter_dir / "adapter_model.safetensors").exists()
    assert (adapter_dir / "adapter_config.json").exists()
    return adapter_dir


# Markers in adapter key names that indicate MLP/expert modules.
# Attention keys are identified by the absence of these markers.
_MLP_MARKERS = {"gate_proj", "up_proj", "down_proj", "experts", "shared_experts", "w1", "w2", "w3"}
# Markers that indicate attention modules (covers both dense q/k/v/o_proj
# and Qwen3.5 linear_attn.in_proj_k style keys).
_ATTN_MARKERS = {
    "self_attn",
    "linear_attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_q",
    "in_proj_k",
    "in_proj_v",
}


def _has_any_marker(key: str, markers: set[str]) -> bool:
    parts = key.split(".")
    return any(p in markers for p in parts)


def _assert_only_attn_keys(raw_weights: dict[str, object]) -> None:
    """Assert no key in the adapter belongs to MLP/expert modules."""
    for key in raw_weights:
        assert not _has_any_marker(key, _MLP_MARKERS), (
            f"Expected attn-only but found MLP/expert key: {key}"
        )


def _assert_only_mlp_keys(raw_weights: dict[str, object]) -> None:
    """Assert no key in the adapter belongs to attention modules."""
    for key in raw_weights:
        assert not _has_any_marker(key, _ATTN_MARKERS), (
            f"Expected mlp-only but found attn key: {key}"
        )


def _assert_adapter_export(adapter_dir: Path, model_name: str, output_dir: Path) -> None:
    """Run build_lora_adapter and verify output."""
    build_lora_adapter(
        base_model=model_name,
        adapter_path=str(adapter_dir),
        output_path=str(output_dir),
    )
    assert (output_dir / "adapter_model.safetensors").exists()
    assert (output_dir / "adapter_config.json").exists()

    with open(output_dir / "adapter_config.json") as f:
        config = json.load(f)
    assert config["peft_type"] == "LORA"
    assert config["r"] == LORA_RANK


def _assert_full_merge(adapter_dir: Path, model_name: str, output_dir: Path) -> None:
    """Run build_hf_model and verify output."""
    build_hf_model(
        base_model=model_name,
        adapter_path=str(adapter_dir),
        output_path=str(output_dir),
    )
    assert (output_dir / "config.json").exists()
    assert any(output_dir.glob("*.safetensors"))


# -------------------------------------------------------------------------
# Dense model tests (Qwen3-4B-Instruct-2507)
# -------------------------------------------------------------------------

_DENSE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_DENSE_RENDERER = "qwen3"


@pytest.mark.integration
@pytest.mark.timeout(900)
class TestPartialLoraDense:
    """E2E: partial LoRA on dense Qwen3-4B → download → adapter export + full merge."""

    def test_attn_only(self) -> None:
        """train_attn=True, train_mlp=False, train_unembed=False."""
        tinker_path = _train_one_step_and_save(
            model_name=_DENSE_MODEL,
            renderer_name=_DENSE_RENDERER,
            train_attn=True,
            train_mlp=False,
            train_unembed=False,
            checkpoint_name="dense_attn_only",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter_dir = _download_and_verify_adapter(tinker_path, root / "adapter")
            raw_weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
            _assert_only_attn_keys(raw_weights)
            _assert_adapter_export(adapter_dir, _DENSE_MODEL, root / "peft")
            _assert_full_merge(adapter_dir, _DENSE_MODEL, root / "merged")

    def test_mlp_only(self) -> None:
        """train_attn=False, train_mlp=True, train_unembed=False."""
        tinker_path = _train_one_step_and_save(
            model_name=_DENSE_MODEL,
            renderer_name=_DENSE_RENDERER,
            train_attn=False,
            train_mlp=True,
            train_unembed=False,
            checkpoint_name="dense_mlp_only",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter_dir = _download_and_verify_adapter(tinker_path, root / "adapter")
            raw_weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
            _assert_only_mlp_keys(raw_weights)
            _assert_adapter_export(adapter_dir, _DENSE_MODEL, root / "peft")
            _assert_full_merge(adapter_dir, _DENSE_MODEL, root / "merged")


# -------------------------------------------------------------------------
# Qwen3.5 model tests (Qwen3.5-4B — different architecture from Qwen3)
# -------------------------------------------------------------------------

_QWEN35_MODEL = "Qwen/Qwen3.5-4B"
_QWEN35_RENDERER = "qwen3_5"


@pytest.mark.integration
@pytest.mark.timeout(900)
class TestPartialLoraQwen35:
    """E2E: partial LoRA on Qwen3.5-4B → download → adapter export + full merge.

    Qwen3.5 has a different architecture from Qwen3 (e.g. linear_attn with
    in_proj_k/in_proj_q/in_proj_v instead of separate q_proj/k_proj/v_proj).
    """

    def test_attn_only(self) -> None:
        """train_attn=True, train_mlp=False, train_unembed=False on Qwen3.5."""
        tinker_path = _train_one_step_and_save(
            model_name=_QWEN35_MODEL,
            renderer_name=_QWEN35_RENDERER,
            train_attn=True,
            train_mlp=False,
            train_unembed=False,
            checkpoint_name="qwen35_attn_only",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter_dir = _download_and_verify_adapter(tinker_path, root / "adapter")
            raw_weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
            _assert_only_attn_keys(raw_weights)
            _assert_adapter_export(adapter_dir, _QWEN35_MODEL, root / "peft")
            _assert_full_merge(adapter_dir, _QWEN35_MODEL, root / "merged")

    def test_mlp_only(self) -> None:
        """train_attn=False, train_mlp=True, train_unembed=False on Qwen3.5."""
        tinker_path = _train_one_step_and_save(
            model_name=_QWEN35_MODEL,
            renderer_name=_QWEN35_RENDERER,
            train_attn=False,
            train_mlp=True,
            train_unembed=False,
            checkpoint_name="qwen35_mlp_only",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter_dir = _download_and_verify_adapter(tinker_path, root / "adapter")
            raw_weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
            _assert_only_mlp_keys(raw_weights)
            _assert_adapter_export(adapter_dir, _QWEN35_MODEL, root / "peft")
            _assert_full_merge(adapter_dir, _QWEN35_MODEL, root / "merged")


# -------------------------------------------------------------------------
# MoE model tests (Qwen3.5-30B-A3B)
# -------------------------------------------------------------------------

_MOE_MODEL = "Qwen/Qwen3.5-35B-A3B"
_MOE_RENDERER = "qwen3_5"


@pytest.mark.integration
@pytest.mark.timeout(900)
class TestPartialLoraMoE:
    """E2E: partial LoRA on MoE Qwen3.5-35B-A3B → download → adapter export + full merge."""

    def test_attn_only(self) -> None:
        """train_attn=True, train_mlp=False, train_unembed=False on MoE."""
        tinker_path = _train_one_step_and_save(
            model_name=_MOE_MODEL,
            renderer_name=_MOE_RENDERER,
            train_attn=True,
            train_mlp=False,
            train_unembed=False,
            checkpoint_name="moe_attn_only",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter_dir = _download_and_verify_adapter(tinker_path, root / "adapter")
            raw_weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
            _assert_only_attn_keys(raw_weights)
            _assert_adapter_export(adapter_dir, _MOE_MODEL, root / "peft")
            _assert_full_merge(adapter_dir, _MOE_MODEL, root / "merged")

    def test_mlp_only(self) -> None:
        """train_attn=False, train_mlp=True, train_unembed=False on MoE."""
        tinker_path = _train_one_step_and_save(
            model_name=_MOE_MODEL,
            renderer_name=_MOE_RENDERER,
            train_attn=False,
            train_mlp=True,
            train_unembed=False,
            checkpoint_name="moe_mlp_only",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter_dir = _download_and_verify_adapter(tinker_path, root / "adapter")
            raw_weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
            _assert_only_mlp_keys(raw_weights)
            _assert_adapter_export(adapter_dir, _MOE_MODEL, root / "peft")
            _assert_full_merge(adapter_dir, _MOE_MODEL, root / "merged")
