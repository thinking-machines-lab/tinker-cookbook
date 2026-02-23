"""
Merge Tinker adapter weights to a HuggingFace model shard-by-shard, and save the new model to a given path.

This approach processes one safetensor shard at a time, so peak memory is proportional to the
largest shard rather than the full model. Supports dequantizing quantized checkpoints on the fly.

Please refer to the following documentation for how to download a Tinker sampler adapter weights: https://tinker-docs.thinkingmachines.ai/download-weights

Usage:
python merge_tinker_adapter_to_hf_model.py --hf-model <name_or_path_to_hf_model> --tinker-adapter-path <local_path_to_tinker_adapter_weights> --output-path <output_path_to_save_merged_model>
"""

import argparse
import collections
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Lightweight files needed to load model structure and tokenizer.
CONFIG_FILE_PATTERNS = ["*.json", "*.py"]

MAX_SHARD_SIZE = 10 * (1024**3)  # 10 GB


@dataclass
class MergeOp:
    """A single LoRA merge operation to apply to a base model weight.

    Stores the small lora_A/B matrices (rank-sized). The full delta is computed
    on the fly during shard processing to avoid holding model-sized tensors.
    """

    target_key: str
    lora_A: torch.Tensor  # (rank, in_dim) or (num_experts, rank, in_dim) for gpt-oss experts
    lora_B: torch.Tensor  # (out_dim, rank) or (num_experts, out_dim, rank), pre-scaled by alpha/r
    # True for gpt-oss expert weights where lora_A/B are 3D and need bmm
    is_expert_3d: bool = False
    # For gpt-oss interleaved expert weights (gate_up_proj): 0 for gate, 1 for up
    gpt_oss_interleave_idx: int | None = None


class ShardWriter:
    """Accumulates tensors and writes to numbered shard files with a size limit."""

    def __init__(self, output_path: Path, max_shard_size: int = MAX_SHARD_SIZE):
        self.output_path = output_path
        self.max_shard_size = max_shard_size
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_size: int = 0
        self.shard_count: int = 0
        self.shard_keys: list[list[str]] = []  # keys written per shard
        self.total_size: int = 0

    def add_tensor(self, key: str, tensor: torch.Tensor):
        size = tensor.nelement() * tensor.element_size()
        if self.pending and self.pending_size + size > self.max_shard_size:
            self.flush()
        self.pending[key] = tensor
        self.pending_size += size
        self.total_size += size

    def flush(self):
        if not self.pending:
            return
        self.shard_count += 1
        temp_name = f"shard-{self.shard_count:05d}.tmp.safetensors"
        save_file(self.pending, str(self.output_path / temp_name))
        self.shard_keys.append(list(self.pending.keys()))
        log(f"  Flushed {len(self.pending)} tensors to {temp_name}")
        self.pending = {}
        self.pending_size = 0

    def finalize(self) -> tuple[dict[str, str], int]:
        """Flush remaining tensors, rename to final names, return (weight_map, total_size)."""
        self.flush()
        total = self.shard_count
        weight_map: dict[str, str] = {}

        for i in range(total):
            temp_name = f"shard-{i + 1:05d}.tmp.safetensors"
            if total == 1:
                final_name = "model.safetensors"
            else:
                final_name = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
            (self.output_path / temp_name).rename(self.output_path / final_name)
            for key in self.shard_keys[i]:
                weight_map[key] = final_name

        log(f"  {total} output shard(s), total size {self.total_size / (1024**3):.1f} GB")
        return weight_map, self.total_size


def log(s: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {s}")


def resolve_model_dir(hf_model: str) -> Path:
    """Download config files to local disk (or use existing local path).

    Only fetches lightweight files (configs, tokenizer, model code).
    Safetensors weight shards are downloaded separately later.
    """
    if os.path.isdir(hf_model):
        log(f"Using local model directory: {hf_model}")
        return Path(hf_model)

    from huggingface_hub import snapshot_download

    log(f"Downloading config files for {hf_model}")
    local_dir = snapshot_download(repo_id=hf_model, allow_patterns=CONFIG_FILE_PATTERNS)
    return Path(local_dir)


def download_safetensors(hf_model: str):
    """Download safetensors weight shards for a remote model."""
    if os.path.isdir(hf_model):
        return  # Already local

    from huggingface_hub import snapshot_download

    log("Downloading weight shards")
    snapshot_download(repo_id=hf_model, allow_patterns=["*.safetensors"])


def load_meta_model(model_dir: Path) -> torch.nn.Module:
    """Load model on meta device (no memory) to get architecture and state dict keys."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.utils import import_utils

    # Some custom model code references is_torch_fx_available, removed in newer transformers.
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    return model


def get_input_shard_files(model_dir: Path) -> list[str]:
    """Get sorted list of safetensors shard filenames in the model directory."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return sorted(set(json.load(f)["weight_map"].values()))
    shard_files = sorted(model_dir.glob("*.safetensors"))
    assert shard_files, f"No .safetensors files found in {model_dir}"
    return [f.name for f in shard_files]


def load_adapter_weights(adapter_path: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load adapter weights and config from disk."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = load_file(
        os.path.expanduser(adapter_path + "/adapter_model.safetensors"), device=device
    )
    with open(os.path.expanduser(adapter_path + "/adapter_config.json"), "r") as f:
        config = json.load(f)
    return weights, config


def build_merge_ops(
    adapter_weights: dict[str, torch.Tensor],
    config: dict,
    model_state_keys: set[str],
    is_gpt_oss: bool,
) -> dict[str, list[MergeOp]]:
    """Build a mapping from target weight key to merge operations.

    Validates target keys against the model's state dict. Stores only the small
    lora_A/B matrices; the full delta is computed on the fly during shard processing.
    """
    scaling = config["lora_alpha"] / config["r"]
    adapter_weight_names = [n.replace(".lora_A", "") for n in adapter_weights if ".lora_A" in n]

    merge_ops: dict[str, list[MergeOp]] = {}

    for n in adapter_weight_names:
        target_key = n.replace("base_model.model.", "").replace("model.unembed_tokens", "lm_head")
        lora_A = adapter_weights[n.replace(".weight", ".lora_A.weight")].float()
        lora_B = adapter_weights[n.replace(".weight", ".lora_B.weight")].float() * scaling

        if ".experts" not in n:
            if is_gpt_oss:
                target_key = target_key.replace(".attn", ".self_attn")

            assert target_key in model_state_keys, (
                f"Target key '{target_key}' (from adapter key '{n}') not found in model state dict"
            )
            merge_ops.setdefault(target_key, []).append(
                MergeOp(target_key=target_key, lora_A=lora_A, lora_B=lora_B)
            )
        else:
            # Expert weights: 3D tensors (num_experts, rank, dim)
            assert len(lora_A.shape) == 3 and len(lora_B.shape) == 3, (
                lora_A.shape,
                lora_B.shape,
            )
            if lora_A.shape[0] == 1:
                assert lora_B.shape[0] > 1
                lora_A = lora_A.expand(lora_B.shape[0], -1, -1)
            elif lora_B.shape[0] == 1:
                assert lora_A.shape[0] > 1
                lora_B = lora_B.expand(lora_A.shape[0], -1, -1)

            target_key = target_key.replace(".w1.weight", ".gate_proj.weight")
            target_key = target_key.replace(".w3.weight", ".up_proj.weight")
            target_key = target_key.replace(".w2.weight", ".down_proj.weight")

            if not is_gpt_oss:
                # Pre-slice per expert: each slice is 2D (rank, dim), same math as non-expert
                for exp_idx in range(lora_A.shape[0]):
                    target_key_exp = target_key.replace(".experts", f".experts.{exp_idx}")
                    assert target_key_exp in model_state_keys, (
                        f"Target key '{target_key_exp}' (from adapter key '{n}') not found in model state dict"
                    )
                    merge_ops.setdefault(target_key_exp, []).append(
                        MergeOp(
                            target_key=target_key_exp,
                            lora_A=lora_A[exp_idx],
                            lora_B=lora_B[exp_idx],
                        )
                    )
            else:
                # gpt-oss: fused/interleaved weights, keep 3D for bmm
                if target_key.endswith((".gate_proj.weight", ".up_proj.weight")):
                    target_key = target_key.replace(".gate_proj.weight", ".gate_up_proj").replace(
                        ".up_proj.weight", ".gate_up_proj"
                    )
                    idx = 0 if target_key.endswith(".gate_up_proj") else 1
                    assert target_key in model_state_keys, (
                        f"Target key '{target_key}' (from adapter key '{n}') not found in model state dict"
                    )
                    merge_ops.setdefault(target_key, []).append(
                        MergeOp(
                            target_key=target_key,
                            lora_A=lora_A,
                            lora_B=lora_B,
                            is_expert_3d=True,
                            gpt_oss_interleave_idx=idx,
                        )
                    )
                else:
                    target_key = target_key.replace(".down_proj.weight", ".down_proj")
                    assert target_key in model_state_keys, (
                        f"Target key '{target_key}' (from adapter key '{n}') not found in model state dict"
                    )
                    merge_ops.setdefault(target_key, []).append(
                        MergeOp(
                            target_key=target_key,
                            lora_A=lora_A,
                            lora_B=lora_B,
                            is_expert_3d=True,
                        )
                    )

    return merge_ops


def apply_merge_op(tensors: dict[str, torch.Tensor], op: MergeOp):
    """Compute LoRA delta on the fly and merge it into the shard's tensors."""
    target = tensors[op.target_key]

    if op.is_expert_3d:
        # gpt-oss expert: 3D bmm
        # (num_experts, lora_rank, in_dim), (num_experts, out_dim, lora_rank)
        # -> (num_experts, in_dim, out_dim)
        delta = torch.bmm(op.lora_A.transpose(-1, -2), op.lora_B.transpose(-1, -2)).to(
            target.device
        )
        if op.gpt_oss_interleave_idx is not None:
            target_view = target[:, :, op.gpt_oss_interleave_idx :: 2]
            assert target_view.shape == delta.shape, (
                op.target_key,
                target_view.shape,
                delta.shape,
            )
            new_data = target_view.float() + delta
            target[:, :, op.gpt_oss_interleave_idx :: 2] = new_data.to(target_view.dtype)
        else:
            assert target.shape == delta.shape, (op.target_key, target.shape, delta.shape)
            tensors[op.target_key] = (target.float() + delta).to(target.dtype)
    else:
        # 2D: non-expert or pre-sliced per-expert
        # (lora_rank, in_dim), (out_dim, lora_rank) -> (out_dim, in_dim)
        delta = torch.nn.functional.linear(op.lora_A.T, op.lora_B).T.to(target.device)
        assert target.shape == delta.shape, (op.target_key, target.shape, delta.shape)
        tensors[op.target_key] = (target.float() + delta).to(target.dtype)


def setup_dequantization(model_dir: Path) -> tuple | None:
    """Set up decompressor if model config has quantization_config.

    Returns (quant_compressor, quant_scheme) or None.
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        loaded_config = json.load(f)
    if "quantization_config" not in loaded_config:
        return None

    quant_config = loaded_config["quantization_config"]
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.quantization.quant_scheme import QuantizationScheme

    model_compressor = ModelCompressor.from_compression_config(quant_config)
    assert model_compressor is not None, "Model compressor not found"
    assert model_compressor.quantization_compressor is not None, "Quantization compressor not found"

    quant_compressor = model_compressor.quantization_compressor[quant_config["format"]]
    quant_scheme = QuantizationScheme.model_validate(quant_config["config_groups"]["group_0"])
    return quant_compressor, quant_scheme


def dequantize_tensors(
    tensors: dict[str, torch.Tensor],
    quant_compressor: object,
    quant_scheme: object,
) -> dict[str, torch.Tensor]:
    """Dequantize quantized tensors, keeping non-quantized ones as-is."""
    decompressed: dict[str, torch.Tensor] = {}
    decompressed_module_paths: set[str] = set()

    for module_path, weight_data in quant_compressor.decompress_from_state_dict(
        tensors, collections.defaultdict(lambda: quant_scheme)
    ):
        decompressed[f"{module_path}.weight"] = weight_data["weight"]
        decompressed_module_paths.add(module_path)

    for name, param in tensors.items():
        if name.rsplit(".", 1)[0] not in decompressed_module_paths:
            decompressed[name] = param

    return decompressed


def copy_non_weight_files(model_dir: Path, output_path: Path):
    """Copy config and tokenizer files from model dir to output."""
    for pattern in CONFIG_FILE_PATTERNS:
        for item in sorted(model_dir.glob(pattern)):
            shutil.copy2(item, output_path / item.name)
            log(f"  Copied {item.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Tinker adapter weights into a HuggingFace model shard-by-shard."
    )
    parser.add_argument(
        "--tinker-adapter-path", type=str, required=True, help="Path to the Tinker adapter"
    )
    parser.add_argument(
        "--hf-model", type=str, required=True, help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to save the merged model"
    )
    parser.add_argument(
        "--dequantize",
        action="store_true",
        help="Dequantize quantized weights to full precision",
    )
    args = parser.parse_args()

    # Step 1: Download config files (lightweight)
    model_dir = resolve_model_dir(args.hf_model)

    # Step 2: Load model on meta device for architecture info and state dict keys
    log("Loading model structure (meta device)")
    meta_model = load_meta_model(model_dir)
    is_gpt_oss = "GptOss" in str(type(meta_model))
    model_state_keys = set(meta_model.state_dict().keys())
    del meta_model
    if is_gpt_oss:
        log("Detected gpt-oss model architecture")

    # Step 3: Download safetensors weight files
    download_safetensors(args.hf_model)

    # Step 4: Discover input shard files
    shard_files = get_input_shard_files(model_dir)
    log(f"Found {len(shard_files)} input shard(s)")

    # Step 5: Load adapter, build merge ops
    log("Loading adapter weights")
    adapter_weights, adapter_config = load_adapter_weights(args.tinker_adapter_path)

    log("Building merge ops")
    merge_ops = build_merge_ops(adapter_weights, adapter_config, model_state_keys, is_gpt_oss)
    total_ops = sum(len(ops) for ops in merge_ops.values())
    log(f"Merge ops: {total_ops} operations across {len(merge_ops)} target keys")

    # Step 6: Set up dequantization if requested
    dequant_info = setup_dequantization(model_dir)
    if dequant_info:
        log("Dequantization enabled")

    # Step 7: Process each input shard → dequantize → merge LoRA → write output shards
    os.makedirs(args.output_path, exist_ok=True)
    output_path = Path(args.output_path)
    writer = ShardWriter(output_path)

    for i, shard_file in enumerate(shard_files):
        log(f"Processing shard {i + 1}/{len(shard_files)}: {shard_file}")

        tensors = load_file(str(model_dir / shard_file))

        if dequant_info:
            tensors = dequantize_tensors(tensors, *dequant_info)

        for key in list(tensors.keys()):
            ops = merge_ops.pop(key, [])
            for op in ops:
                apply_merge_op(tensors, op)

        for key, tensor in tensors.items():
            writer.add_tensor(key, tensor)
        del tensors
        writer.flush()

    output_weight_map, total_size = writer.finalize()
    assert len(merge_ops) == 0, f"Merge ops not applied: {merge_ops}"

    # Step 8: Copy non-weight files (config, tokenizer, etc.)
    log("Copying non-weight files")
    copy_non_weight_files(model_dir, output_path)

    # Step 9: Generate index from what we actually wrote
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    index_path = output_path / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    log(f"Wrote {index_path.name}")

    if dequant_info:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.output_path, trust_remote_code=True)
        del config.quantization_config
        config.save_pretrained(args.output_path)

    log(f"Merged model saved to {args.output_path}")


if __name__ == "__main__":
    main()
