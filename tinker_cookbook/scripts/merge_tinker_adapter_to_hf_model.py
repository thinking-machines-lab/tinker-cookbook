"""
Merge Tinker adapter weights to a HuggingFace model, and save the new model to a given path.

Please refer to the following documentation for how to download a Tinker sampler adapter weights: https://tinker-docs.thinkingmachines.ai/download-weights

Usage:
python merge_tinker_adapter_to_hf_model.py --hf-model <name_or_path_to_hf_model> --tinker-adapter-path <local_path_to_tinker_adapter_weights> --output-path <output_path_to_save_merged_model>
"""

import argparse
import json
import os
from datetime import datetime

import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer)


def load_model(model_path: str):
    config = AutoConfig.from_pretrained(model_path)
    cls = AutoModelForCausalLM
    if getattr(config, "vision_config") is not None:
        cls = AutoModelForImageTextToText
    return cls.from_pretrained(model_path, device_map="auto", dtype=torch.bfloat16)


def load_adapter_weights(adapter_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = load_file(
        os.path.expanduser(adapter_path + "/adapter_model.safetensors"), device=device
    )
    with open(os.path.expanduser(adapter_path + "/adapter_config.json"), "r") as f:
        config = json.load(f)
    return weights, config


def log(s: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {s}")


def apply_merged_weight(target: torch.Tensor, merged_lora: torch.Tensor):
    assert target.shape == merged_lora.shape, (target.shape, merged_lora.shape)
    new_data = target.float() + merged_lora.float().to(target.device)
    target.copy_(new_data.to(target.dtype))


def merge_adapter_weights(
    base_model: torch.nn.Module, adapter_weights: dict[str, torch.Tensor], config: dict
):
    scaling = config["lora_alpha"] / config["r"]
    adapter_weight_names = [n.replace(".lora_A", "") for n in adapter_weights if ".lora_A" in n]

    model_state_dict = base_model.state_dict()
    is_gpt_oss = "GptOss" in str(type(base_model))
    is_qwen3vl_moe = "Qwen3VLMoe" in str(type(base_model))

    for n in adapter_weight_names:
        target_key = n.replace("base_model.model.", "").replace("model.unembed_tokens", "lm_head")
        lora_A = adapter_weights[n.replace(".weight", ".lora_A.weight")].float()
        lora_B = adapter_weights[n.replace(".weight", ".lora_B.weight")].float() * scaling
        if is_qwen3vl_moe:
            target_key = target_key.replace("model.layers.", "model.language_model.layers.")
        if ".experts" not in n:
            if is_gpt_oss:
                target_key = target_key.replace(".attn", ".self_attn")
            assert target_key in model_state_dict, (n, target_key)
            # (lora_rank, in_dim), (out_dim lora_rank) -> (out_dim, in_dim)
            merged_lora = torch.nn.functional.linear(lora_A.T, lora_B).T
            assert merged_lora.shape == model_state_dict[target_key].shape, (
                n,
                merged_lora.shape,
                model_state_dict[target_key].shape,
            )
            apply_merged_weight(model_state_dict[target_key], merged_lora)
        else:
            # Experts weights are fused together, and some are potentially shared across experts
            assert len(lora_A.shape) == 3 and len(lora_B.shape) == 3, (lora_A.shape, lora_B.shape)
            if lora_A.shape[0] == 1:
                assert lora_B.shape[0] > 1
                lora_A = lora_A.expand(lora_B.shape[0], -1, -1)
            elif lora_B.shape[0] == 1:
                assert lora_A.shape[0] > 1
                lora_B = lora_B.expand(lora_A.shape[0], -1, -1)
            # (num_experts, lora_rank, in_dim),(num_experts, out_dim, lora_rank) -> (num_experts, in_dim, out_dim)
            merged_lora = torch.bmm(lora_A.transpose(-1, -2), lora_B.transpose(-1, -2))

            target_key = target_key.replace(".w1.weight", ".gate_proj.weight")
            target_key = target_key.replace(".w3.weight", ".up_proj.weight")
            target_key = target_key.replace(".w2.weight", ".down_proj.weight")

            if not (is_gpt_oss or is_qwen3vl_moe):
                # Separate linear/weight per expert, target shape is <out_dim, in_dim>
                merged_lora = merged_lora.transpose(-1, -2)  # -> (num_experts, out_dim, in_dim)
                for exp_idx in range(merged_lora.shape[0]):
                    target_key_exp = target_key.replace(".experts", f".experts.{exp_idx}")
                    assert target_key_exp in model_state_dict, (n, target_key_exp)
                    assert merged_lora[exp_idx].shape == model_state_dict[target_key_exp].shape, (
                        target_key_exp,
                        merged_lora[exp_idx].shape,
                        model_state_dict[target_key_exp].shape,
                    )
                    apply_merged_weight(model_state_dict[target_key_exp], merged_lora[exp_idx])
            else:
                # Single/fused weight and interleaved w13 for gpt-oss, shape is <num_experts, in_dim, out_dim>
                if target_key.endswith((".gate_proj.weight", ".up_proj.weight")):
                    target_key = target_key.replace(".gate_proj.weight", ".gate_up_proj").replace(
                        ".up_proj.weight", ".gate_up_proj"
                    )
                    idx = 0 if target_key.endswith(".gate_up_proj") else 1
                    target = model_state_dict[target_key][:, :, idx::2]
                else:
                    target_key = target_key.replace(".down_proj.weight", ".down_proj")
                    target = model_state_dict[target_key]
                apply_merged_weight(target, merged_lora)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tinker-adapter-path", type=str, required=True, help="Path to the Tinker adapter"
    )
    parser.add_argument(
        "--hf-model", type=str, required=True, help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to save the merged model"
    )
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=False)

    log("Loading HF Model")
    hf_model = load_model(args.hf_model)

    log("Loading Adapter Weights")
    adapter_weights, adapter_config = load_adapter_weights(args.tinker_adapter_path)

    log("Merging Adapter Weights")
    merge_adapter_weights(hf_model, adapter_weights, adapter_config)

    log("Saving Merged Model")
    hf_model.save_pretrained(args.output_path)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    tokenizer.save_pretrained(args.output_path)
    log(f"Merged model saved to {args.output_path}")


if __name__ == "__main__":
    main()
