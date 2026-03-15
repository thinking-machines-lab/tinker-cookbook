"""LoRA adapter merge logic.

Handles merging Tinker LoRA adapter weights into HuggingFace model weights,
including model-specific quirks (GPT-OSS interleaved experts, vision model
prefixes, fused vs separate expert weights).
"""

from __future__ import annotations

import torch


def apply_merged_weight(target: torch.Tensor, merged_lora: torch.Tensor) -> None:
    """Add a merged LoRA delta to a model weight tensor in-place."""
    assert target.shape == merged_lora.shape, (target.shape, merged_lora.shape)
    new_data = target.float() + merged_lora.float().to(target.device)
    target.copy_(new_data.to(target.dtype))


def merge_adapter_weights(
    base_model: torch.nn.Module, adapter_weights: dict[str, torch.Tensor], config: dict
) -> None:
    """Merge LoRA adapter weights into a base model's state dict in-place.

    Handles:
    - Standard (non-expert) linear layers
    - Separate per-expert weights (Qwen3 MoE, DeepSeek, Kimi)
    - Fused expert weights with interleaved layout (GPT-OSS)
    - Fused expert weights with concatenated layout (Qwen3.5, Qwen3-VL)
    - Vision model name prefix remapping
    - GPT-OSS attention name remapping

    Args:
        base_model: The HuggingFace model to merge into.
        adapter_weights: Dict of LoRA weight tensors from the adapter.
        config: Adapter config dict with ``lora_alpha`` and ``r`` keys.
    """
    scaling = config["lora_alpha"] / config["r"]
    adapter_weight_names = [n.replace(".lora_A", "") for n in adapter_weights if ".lora_A" in n]

    model_state_dict = base_model.state_dict()
    is_gpt_oss = "GptOss" in str(type(base_model))
    is_fused_experts = any(k.endswith(".experts.gate_up_proj") for k in model_state_dict)
    name_remaps = {
        "base_model.model.": "",
        "model.unembed_tokens": "lm_head",
    }
    if any(k.startswith("model.language_model.") for k in model_state_dict):
        # Tinker adapter doesn't include the language_model prefix for vision models
        name_remaps["model."] = "model.language_model."

    for n in adapter_weight_names:
        target_key = n
        for old, new in name_remaps.items():
            target_key = target_key.replace(old, new)
        lora_A = adapter_weights[n.replace(".weight", ".lora_A.weight")].float()
        lora_B = adapter_weights[n.replace(".weight", ".lora_B.weight")].float() * scaling
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

            if not is_fused_experts:
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
                # Single/fused weight and shared w13, shape is <num_experts, in_dim, out_dim>
                if target_key.endswith(".gate_proj.weight"):
                    idx = 0
                    target_key = target_key.replace(".gate_proj.weight", ".gate_up_proj")
                elif target_key.endswith(".up_proj.weight"):
                    idx = 1
                    target_key = target_key.replace(".up_proj.weight", ".gate_up_proj")
                else:
                    idx = None

                if idx is not None:
                    if is_gpt_oss:
                        # gpt-oss has interleaved w13
                        target = model_state_dict[target_key][:, :, idx::2]
                    else:
                        sz = model_state_dict[target_key].shape[-1] // 2
                        target = model_state_dict[target_key][:, :, idx * sz : (idx + 1) * sz]
                else:
                    target_key = target_key.replace(".down_proj.weight", ".down_proj")
                    target = model_state_dict[target_key]
                apply_merged_weight(target, merged_lora)
