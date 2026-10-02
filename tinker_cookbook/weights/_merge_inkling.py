"""Inkling merge planning for raw Hugging Face checkpoint layouts."""

from __future__ import annotations

import torch

from tinker_cookbook.exceptions import WeightsMergeError
from tinker_cookbook.weights._merge import (
    MergeOp,
    MergeProfile,
    expand_expert_lora_tensors,
)
from tinker_cookbook.weights._merge_utils import (
    extract_adapter_weight_names,
    plan_standard_op,
    remap_adapter_name,
    validate_adapter_config,
)

_NAME_REMAPS = (
    ("base_model.model.", ""),
    ("language_model.lm_head", "model.llm.unembed"),
    ("language_model.", "model.llm."),
)


def detect_profile(model_config: dict, model_state_keys: set[str]) -> MergeProfile | None:
    """Detect Inkling multimodal checkpoints."""
    architectures = model_config.get("architectures", [])
    if model_config.get("model_type") != "inkling_mm_model" and not any(
        architecture.startswith("Inkling") for architecture in architectures
    ):
        return None

    text_config = model_config.get("text_config", {})
    return MergeProfile(
        model_family="inkling",
        expert_layout="fused_interleaved",
        num_shared_experts=text_config.get("n_shared_experts"),
    )


def _append_routed_expert_op(
    *,
    target_key: str,
    projection: str,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    adapter_name: str,
    model_state_keys: set[str],
    ops: dict[str, list[MergeOp]],
) -> None:
    if projection not in ("w1", "w2", "w3"):
        raise WeightsMergeError(
            f"Unsupported Inkling routed expert projection {projection!r} in {adapter_name!r}"
        )
    lora_A, lora_B = expand_expert_lora_tensors(lora_A, lora_B)
    prefix = target_key.removesuffix(f".{projection}.weight")

    if projection in ("w1", "w3"):
        target_key = f"{prefix}.w13_weight"
        op = MergeOp(
            target_key=target_key,
            lora_A=lora_A,
            lora_B=lora_B,
            is_expert_3d=True,
            fused_proj_idx={"w1": 0, "w3": 1}[projection],
            fused_proj_interleaved=True,
            fused_axis=1,
        )
    else:
        target_key = f"{prefix}.w2_weight"
        op = MergeOp(
            target_key=target_key,
            lora_A=lora_A,
            lora_B=lora_B,
            is_expert_3d=True,
        )

    if target_key not in model_state_keys:
        raise WeightsMergeError(
            f"Adapter weight {adapter_name!r} mapped to {target_key!r} "
            f"which does not exist in the model state dict"
        )
    ops.setdefault(target_key, []).append(op)


def _append_shared_expert_op(
    *,
    target_key: str,
    projection: str,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    adapter_name: str,
    profile: MergeProfile,
    model_state_keys: set[str],
    ops: dict[str, list[MergeOp]],
) -> None:
    if projection not in ("w1", "w2", "w3"):
        raise WeightsMergeError(
            f"Unsupported Inkling shared expert projection {projection!r} in {adapter_name!r}"
        )
    num_experts = profile.num_shared_experts
    if not num_experts:
        raise WeightsMergeError("Inkling config is missing text_config.n_shared_experts")

    prefix = target_key.removesuffix(f".{projection}.weight")
    rank = lora_A.shape[-2]

    if projection in ("w1", "w3"):
        if lora_B.shape[0] % num_experts:
            raise WeightsMergeError(
                f"Shared expert adapter {adapter_name!r} has output dimension "
                f"{lora_B.shape[0]}, which is not divisible by {num_experts}"
            )
        target_key = f"{prefix}.shared_w13_weight"
        lora_A = lora_A.unsqueeze(0).expand(num_experts, -1, -1)
        lora_B = lora_B.reshape(num_experts, -1, rank)
        op = MergeOp(
            target_key=target_key,
            lora_A=lora_A,
            lora_B=lora_B,
            is_expert_3d=True,
            fused_proj_idx={"w1": 0, "w3": 1}[projection],
            fused_proj_interleaved=True,
            fused_axis=1,
        )
    else:
        if lora_A.shape[1] % num_experts:
            raise WeightsMergeError(
                f"Shared expert adapter {adapter_name!r} has input dimension "
                f"{lora_A.shape[1]}, which is not divisible by {num_experts}"
            )
        target_key = f"{prefix}.shared_w2_weight"
        lora_A = lora_A.reshape(rank, num_experts, -1).permute(1, 0, 2)
        lora_B = lora_B.unsqueeze(0).expand(num_experts, -1, -1)
        op = MergeOp(
            target_key=target_key,
            lora_A=lora_A,
            lora_B=lora_B,
            is_expert_3d=True,
        )

    if target_key not in model_state_keys:
        raise WeightsMergeError(
            f"Adapter weight {adapter_name!r} mapped to {target_key!r} "
            f"which does not exist in the model state dict"
        )
    ops.setdefault(target_key, []).append(op)


def _append_dense_gate_up_ops(
    *,
    target_key: str,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    adapter_name: str,
    model_state_keys: set[str],
    ops: dict[str, list[MergeOp]],
) -> None:
    target_key = target_key.replace(".mlp.gate_up_proj.weight", ".mlp.w13_dn.weight")
    if target_key not in model_state_keys:
        raise WeightsMergeError(
            f"Adapter weight {adapter_name!r} mapped to {target_key!r} "
            f"which does not exist in the model state dict"
        )
    if lora_B.shape[0] % 2:
        raise WeightsMergeError(
            f"Dense gate/up adapter {adapter_name!r} has odd output dimension {lora_B.shape[0]}"
        )

    gate_B, up_B = lora_B.chunk(2, dim=0)
    for projection, projection_B in enumerate((gate_B, up_B)):
        ops.setdefault(target_key, []).append(
            MergeOp(
                target_key=target_key,
                lora_A=lora_A,
                lora_B=projection_B,
                fused_proj_idx=projection,
                fused_proj_interleaved=True,
                fused_axis=0,
            )
        )


def plan_merge_ops(
    adapter_weights: dict[str, torch.Tensor],
    adapter_config: dict,
    model_state_keys: set[str],
    profile: MergeProfile,
) -> dict[str, list[MergeOp]]:
    """Plan adapter updates against Inkling's raw, interleaved checkpoint tensors."""
    scaling = validate_adapter_config(adapter_config, profile)
    ops: dict[str, list[MergeOp]] = {}

    for adapter_name in extract_adapter_weight_names(adapter_weights):
        target_key = remap_adapter_name(adapter_name, list(_NAME_REMAPS))
        lora_A = adapter_weights[adapter_name.replace(".weight", ".lora_A.weight")].float()
        lora_B = (
            adapter_weights[adapter_name.replace(".weight", ".lora_B.weight")].float() * scaling
        )

        if ".mlp.experts." in adapter_name:
            _append_routed_expert_op(
                target_key=target_key,
                projection=adapter_name.removesuffix(".weight").rsplit(".", 1)[-1],
                lora_A=lora_A,
                lora_B=lora_B,
                adapter_name=adapter_name,
                model_state_keys=model_state_keys,
                ops=ops,
            )
        elif ".mlp.shared_experts." in adapter_name:
            _append_shared_expert_op(
                target_key=target_key,
                projection=adapter_name.removesuffix(".weight").rsplit(".", 1)[-1],
                lora_A=lora_A,
                lora_B=lora_B,
                adapter_name=adapter_name,
                profile=profile,
                model_state_keys=model_state_keys,
                ops=ops,
            )
        elif target_key.endswith(".mlp.gate_up_proj.weight"):
            _append_dense_gate_up_ops(
                target_key=target_key,
                lora_A=lora_A,
                lora_B=lora_B,
                adapter_name=adapter_name,
                model_state_keys=model_state_keys,
                ops=ops,
            )
        else:
            if target_key.endswith(".mlp.down_proj.weight"):
                target_key = target_key.replace(".mlp.down_proj.weight", ".mlp.w2_md.weight")
            plan_standard_op(
                target_key,
                lora_A,
                lora_B,
                adapter_name,
                profile,
                model_state_keys,
                ops,
            )

    return ops
