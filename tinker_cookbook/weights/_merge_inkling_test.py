from __future__ import annotations

import torch

from tinker_cookbook.weights._merge import (
    apply_merge_op,
    detect_merge_profile,
    plan_merge_ops,
    validate_merge_op_shapes,
)
from tinker_cookbook.weights._merge_inkling import detect_profile


def _pair(prefix: str, lora_A: torch.Tensor, lora_B: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.lora_A.weight": lora_A,
        f"{prefix}.lora_B.weight": lora_B,
    }


def _config() -> dict:
    return {
        "architectures": ["InklingForConditionalGeneration"],
        "model_type": "inkling_mm_model",
        "text_config": {"n_shared_experts": 2},
    }


def test_detect_profile() -> None:
    profile = detect_profile(_config(), {"model.llm.layers.0.attn.wk_dv.weight"})

    assert profile is not None
    assert profile.model_family == "inkling"
    assert profile.expert_layout == "fused_interleaved"
    assert profile.num_shared_experts == 2
    assert (
        detect_merge_profile(_config(), {"model.llm.layers.0.attn.wk_dv.weight"}).model_family
        == "inkling"
    )


def test_plan_and_apply_all_inkling_layouts() -> None:
    hidden = 3
    intermediate = 2
    num_experts = 2
    num_shared_experts = 2

    state = {
        "model.llm.layers.0.attn.wk_dv.weight": torch.zeros(2, hidden),
        "model.llm.layers.0.mlp.w13_dn.weight": torch.zeros(2 * intermediate, hidden),
        "model.llm.layers.0.mlp.w2_md.weight": torch.zeros(hidden, intermediate),
        "model.llm.layers.2.mlp.experts.w13_weight": torch.zeros(
            num_experts, 2 * intermediate, hidden
        ),
        "model.llm.layers.2.mlp.experts.w2_weight": torch.zeros(num_experts, hidden, intermediate),
        "model.llm.layers.2.mlp.shared_experts.shared_w13_weight": torch.zeros(
            num_shared_experts, 2 * intermediate, hidden
        ),
        "model.llm.layers.2.mlp.shared_experts.shared_w2_weight": torch.zeros(
            num_shared_experts, hidden, intermediate
        ),
        "model.llm.unembed.weight": torch.zeros(5, hidden),
    }

    dense_gate_up_A = torch.ones(1, hidden)
    dense_gate_up_B = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1)
    routed_w1_A = torch.ones(1, 1, hidden)
    routed_w1_B = torch.arange(1, 5, dtype=torch.float32).reshape(2, 2, 1)
    routed_w3_A = torch.ones(1, 1, hidden)
    routed_w3_B = torch.arange(5, 9, dtype=torch.float32).reshape(2, 2, 1)
    routed_w2_A = torch.arange(1, 5, dtype=torch.float32).reshape(2, 1, 2)
    routed_w2_B = torch.arange(1, 4, dtype=torch.float32).reshape(1, 3, 1)
    shared_w1_A = torch.ones(1, hidden)
    shared_w1_B = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1)
    shared_w3_A = torch.ones(1, hidden)
    shared_w3_B = torch.arange(5, 9, dtype=torch.float32).reshape(4, 1)
    shared_w2_A = torch.arange(1, 5, dtype=torch.float32).reshape(1, 4)
    shared_w2_B = torch.arange(1, 4, dtype=torch.float32).reshape(3, 1)

    adapter = {
        **_pair(
            "language_model.layers.0.attn.wk_dv",
            torch.ones(1, hidden),
            torch.ones(2, 1),
        ),
        **_pair(
            "language_model.layers.0.mlp.gate_up_proj",
            dense_gate_up_A,
            dense_gate_up_B,
        ),
        **_pair(
            "language_model.layers.0.mlp.down_proj",
            torch.ones(1, intermediate),
            torch.ones(hidden, 1),
        ),
        **_pair("language_model.layers.2.mlp.experts.w1", routed_w1_A, routed_w1_B),
        **_pair("language_model.layers.2.mlp.experts.w2", routed_w2_A, routed_w2_B),
        **_pair("language_model.layers.2.mlp.experts.w3", routed_w3_A, routed_w3_B),
        **_pair(
            "language_model.layers.2.mlp.shared_experts.w1",
            shared_w1_A,
            shared_w1_B,
        ),
        **_pair(
            "language_model.layers.2.mlp.shared_experts.w2",
            shared_w2_A,
            shared_w2_B,
        ),
        **_pair(
            "language_model.layers.2.mlp.shared_experts.w3",
            shared_w3_A,
            shared_w3_B,
        ),
        **_pair("language_model.lm_head", torch.ones(1, hidden), torch.ones(5, 1)),
    }

    profile = detect_merge_profile(_config(), set(state))
    ops = plan_merge_ops(adapter, {"lora_alpha": 1, "r": 1}, set(state), profile)
    validate_merge_op_shapes(ops, {key: tuple(value.shape) for key, value in state.items()})
    for op_list in ops.values():
        for op in op_list:
            apply_merge_op(state, op)

    dense = state["model.llm.layers.0.mlp.w13_dn.weight"]
    torch.testing.assert_close(dense[0::2], dense_gate_up_B[:intermediate] @ dense_gate_up_A)
    torch.testing.assert_close(dense[1::2], dense_gate_up_B[intermediate:] @ dense_gate_up_A)

    routed_w13 = state["model.llm.layers.2.mlp.experts.w13_weight"]
    torch.testing.assert_close(
        routed_w13[:, 0::2],
        torch.bmm(routed_w1_B, routed_w1_A.expand(num_experts, -1, -1)),
    )
    torch.testing.assert_close(
        routed_w13[:, 1::2],
        torch.bmm(routed_w3_B, routed_w3_A.expand(num_experts, -1, -1)),
    )
    torch.testing.assert_close(
        state["model.llm.layers.2.mlp.experts.w2_weight"],
        torch.bmm(routed_w2_B.expand(num_experts, -1, -1), routed_w2_A),
    )

    shared_w13 = state["model.llm.layers.2.mlp.shared_experts.shared_w13_weight"]
    torch.testing.assert_close(
        shared_w13[:, 0::2],
        torch.bmm(
            shared_w1_B.reshape(num_shared_experts, intermediate, 1),
            shared_w1_A.expand(num_shared_experts, -1, -1),
        ),
    )
    torch.testing.assert_close(
        shared_w13[:, 1::2],
        torch.bmm(
            shared_w3_B.reshape(num_shared_experts, intermediate, 1),
            shared_w3_A.expand(num_shared_experts, -1, -1),
        ),
    )
    torch.testing.assert_close(
        state["model.llm.layers.2.mlp.shared_experts.shared_w2_weight"],
        torch.bmm(
            shared_w2_B.expand(num_shared_experts, -1, -1),
            shared_w2_A.reshape(1, num_shared_experts, intermediate).permute(1, 0, 2).contiguous(),
        ),
    )

    assert state["model.llm.layers.0.attn.wk_dv.weight"].abs().sum() > 0
    assert state["model.llm.layers.0.mlp.w2_md.weight"].abs().sum() > 0
    assert state["model.llm.unembed.weight"].abs().sum() > 0
