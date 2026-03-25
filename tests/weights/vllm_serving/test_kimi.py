"""vLLM serving tests for Kimi-K2 / K2.5.

Kimi-K2 (model_type=kimi_k2, architecture=DeepseekV3ForCausalLM):
  - Adapter conversion supported (separate per-expert expansion)
  - vLLM LoRA supported via DeepseekV2ForCausalLM (SupportsLoRA)
  - ~1TB in bf16 — too large for routine e2e testing on 8xH200
  - Conversion correctness verified by tests/weights/test_adapter_kimi.py

Kimi-K2.5 (model_type=kimi_k25, architecture=KimiK25ForConditionalGeneration):
  - Adapter conversion supported
  - vLLM 0.18 does NOT implement SupportsLoRA for KimiK25ForConditionalGeneration
  - 595 GB in bf16 — fits on 8xH200 but no LoRA support to test

Add vLLM serving tests here when either:
  - A smaller Kimi-K2 variant is available, or
  - vLLM adds LoRA support for KimiK25ForConditionalGeneration
"""
