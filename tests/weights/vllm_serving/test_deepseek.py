"""vLLM serving tests for DeepSeek V3/V3.1.

DeepSeek V3/V3.1 (model_type=deepseek_v3) is intentionally unsupported by
build_lora_adapter — vLLM and SGLang do not support LoRA for this architecture.
The unsupported error is verified by tests/weights/test_adapter_deepseek.py.

Add vLLM serving tests here if/when DeepSeek LoRA support lands in vLLM.
"""
