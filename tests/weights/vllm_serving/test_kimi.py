"""vLLM serving tests for Kimi-K2: DeepSeek architecture with per-expert expansion.

Kimi-K2 (~236B parameters) is too large for routine local testing.
The adapter conversion correctness is verified by unit tests in
tests/weights/test_adapter_kimi.py. Add vLLM serving tests here
when a smaller Kimi model is available or infra supports it.
"""
