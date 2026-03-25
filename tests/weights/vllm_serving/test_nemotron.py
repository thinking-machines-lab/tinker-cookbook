"""vLLM serving tests for Nemotron-3.

Nemotron-3 (model_type=nemotron_h) adapter conversion is not yet supported
due to the non-standard 'backbone.*' weight prefix. The unsupported error
is verified by unit tests in tinker_cookbook/weights/adapter_test.py.

Add vLLM serving tests here when Nemotron adapter conversion is implemented.
"""
