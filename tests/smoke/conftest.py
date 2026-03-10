"""Pytest configuration for smoke tests.

Recipes NOT yet covered by smoke tests:
  - code_rl: requires external sandbox service (SandboxFusion)
  - search_tool: requires running Chroma vector DB + embedding API
  - verifiers_rl: requires verifiers framework environment
  - if_rl: requires if_verifiable library + IFBench data
  - math_rl: could use ArithmeticDatasetBuilder (partially tested via smoke_tests.py)
  - rubric: needs generated JSONL data (has generate_data.py script)
  - rl_basic, sl_basic, rl_loop, sl_loop: standalone tutorial scripts (not full recipes)
  - prompt_distillation: needs a local JSONL data file
  - harbor_rl: needs Modal + downloaded Harbor tasks
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip all smoke tests when TINKER_API_KEY is not set."""
    if os.environ.get("TINKER_API_KEY"):
        return
    skip = pytest.mark.skip(reason="TINKER_API_KEY not set")
    for item in items:
        item.add_marker(skip)
