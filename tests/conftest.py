"""Pytest configuration for integration tests.

Recipes NOT yet covered by integration tests:
  - code_rl: requires external sandbox service (SandboxFusion)
  - search_tool: requires running Chroma vector DB + embedding API
  - verifiers_rl: requires verifiers framework environment
  - if_rl: requires if_verifiable library + IFBench data
  - rubric: needs generated JSONL data (has generate_data.py script)
  - rl_basic, sl_basic, rl_loop, sl_loop: standalone tutorial scripts (not full recipes)
  - prompt_distillation: needs a local JSONL data file
  - harbor_rl: needs Modal + downloaded Harbor tasks
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip smoke tests locally when TINKER_API_KEY is not set. Fail on CI.

    Sandbox-backend smokes (Modal / Fystash) gate on their own credentials and
    do not need a Tinker API key.
    """
    if os.environ.get("TINKER_API_KEY"):
        return

    sandbox_only = {"test_modal_sandbox.py", "test_fystash_sandbox.py"}

    # Separate smoke tests from downstream_compat tests (which don't need API keys)
    # and from sandbox-only tests that use their own auth skip markers.
    smoke_items = [
        item
        for item in items
        if "downstream_compat" not in str(item.fspath)
        and os.path.basename(str(item.fspath)) not in sandbox_only
    ]
    if not smoke_items:
        return

    if os.environ.get("CI"):
        pytest.fail("TINKER_API_KEY is not set but CI=true — smoke tests require an API key")
    skip = pytest.mark.skip(
        reason="TINKER_API_KEY not set (set it or run pytest tinker_cookbook/ for unit tests)"
    )
    for item in smoke_items:
        item.add_marker(skip)
