"""Auto-apply the downstream_compat marker to every test in this directory.

These tests verify that tinker-cookbook's public API surface remains compatible
with downstream consumers (e.g., the internal monorepo). They are fast, require
no API keys or GPU, and run on every PR.

Run just these tests:
    uv run pytest tests/downstream_compat/
    uv run pytest -m downstream_compat
"""

import pytest


def pytest_collection_modifyitems(config, items):
    marker = pytest.mark.downstream_compat
    for item in items:
        if "downstream_compat" in str(item.fspath):
            item.add_marker(marker)
