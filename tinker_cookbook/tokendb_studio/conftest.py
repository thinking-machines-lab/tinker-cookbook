"""Shared test fixtures for the tokendb studio package."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_tokendb_registry(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the run registry and segment cache at per-test directories.

    Mirrors the fixture in ``tinker_cookbook/tokendb/conftest.py``: writers
    register runs on construction (best-effort), so without this the test
    suite would pollute the developer's real registry under
    ``~/.cache/tinker-cookbook/tokendb/runs``.
    """
    monkeypatch.setenv("TINKER_TOKENDB_REGISTRY", str(tmp_path / "tokendb-registry"))
    monkeypatch.setenv("TINKER_TOKENDB_SEGCACHE", str(tmp_path / "tokendb-segcache"))
