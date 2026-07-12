"""Shared test fixtures for the tokendb package."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_tokendb_registry(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the run registry at a per-test directory.

    Writers register runs on construction (best-effort), so without this the
    test suite would pollute the developer's real registry under
    ``~/.cache/tinker-cookbook/tokendb/runs``.
    """
    monkeypatch.setenv("TINKER_TOKENDB_REGISTRY", str(tmp_path / "tokendb-registry"))
