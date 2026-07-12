"""Tests for the chat agent's prompt rendering.

Schema-card *aggregation* (data side) lives in the persistence layer and is
tested in ``tinker_cookbook/tokendb/registry_backend_test.py``; this covers
the studio-side rendering of those cards into prompt text.
"""

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from tinker_cookbook.tokendb.registry_backend import RegistryBackend
from tinker_cookbook.tokendb.schema import TokenRow
from tinker_cookbook.tokendb.writer import TokenDbWriter
from tinker_cookbook.tokendb_studio.agent_prompt import build_system_prompt, format_schema_card


def make_row(**overrides) -> TokenRow:
    defaults: dict = {
        "split": "train",
        "iteration": 0,
        "group_idx": 0,
        "traj_idx": 0,
        "step_idx": 0,
        "ob_tokens": [1, 2, 3],
        "ac_tokens": [4, 5],
    }
    defaults.update(overrides)
    return TokenRow(**defaults)


@pytest.fixture
def keyed_runs(tmp_path: Path) -> dict:
    with TokenDbWriter(tmp_path / "run-a", context={}) as writer:
        writer.append_rows(
            [make_row(metrics={"acc": 1.0, "shared": 0.5}, attrs={"dataset": "gsm8k"})]
        )
        id_a = writer.run_id
    with TokenDbWriter(tmp_path / "run-b", context={}) as writer:
        writer.append_rows([make_row(metrics={"shared": 0.1}, tags=["hard"])])
        id_b = writer.run_id
    return {"a": id_a, "b": id_b}


def test_prompt_rendering_attributes_partial_keys(keyed_runs: dict):
    text = format_schema_card(RegistryBackend(refresh_ttl_s=0.0).schema_card())
    assert "Observed keys across runs (2 runs)" in text
    # `acc` exists only in run a; `shared` everywhere (no suffix).
    assert f"`acc` (only: {keyed_runs['a']})" in text
    assert "`shared` (only:" not in text
    assert "`shared`" in text


def test_single_run_card_rendering():
    text = format_schema_card(
        {"metrics_keys": ["acc"], "attrs_keys": [], "tags": [], "keys_truncated": False}
    )
    assert "This run's observed keys" in text
    assert "`acc`" in text


def test_build_system_prompt_injects_card_and_sql_url():
    text = build_system_prompt(
        sql_url="/api/sql",
        schema_card={"metrics_keys": ["acc"], "keys_truncated": False},
    )
    assert '"/api/sql"' in text
    assert "This run's observed keys" in text
