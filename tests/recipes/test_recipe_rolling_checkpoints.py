"""Smoke test for rolling checkpoint save and resume.

Verifies that:
1. Training completes with rolling_save_every=1 enabled.
2. Rolling checkpoint records (with "rolling": true) appear in checkpoints.jsonl.
3. Resuming from rolling checkpoints works correctly.
"""

import json
from pathlib import Path

import pytest

from tests.helpers import run_recipe

MODULE = "tinker_cookbook.recipes.chat_sl.train"
LOG_PATH = "/tmp/tinker-smoke-test/rolling_checkpoints"


def _read_checkpoints(log_path: str) -> list[dict]:
    """Read all records from checkpoints.jsonl."""
    ckpt_path = Path(log_path) / "checkpoints.jsonl"
    assert ckpt_path.exists(), f"checkpoints.jsonl not found at {ckpt_path}"
    return [json.loads(line) for line in ckpt_path.read_text().strip().split("\n")]


@pytest.mark.integration
def test_rolling_checkpoint_train():
    """Train SFT for 4 steps with rolling_save_every=1 and save_every=2.

    Expects rolling checkpoints at steps 1, 3 and periodic checkpoints at step 2.
    Step 4 gets a final checkpoint.
    """
    run_recipe(
        MODULE,
        [
            "behavior_if_log_dir_exists=delete",
            f"log_path={LOG_PATH}",
            "save_every=2",
            "rolling_save_every=1",
            "rolling_ttl_seconds=300",
        ],
        max_steps=4,
    )

    records = _read_checkpoints(LOG_PATH)

    # Should have rolling + periodic + final entries
    rolling_records = [r for r in records if r.get("rolling") is True]
    periodic_records = [r for r in records if r.get("rolling") is not True and r["name"] != "final"]
    final_records = [r for r in records if r["name"] == "final"]

    assert len(rolling_records) > 0, f"Expected rolling checkpoints, got: {records}"
    assert len(periodic_records) > 0, f"Expected periodic checkpoints, got: {records}"
    assert len(final_records) == 1, f"Expected exactly one final checkpoint, got: {records}"

    # Rolling records should have state_path but no sampler_path (no sampler export)
    for r in rolling_records:
        assert "state_path" in r, f"Rolling record missing state_path: {r}"
        assert "sampler_path" not in r, f"Rolling record should not have sampler_path: {r}"

    # Periodic records should have both state_path and sampler_path
    for r in periodic_records:
        assert "state_path" in r, f"Periodic record missing state_path: {r}"
        assert "sampler_path" in r, f"Periodic record missing sampler_path: {r}"


@pytest.mark.integration
def test_rolling_checkpoint_resume():
    """Resume training from the checkpoints saved by test_rolling_checkpoint_train."""
    run_recipe(
        MODULE,
        [
            "behavior_if_log_dir_exists=resume",
            f"log_path={LOG_PATH}",
            "save_every=2",
            "rolling_save_every=1",
            "rolling_ttl_seconds=300",
        ],
        max_steps=6,
    )
