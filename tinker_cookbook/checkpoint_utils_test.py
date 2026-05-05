"""Tests for checkpoint_utils path handling."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tinker_cookbook.checkpoint_utils import (
    CheckpointManager,
    CheckpointRecord,
    get_last_checkpoint,
    load_checkpoints_file,
)


def _write_checkpoints_jsonl(log_dir: str, records: list[dict]) -> None:
    path = Path(log_dir) / "checkpoints.jsonl"
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_load_checkpoints_file_missing_dir():
    """load_checkpoints_file returns [] when the directory doesn't exist."""
    result = load_checkpoints_file("/tmp/nonexistent_dir_abc123")
    assert result == []


def test_load_checkpoints_file_missing_file():
    """load_checkpoints_file returns [] when checkpoints.jsonl is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_checkpoints_file(tmpdir)
        assert result == []


def test_load_checkpoints_file_reads_records():
    """load_checkpoints_file reads and deserializes checkpoint records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_checkpoints_jsonl(
            tmpdir,
            [
                {"name": "000005", "batch": 5, "state_path": "tinker://state/5"},
                {"name": "000010", "batch": 10, "state_path": "tinker://state/10"},
            ],
        )
        result = load_checkpoints_file(tmpdir)
        assert len(result) == 2
        assert isinstance(result[0], CheckpointRecord)
        assert result[0].name == "000005"
        assert result[1].batch == 10


def test_get_last_checkpoint_returns_last():
    """get_last_checkpoint returns the last record with the required key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_checkpoints_jsonl(
            tmpdir,
            [
                {"name": "000005", "batch": 5, "state_path": "tinker://state/5"},
                {"name": "000010", "batch": 10, "sampler_path": "tinker://sampler/10"},
                {"name": "000015", "batch": 15, "state_path": "tinker://state/15"},
            ],
        )
        result = get_last_checkpoint(tmpdir, required_key="state_path")
        assert result is not None
        assert result.name == "000015"


def test_get_last_checkpoint_returns_none_when_empty():
    """get_last_checkpoint returns None when no checkpoints exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = get_last_checkpoint(tmpdir)
        assert result is None


def test_get_last_checkpoint_returns_none_when_key_missing():
    """get_last_checkpoint returns None when no record has the required key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_checkpoints_jsonl(
            tmpdir,
            [{"name": "000005", "batch": 5, "sampler_path": "tinker://sampler/5"}],
        )
        result = get_last_checkpoint(tmpdir, required_key="state_path")
        assert result is None


def test_load_checkpoints_file_without_batch():
    """Entries without 'batch' should deserialize without error (backward compat)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_checkpoints_jsonl(
            tmpdir,
            [
                {"name": "000000", "step": 0},
                {"name": "000010", "step": 10, "state_path": "tinker://state/10"},
            ],
        )
        result = load_checkpoints_file(tmpdir)
        assert len(result) == 2
        assert result[0].batch is None
        assert result[0].extra["step"] == 0
        assert result[1].state_path == "tinker://state/10"


def test_checkpoint_record_extra_round_trips():
    """Unknown keys land in extra and survive to_dict/from_dict round-trip."""
    record = CheckpointRecord.from_dict(
        {"name": "000005", "batch": 5, "step": 5, "custom_key": "val"}
    )
    assert record.extra == {"step": 5, "custom_key": "val"}
    d = record.to_dict()
    assert d["step"] == 5
    assert d["custom_key"] == "val"
    restored = CheckpointRecord.from_dict(d)
    assert restored.extra == {"step": 5, "custom_key": "val"}


def test_checkpoint_record_name_only():
    """A minimal entry with only 'name' should deserialize (batch None)."""
    record = CheckpointRecord.from_dict({"name": "000000"})
    assert record.name == "000000"
    assert record.batch is None


def test_checkpoint_record_get_known_field():
    """get() returns known field values, including None for unset optional fields."""
    record = CheckpointRecord(name="test", batch=5, state_path="tinker://state/5")
    assert record.get("batch") == 5
    assert record.get("state_path") == "tinker://state/5"
    # Known fields always return the attribute value, even when None.
    # This distinguishes "field exists but is unset" from "key is unknown".
    assert record.get("epoch") is None
    assert record.get("epoch", -1) is None


def test_checkpoint_record_get_extra_field():
    """get() falls through to extra for unknown keys."""
    record = CheckpointRecord(name="test", extra={"step": 10, "custom": "val"})
    assert record.get("step") == 10
    assert record.get("custom") == "val"
    assert record.get("missing") is None
    assert record.get("missing", "default") == "default"


def test_checkpoint_record_has_extra_field():
    """has() works for both known fields and extra keys."""
    record = CheckpointRecord(name="test", batch=5, extra={"step": 10})
    assert record.has("batch")
    assert not record.has("epoch")
    assert record.has("step")
    assert not record.has("missing")


def test_checkpoint_record_extra_overlap_with_known_keys():
    """Known keys in extra are dropped defensively to prevent to_dict() conflicts."""
    record = CheckpointRecord(name="test", batch=5, extra={"batch": 99, "custom": "val"})
    # "batch" should be stripped from extra; the attribute value (5) wins
    assert record.batch == 5
    assert "batch" not in record.extra
    assert record.extra == {"custom": "val"}
    # to_dict() should have batch=5, not 99
    d = record.to_dict()
    assert d["batch"] == 5


# ---------------------------------------------------------------------------
# CheckpointManager tests
# ---------------------------------------------------------------------------


class TestCheckpointManagerShouldSave:
    """Tests for _should_save_rolling logic."""

    def _make_mgr(self, rolling_save_every: int = 1, save_every: int = 0) -> CheckpointManager:
        return CheckpointManager(
            training_client=MagicMock(),
            service_client=MagicMock(),
            log_path="/tmp/unused",
            rolling_save_every=rolling_save_every,
            save_every=save_every,
        )

    def test_disabled_when_zero(self):
        mgr = self._make_mgr(rolling_save_every=0)
        assert not mgr._should_save_rolling(1)
        assert not mgr._should_save_rolling(5)

    def test_skips_step_zero(self):
        mgr = self._make_mgr(rolling_save_every=1)
        assert not mgr._should_save_rolling(0)

    def test_saves_on_cadence(self):
        mgr = self._make_mgr(rolling_save_every=3)
        assert not mgr._should_save_rolling(1)
        assert not mgr._should_save_rolling(2)
        assert mgr._should_save_rolling(3)
        assert not mgr._should_save_rolling(4)
        assert not mgr._should_save_rolling(5)
        assert mgr._should_save_rolling(6)

    def test_saves_every_step(self):
        mgr = self._make_mgr(rolling_save_every=1)
        assert mgr._should_save_rolling(1)
        assert mgr._should_save_rolling(2)
        assert mgr._should_save_rolling(3)

    def test_skips_when_periodic_fires(self):
        mgr = self._make_mgr(rolling_save_every=1, save_every=5)
        assert mgr._should_save_rolling(1)
        assert mgr._should_save_rolling(4)
        assert not mgr._should_save_rolling(5)  # periodic fires here
        assert mgr._should_save_rolling(6)
        assert not mgr._should_save_rolling(10)  # periodic fires here

    def test_skips_when_periodic_fires_same_cadence(self):
        """Rolling and periodic on same cadence means rolling never fires."""
        mgr = self._make_mgr(rolling_save_every=5, save_every=5)
        assert not mgr._should_save_rolling(5)
        assert not mgr._should_save_rolling(10)


class TestCheckpointManagerShouldSavePeriodic:
    """Tests for should_save_periodic logic."""

    def _make_mgr(self, save_every: int = 5) -> CheckpointManager:
        return CheckpointManager(
            training_client=MagicMock(),
            service_client=MagicMock(),
            log_path="/tmp/unused",
            save_every=save_every,
        )

    def test_disabled_when_zero(self):
        mgr = self._make_mgr(save_every=0)
        assert not mgr.should_save_periodic(1)
        assert not mgr.should_save_periodic(5)

    def test_skips_step_zero(self):
        mgr = self._make_mgr(save_every=5)
        assert not mgr.should_save_periodic(0)

    def test_saves_on_cadence(self):
        mgr = self._make_mgr(save_every=5)
        assert not mgr.should_save_periodic(1)
        assert not mgr.should_save_periodic(4)
        assert mgr.should_save_periodic(5)
        assert not mgr.should_save_periodic(6)
        assert mgr.should_save_periodic(10)

    def test_saves_every_step(self):
        mgr = self._make_mgr(save_every=1)
        assert mgr.should_save_periodic(1)
        assert mgr.should_save_periodic(2)
        assert mgr.should_save_periodic(3)


def _make_save_state_mock(paths: list[str]) -> AsyncMock:
    """Create a save_state_async mock that returns different paths on each call."""
    call_idx = 0

    async def _save_state_async(name: str, ttl_seconds: int | None = None) -> MagicMock:
        nonlocal call_idx
        path = paths[call_idx % len(paths)]
        call_idx += 1
        future = MagicMock()
        result = MagicMock()
        result.path = path
        future.result_async = AsyncMock(return_value=result)
        return future

    return AsyncMock(side_effect=_save_state_async)


@pytest.mark.asyncio
async def test_maybe_save_async_saves_and_deletes():
    """Verify rolling checkpoint is saved with correct params, and old one is deleted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_training_client = MagicMock()
        mock_service_client = MagicMock()
        mock_rest_client = MagicMock()
        mock_service_client.create_rest_client.return_value = mock_rest_client
        mock_rest_client.delete_checkpoint_from_tinker_path_async = AsyncMock()

        mock_training_client.save_state_async = _make_save_state_mock(
            ["tinker://run/state/000001", "tinker://run/state/000002", "tinker://run/state/000003"]
        )

        mgr = CheckpointManager(
            training_client=mock_training_client,
            service_client=mock_service_client,
            log_path=tmpdir,
            rolling_save_every=1,
            save_every=10,
            rolling_ttl_seconds=3600,
        )

        # Step 1: fire rolling save
        await mgr.maybe_save_async(step=1, loop_state={"batch": 1})
        # Task is pending; resolve it by calling again at step 2
        await mgr.maybe_save_async(step=2, loop_state={"batch": 2})

        # After step 2, the step-1 save should have completed
        # and no delete (first rolling checkpoint, no prev)
        mock_rest_client.delete_checkpoint_from_tinker_path_async.assert_not_called()

        # Now resolve step 2 by calling step 3
        await mgr.maybe_save_async(step=3, loop_state={"batch": 3})

        # Step 2's save completed → delete step 1's checkpoint
        mock_rest_client.delete_checkpoint_from_tinker_path_async.assert_called_once_with(
            "tinker://run/state/000001"
        )

        # Verify checkpoints.jsonl was written
        ckpts = [
            CheckpointRecord.from_dict(json.loads(line))
            for line in (Path(tmpdir) / "checkpoints.jsonl").read_text().strip().split("\n")
        ]
        assert len(ckpts) >= 2
        assert all(c.extra.get("rolling") is True for c in ckpts)
        assert all(c.sampler_path is None for c in ckpts)  # no sampler export


@pytest.mark.asyncio
async def test_maybe_save_async_skips_when_disabled():
    """No save should happen when rolling_save_every=0."""
    mock_training_client = MagicMock()
    mock_training_client.save_state_async = AsyncMock()

    mgr = CheckpointManager(
        training_client=mock_training_client,
        service_client=MagicMock(),
        log_path="/tmp/unused",
        rolling_save_every=0,
    )

    await mgr.maybe_save_async(step=1, loop_state={"batch": 1})
    await mgr.maybe_save_async(step=2, loop_state={"batch": 2})
    mock_training_client.save_state_async.assert_not_called()


@pytest.mark.asyncio
async def test_finalize_deletes_last_rolling_checkpoint():
    """finalize_async should delete the last rolling checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_training_client = MagicMock()
        mock_service_client = MagicMock()
        mock_rest_client = MagicMock()
        mock_service_client.create_rest_client.return_value = mock_rest_client
        mock_rest_client.delete_checkpoint_from_tinker_path_async = AsyncMock()

        mock_training_client.save_state_async = _make_save_state_mock(["tinker://run/state/000001"])

        mgr = CheckpointManager(
            training_client=mock_training_client,
            service_client=mock_service_client,
            log_path=tmpdir,
            rolling_save_every=1,
        )

        await mgr.maybe_save_async(step=1, loop_state={"batch": 1})
        await mgr.finalize_async()

        # Should delete the step-1 checkpoint (the only/last one)
        mock_rest_client.delete_checkpoint_from_tinker_path_async.assert_called_once_with(
            "tinker://run/state/000001"
        )
        # After finalize, _prev_state_path should be None
        assert mgr._prev_state_path is None


@pytest.mark.asyncio
async def test_save_failure_is_swallowed():
    """A failed rolling save should be logged but not raise."""
    mock_training_client = MagicMock()
    mock_training_client.save_state_async = AsyncMock(side_effect=RuntimeError("server error"))

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(
            training_client=mock_training_client,
            service_client=MagicMock(),
            log_path=tmpdir,
            rolling_save_every=1,
        )

        # Should not raise
        await mgr.maybe_save_async(step=1, loop_state={"batch": 1})
        # Resolve the failed task
        await mgr.maybe_save_async(step=2, loop_state={"batch": 2})


@pytest.mark.asyncio
async def test_delete_failure_is_swallowed():
    """A failed delete should be logged but not raise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_training_client = MagicMock()
        mock_service_client = MagicMock()
        mock_rest_client = MagicMock()
        mock_service_client.create_rest_client.return_value = mock_rest_client
        mock_rest_client.delete_checkpoint_from_tinker_path_async = AsyncMock(
            side_effect=RuntimeError("delete failed")
        )

        mock_training_client.save_state_async = _make_save_state_mock(
            ["tinker://run/state/000001", "tinker://run/state/000002", "tinker://run/state/000003"]
        )

        mgr = CheckpointManager(
            training_client=mock_training_client,
            service_client=mock_service_client,
            log_path=tmpdir,
            rolling_save_every=1,
        )

        # Save at step 1
        await mgr.maybe_save_async(step=1, loop_state={"batch": 1})
        # Resolve step 1, save at step 2 — delete of step 1 will fail
        await mgr.maybe_save_async(step=2, loop_state={"batch": 2})
        # Resolve step 2 (triggers delete of step 1, which fails)
        await mgr.maybe_save_async(step=3, loop_state={"batch": 3})

        # Should not raise — delete failure is swallowed
        assert mock_rest_client.delete_checkpoint_from_tinker_path_async.called


# ---------------------------------------------------------------------------
# Async periodic saves tests
# ---------------------------------------------------------------------------


def _make_both_save_mocks(
    state_paths: list[str], sampler_paths: list[str]
) -> tuple[AsyncMock, AsyncMock]:
    """Create mocks for save_state_async and save_weights_for_sampler_async."""
    state_mock = _make_save_state_mock(state_paths)

    sampler_idx = 0

    async def _save_sampler_async(name: str, ttl_seconds: int | None = None) -> MagicMock:
        nonlocal sampler_idx
        path = sampler_paths[sampler_idx % len(sampler_paths)]
        sampler_idx += 1
        future = MagicMock()
        result = MagicMock()
        result.path = path
        future.result_async = AsyncMock(return_value=result)
        return future

    return state_mock, AsyncMock(side_effect=_save_sampler_async)


@pytest.mark.asyncio
async def test_async_periodic_saves_fire_and_forget():
    """Periodic saves should run as background tasks when async_periodic_saves=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_training_client = MagicMock()
        state_mock, sampler_mock = _make_both_save_mocks(
            ["tinker://run/state/000005"], ["tinker://run/sampler/000005"]
        )
        mock_training_client.save_state_async = state_mock
        mock_training_client.save_weights_for_sampler_async = sampler_mock

        mgr = CheckpointManager(
            training_client=mock_training_client,
            service_client=MagicMock(),
            log_path=tmpdir,
            save_every=5,
            async_periodic_saves=True,
        )

        # Steps 1-4: no periodic save
        for step in range(1, 5):
            result = await mgr.maybe_save_async(step=step, loop_state={"batch": step})
            assert result is None
            mock_training_client.save_state_async.assert_not_called()

        # Step 5: triggers periodic save, but async — returns None immediately
        result = await mgr.maybe_save_async(step=5, loop_state={"batch": 5})
        assert result is None  # async mode returns None
        assert mgr._pending_periodic_task is not None

        # Resolve by calling finalize
        await mgr.finalize_async()
        assert mgr._pending_periodic_task is None

        # Verify checkpoints.jsonl was written with both state and sampler paths
        ckpts = [
            CheckpointRecord.from_dict(json.loads(line))
            for line in (Path(tmpdir) / "checkpoints.jsonl").read_text().strip().split("\n")
        ]
        assert len(ckpts) == 1
        assert ckpts[0].state_path == "tinker://run/state/000005"
        assert ckpts[0].sampler_path == "tinker://run/sampler/000005"


@pytest.mark.asyncio
async def test_async_periodic_saves_resolves_before_next():
    """A pending async periodic save should resolve before the next one fires."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_training_client = MagicMock()
        state_mock, sampler_mock = _make_both_save_mocks(
            ["tinker://run/state/000005", "tinker://run/state/000010"],
            ["tinker://run/sampler/000005", "tinker://run/sampler/000010"],
        )
        mock_training_client.save_state_async = state_mock
        mock_training_client.save_weights_for_sampler_async = sampler_mock

        mgr = CheckpointManager(
            training_client=mock_training_client,
            service_client=MagicMock(),
            log_path=tmpdir,
            save_every=5,
            async_periodic_saves=True,
        )

        # Step 5: fires first periodic save
        await mgr.maybe_save_async(step=5, loop_state={"batch": 5})
        assert mgr._pending_periodic_task is not None

        # Step 10: should resolve step 5 first, then fire step 10
        await mgr.maybe_save_async(step=10, loop_state={"batch": 10})
        # step 5 resolved, step 10 is now pending
        assert mgr._pending_periodic_task is not None

        await mgr.finalize_async()

        ckpts = [
            CheckpointRecord.from_dict(json.loads(line))
            for line in (Path(tmpdir) / "checkpoints.jsonl").read_text().strip().split("\n")
        ]
        assert len(ckpts) == 2
        assert ckpts[0].name == "000005"
        assert ckpts[1].name == "000010"


@pytest.mark.asyncio
async def test_async_periodic_save_final_ordering():
    """Final checkpoint record should always appear after any pending periodic save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_training_client = MagicMock()
        state_mock, sampler_mock = _make_both_save_mocks(
            ["tinker://run/state/000005", "tinker://run/state/final"],
            ["tinker://run/sampler/000005", "tinker://run/sampler/final"],
        )
        mock_training_client.save_state_async = state_mock
        mock_training_client.save_weights_for_sampler_async = sampler_mock

        mgr = CheckpointManager(
            training_client=mock_training_client,
            service_client=MagicMock(),
            log_path=tmpdir,
            save_every=5,
            async_periodic_saves=True,
        )

        # Fire a background periodic save at step 5
        await mgr.maybe_save_async(step=5, loop_state={"batch": 5})
        assert mgr._pending_periodic_task is not None

        # save_final_async should drain the pending periodic save first
        await mgr.save_final_async(loop_state={"epoch": 1, "batch": 0})

        ckpts = [
            CheckpointRecord.from_dict(json.loads(line))
            for line in (Path(tmpdir) / "checkpoints.jsonl").read_text().strip().split("\n")
        ]
        assert len(ckpts) == 2
        assert ckpts[0].name == "000005"
        assert ckpts[1].name == "final"


@pytest.mark.asyncio
async def test_async_periodic_save_failure_is_swallowed():
    """A failed async periodic save should be logged but not raise."""
    mock_training_client = MagicMock()
    mock_training_client.save_state_async = AsyncMock(side_effect=RuntimeError("server error"))

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(
            training_client=mock_training_client,
            service_client=MagicMock(),
            log_path=tmpdir,
            save_every=5,
            async_periodic_saves=True,
        )

        # Step 5: fires periodic save that will fail
        await mgr.maybe_save_async(step=5, loop_state={"batch": 5})

        # Finalize should not raise
        await mgr.finalize_async()


@pytest.mark.asyncio
async def test_sync_periodic_saves_still_work():
    """With async_periodic_saves=False (default), saves block and return paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_training_client = MagicMock()
        state_mock, sampler_mock = _make_both_save_mocks(
            ["tinker://run/state/000005"], ["tinker://run/sampler/000005"]
        )
        mock_training_client.save_state_async = state_mock
        mock_training_client.save_weights_for_sampler_async = sampler_mock

        mgr = CheckpointManager(
            training_client=mock_training_client,
            service_client=MagicMock(),
            log_path=tmpdir,
            save_every=5,
            async_periodic_saves=False,
        )

        result = await mgr.maybe_save_async(step=5, loop_state={"batch": 5})
        assert result is not None
        assert result["state_path"] == "tinker://run/state/000005"
        assert result["sampler_path"] == "tinker://run/sampler/000005"
        assert mgr._pending_periodic_task is None
