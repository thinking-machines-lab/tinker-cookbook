from __future__ import annotations

from typing import cast

import pytest

from tinker_cookbook.preflight import (
    PreflightConfig,
    capture_preflight_snapshot,
    main,
    validate_training_run,
)
from tinker_cookbook.stores.storage import storage_from_uri
from tinker_cookbook.stores.training_store import TrainingRunStore

FINAL_STATE_CHECKPOINT: dict[str, object] = {
    "name": "final",
    "final": True,
    "state_path": "tinker://run/state/final",
}


def _store(path: str) -> TrainingRunStore:
    return TrainingRunStore(storage_from_uri(path))


def _write_metric_run(
    path: str, metrics: dict[str, object], *, step: int | None = None
) -> TrainingRunStore:
    store = _store(path)
    store.write_metrics(metrics, step=step)
    store.write_checkpoint(FINAL_STATE_CHECKPOINT)
    return store


def _write_passing_run(path: str) -> None:
    store = _store(path)
    store.write_metrics({"train_mean_nll": 1.25, "nested": {"grad_norm": 0.5}}, step=0)
    store.write_checkpoint(
        {
            "name": "final",
            "final": True,
            "state_path": "tinker://run/state/final",
            "sampler_path": "tinker://run/sampler_weights/final",
        }
    )


def test_validate_training_run_passes_with_declared_evidence(tmp_path) -> None:
    _write_passing_run(str(tmp_path))

    report = validate_training_run(
        PreflightConfig(
            log_path=str(tmp_path),
            required_metric_keys=("train_mean_nll",),
            require_sampler_checkpoint=True,
        )
    )

    assert report.passed
    assert [check.name for check in report.checks] == [
        "Artifact format",
        "Training metrics",
        "Training-state checkpoint",
        "Sampler checkpoint",
    ]


def test_validate_training_run_fails_when_metrics_are_missing(tmp_path) -> None:
    _store(str(tmp_path)).write_checkpoint(FINAL_STATE_CHECKPOINT)

    report = validate_training_run(PreflightConfig(log_path=str(tmp_path)))

    assert not report.passed
    assert "No metric records" in report.failure_summary()


def test_validate_training_run_fails_on_non_finite_nested_metric(tmp_path) -> None:
    _write_metric_run(
        str(tmp_path),
        {"train_mean_nll": 1.0, "optimizer": {"grad_norm": float("nan")}},
    )

    report = validate_training_run(PreflightConfig(log_path=str(tmp_path)))

    assert not report.passed
    assert "record[0].optimizer.grad_norm" in report.failure_summary()


def test_validate_training_run_fails_on_malformed_artifact_line(tmp_path) -> None:
    store = _store(str(tmp_path))
    store.storage.write(
        "metrics.jsonl",
        b'{"step": 0, "train_mean_nll": 1.0}\n{"step": broken}\n',
    )
    store.write_checkpoint(FINAL_STATE_CHECKPOINT)

    report = validate_training_run(PreflightConfig(log_path=str(tmp_path)))

    assert not report.passed
    assert "metrics.jsonl:2" in report.failure_summary()


def test_validate_training_run_fails_when_required_metric_is_missing(tmp_path) -> None:
    _write_passing_run(str(tmp_path))

    report = validate_training_run(
        PreflightConfig(log_path=str(tmp_path), required_metric_keys=("reward",))
    )

    assert not report.passed
    assert "missing required keys: reward" in report.failure_summary()


@pytest.mark.parametrize("invalid_value", ["not-a-number", True, None])
def test_validate_training_run_rejects_non_numeric_required_metrics(
    tmp_path, invalid_value: object
) -> None:
    _write_metric_run(str(tmp_path), {"train_mean_nll": invalid_value}, step=0)

    report = validate_training_run(
        PreflightConfig(log_path=str(tmp_path), required_metric_keys=("train_mean_nll",))
    )

    assert not report.passed
    assert "required metric values are not numeric" in report.failure_summary()


def test_validate_training_run_accepts_integer_required_metric(tmp_path) -> None:
    _write_metric_run(str(tmp_path), {"reward": 1}, step=0)

    report = validate_training_run(
        PreflightConfig(log_path=str(tmp_path), required_metric_keys=("reward",))
    )

    assert report.passed


def test_validate_training_run_can_require_metric_progress(tmp_path) -> None:
    _write_passing_run(str(tmp_path))

    stale = validate_training_run(PreflightConfig(log_path=str(tmp_path), minimum_metric_step=1))
    _store(str(tmp_path)).write_metrics({"train_mean_nll": 1.0}, step=1)
    current = validate_training_run(PreflightConfig(log_path=str(tmp_path), minimum_metric_step=1))

    assert not stale.passed
    assert "minimum step 1" in stale.failure_summary()
    assert current.passed


def test_required_metric_must_appear_at_or_after_minimum_step(tmp_path) -> None:
    store = _store(str(tmp_path))
    store.write_metrics({"train_mean_nll": 1.0}, step=0)
    store.write_metrics({"time_total": 2.0}, step=3)
    store.write_checkpoint(FINAL_STATE_CHECKPOINT)

    report = validate_training_run(
        PreflightConfig(
            log_path=str(tmp_path),
            required_metric_keys=("train_mean_nll",),
            minimum_metric_step=3,
        )
    )

    assert not report.passed
    assert "missing required keys: train_mean_nll" in report.failure_summary()


def test_snapshot_requires_metrics_and_checkpoints_appended_after_capture(tmp_path) -> None:
    _write_passing_run(str(tmp_path))
    snapshot = capture_preflight_snapshot(str(tmp_path))
    store = _store(str(tmp_path))
    store.write_metrics({"train_mean_nll": 1.0}, step=3)

    stale_checkpoint = validate_training_run(
        PreflightConfig(
            log_path=str(tmp_path),
            required_metric_keys=("train_mean_nll",),
            require_sampler_checkpoint=True,
            minimum_metric_step=3,
        ),
        after=snapshot,
    )

    store.write_checkpoint(
        {
            "name": "final",
            "final": True,
            "state_path": "tinker://run/state/resumed-final",
            "sampler_path": "tinker://run/sampler_weights/resumed-final",
        }
    )
    current_evidence = validate_training_run(
        PreflightConfig(
            log_path=str(tmp_path),
            required_metric_keys=("train_mean_nll",),
            require_sampler_checkpoint=True,
            minimum_metric_step=3,
        ),
        after=snapshot,
    )

    assert not stale_checkpoint.passed
    assert "No final checkpoint" in stale_checkpoint.failure_summary()
    assert current_evidence.passed


def test_snapshot_rejects_truncated_artifact_history(tmp_path) -> None:
    _write_passing_run(str(tmp_path))
    snapshot = capture_preflight_snapshot(str(tmp_path))
    storage = storage_from_uri(str(tmp_path))
    storage.write("metrics.jsonl", b"")

    report = validate_training_run(PreflightConfig(log_path=str(tmp_path)), after=snapshot)

    assert not report.passed
    assert "artifact was truncated" in report.failure_summary()


def test_snapshot_rejects_rewritten_and_regrown_artifact_history(tmp_path) -> None:
    _write_passing_run(str(tmp_path))
    snapshot = capture_preflight_snapshot(str(tmp_path))
    storage = storage_from_uri(str(tmp_path))
    old_metrics = storage.read("metrics.jsonl")
    old_checkpoints = storage.read("checkpoints.jsonl")
    storage.write("metrics.jsonl", b'{"step": 0, "replacement": 1}\n' + old_metrics)
    storage.write(
        "checkpoints.jsonl",
        b'{"name": "replacement", "state_path": "tinker://replacement"}\n' + old_checkpoints,
    )

    report = validate_training_run(PreflightConfig(log_path=str(tmp_path)), after=snapshot)

    assert not report.passed
    assert "artifact changed before the captured append boundary" in report.failure_summary()


def test_snapshot_rejects_a_different_log_path(tmp_path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    snapshot = capture_preflight_snapshot(str(first_path))

    with pytest.raises(ValueError, match="snapshot log_path"):
        validate_training_run(PreflightConfig(log_path=str(second_path)), after=snapshot)


@pytest.mark.parametrize("invalid_step", [-1, True, "1"])
def test_preflight_config_rejects_invalid_minimum_step(tmp_path, invalid_step: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        PreflightConfig(log_path=str(tmp_path), minimum_metric_step=cast(int, invalid_step))


def test_validate_training_run_can_require_sampler_checkpoint(tmp_path) -> None:
    _write_metric_run(str(tmp_path), {"train_mean_nll": 1.0})

    report = validate_training_run(
        PreflightConfig(log_path=str(tmp_path), require_sampler_checkpoint=True)
    )

    assert not report.passed
    assert "sampler_path" in report.failure_summary()


@pytest.mark.parametrize("invalid_path", ["", "https://example.com/checkpoint", 123])
def test_validate_training_run_rejects_invalid_checkpoint_uris(
    tmp_path, invalid_path: object
) -> None:
    store = _store(str(tmp_path))
    store.write_metrics({"train_mean_nll": 1.0})
    store.write_checkpoint({"name": "final", "final": True, "state_path": invalid_path})

    report = validate_training_run(PreflightConfig(log_path=str(tmp_path)))

    assert not report.passed
    assert "invalid state_path" in report.failure_summary()


def test_validate_training_run_rejects_invalid_checkpoint_name(tmp_path) -> None:
    store = _store(str(tmp_path))
    store.write_metrics({"train_mean_nll": 1.0})
    store.write_checkpoint({"name": "", "final": True, "state_path": "tinker://run/state/final"})

    report = validate_training_run(PreflightConfig(log_path=str(tmp_path)))

    assert not report.passed
    assert "invalid name" in report.failure_summary()


def test_checkpoint_validation_does_not_claim_server_lookup(tmp_path) -> None:
    store = _store(str(tmp_path))
    store.write_metrics({"train_mean_nll": 1.0})
    store.write_checkpoint(
        {"name": "final", "final": True, "state_path": "tinker://not-server-validated"}
    )

    report = validate_training_run(PreflightConfig(log_path=str(tmp_path)))

    assert report.passed
    checkpoint_check = next(
        check for check in report.checks if check.name == "Training-state checkpoint"
    )
    assert "prefixed string" in checkpoint_check.summary


def test_validate_training_run_rejects_periodic_checkpoint_when_final_is_required(
    tmp_path,
) -> None:
    store = _store(str(tmp_path))
    store.write_metrics({"train_mean_nll": 1.0})
    store.write_checkpoint(
        {"name": "step-1", "final": False, "state_path": "tinker://run/state/step-1"}
    )

    strict = validate_training_run(PreflightConfig(log_path=str(tmp_path)))
    permissive = validate_training_run(
        PreflightConfig(log_path=str(tmp_path), require_final_checkpoint=False)
    )

    assert not strict.passed
    assert permissive.passed


def test_validate_training_run_accepts_legacy_final_checkpoint_name(tmp_path) -> None:
    store = _store(str(tmp_path))
    store.write_metrics({"train_mean_nll": 1.0})
    store.write_checkpoint({"name": "final", "state_path": "tinker://run/state/final"})

    report = validate_training_run(PreflightConfig(log_path=str(tmp_path)))

    assert report.passed


def test_main_returns_nonzero_for_blocked_run(tmp_path, capsys) -> None:
    exit_code = main([str(tmp_path), "--metric", "train_mean_nll"])

    assert exit_code == 1
    assert "TINKER PREFLIGHT: FAIL" in capsys.readouterr().out


def test_main_reports_invalid_minimum_step_as_usage_error(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path), "--minimum-step", "-1"])

    assert error.value.code == 2
    assert "must be a nonnegative integer" in capsys.readouterr().err
