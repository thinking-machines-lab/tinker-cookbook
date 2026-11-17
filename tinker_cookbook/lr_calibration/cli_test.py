import dataclasses
import json
import math
import statistics
from pathlib import Path
from typing import cast

import pytest
import tinker

from tinker_cookbook import hyperparam_utils
from tinker_cookbook.lr_calibration import __main__ as cli
from tinker_cookbook.lr_calibration.__main__ import (
    MAX_ROUNDS,
    RUNTIME_FILE,
    RunCache,
    _check_runs_dir,
    _combine,
    _fit_or_extend,
    _initial_grid,
    _refinement_lrs,
    _regenerate,
    _round_multiplier,
    _write_record,
    calibrate_model,
    runtime_table,
)
from tinker_cookbook.lr_calibration.coverage import CALIBRATION_FILE
from tinker_cookbook.lr_calibration.fit import FitError, fit_curve
from tinker_cookbook.lr_calibration.sweep import RunResult, SweepConfig

CONFIG = SweepConfig(
    batch_size=8, steps=64, lora_rank=32, n_eval=64, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-8
)


def _result(lr: float, bpb: float | None) -> RunResult:
    return RunResult(lr=lr, diverged=bpb is None, eval_bpb=float("nan") if bpb is None else bpb)


def _record(opt_lr: float) -> dict:
    return {"multiplier": None, "opt_lr": opt_lr, "edge_lr": 3 * opt_lr}


def _table(path, models: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "reference_model": "ref",
                "reference_lr": 1e-3,
                "protocol": {"name": "p1"},
                "models": models,
            }
        )
    )


def test_run_cache_round_trip(tmp_path):
    cache = RunCache(tmp_path, "org/model:peft:1000")
    assert cache.get(1e-4) is None
    cache.put(RunResult(lr=1e-4, diverged=False, eval_bpb=0.5, train_bpb=[0.7, 0.6], tokens=10))
    cache.put(_result(2e-4, None))
    assert cache.get(1e-4) == RunResult(1e-4, False, 0.5, [0.7, 0.6], 10)
    diverged = cache.get(2e-4)
    assert diverged is not None and diverged.diverged
    assert cache.get(3e-4) is None
    # Each seed has its own runs.
    assert RunCache(tmp_path, "org/model:peft:1000", seed=1).get(1e-4) is None


def test_write_record_fills_multipliers_once_reference_exists(tmp_path):
    path = tmp_path / "table.json"
    _table(path, {})
    _write_record(path, None, "a", _record(2e-4))
    table = json.loads(path.read_text())
    assert table["models"]["a"]["multiplier"] is None

    _write_record(path, None, "ref", _record(4e-4))
    table = json.loads(path.read_text())
    assert table["models"]["a"]["multiplier"] == 0.5
    assert table["models"]["ref"]["multiplier"] == 1.0


def test_write_record_regenerates_the_runtime_table(tmp_path):
    path, runtime = tmp_path / "table.json", tmp_path / "runtime.json"
    _table(path, {})
    _write_record(path, runtime, "a", _record(2e-4))
    assert json.loads(runtime.read_text()) == {"reference_lr": 1e-3, "multipliers": {}}

    _write_record(path, runtime, "ref", _record(4e-4))
    assert json.loads(runtime.read_text()) == {
        "reference_lr": 1e-3,
        "multipliers": {"a": 0.5, "ref": 1.0},
    }


def test_write_record_refuses_a_runtime_table_that_disagrees(tmp_path):
    path, runtime = tmp_path / "table.json", tmp_path / "runtime.json"
    _table(path, {})
    _write_record(path, runtime, "ref", _record(4e-4))
    runtime.write_text(json.dumps({"reference_lr": 1e-3, "multipliers": {"ref": 2.0}}))
    with pytest.raises(SystemExit, match="does not match the calibration table"):
        _write_record(path, runtime, "a", _record(2e-4))
    assert "a" not in json.loads(path.read_text())["models"]


def test_regenerate_applies_hand_edits_to_both_tables(tmp_path):
    path, runtime = tmp_path / "table.json", tmp_path / "runtime.json"
    _table(path, {})
    _write_record(path, runtime, "ref", _record(4e-4))
    _write_record(path, runtime, "a", _record(2e-4))

    table = json.loads(path.read_text())
    table["reference_lr"] = 5e-4
    table["models"]["a"]["opt_lr"] = 1e-4
    path.write_text(json.dumps(table))
    _regenerate(path, runtime)

    assert json.loads(path.read_text())["models"]["a"]["multiplier"] == 0.25
    assert json.loads(runtime.read_text()) == {
        "reference_lr": 5e-4,
        "multipliers": {"a": 0.25, "ref": 1.0},
    }
    _write_record(path, runtime, "b", _record(8e-4))


def test_regenerate_rejects_same_source_and_runtime_file(tmp_path: Path):
    path = tmp_path / "table.json"
    _table(path, {"ref": _record(4e-4)})
    before = path.read_bytes()
    with pytest.raises(SystemExit, match="different paths"):
        _regenerate(path, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("alias", ["same", "relative", "symlink"])
def test_regenerate_cli_rejects_source_as_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
):
    path = tmp_path / "table.json"
    _table(path, {"ref": _record(4e-4)})
    before = path.read_bytes()
    if alias == "relative":
        nested = tmp_path / "nested"
        nested.mkdir()
        runtime = nested / ".." / path.name
    elif alias == "symlink":
        runtime = tmp_path / "runtime.json"
        runtime.symlink_to(path)
    else:
        runtime = path
    monkeypatch.setattr(
        "sys.argv",
        [
            "lr_calibration",
            "--regenerate",
            "--data-file",
            str(path),
            "--runtime-file",
            str(runtime),
        ],
    )
    with pytest.raises(SystemExit, match="different paths"):
        cli.main()
    assert path.read_bytes() == before
    if alias == "symlink":
        assert runtime.is_symlink()


@pytest.mark.parametrize("mode", ["--dry-run", "--model", "--missing"])
def test_regenerate_cli_rejects_calibration_modes_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
):
    path, runtime = tmp_path / "table.json", tmp_path / "runtime.json"
    _table(path, {"ref": _record(4e-4)})
    runtime.write_text('{"sentinel": true}')
    before = path.read_bytes(), runtime.read_bytes()
    args = [
        "lr_calibration",
        "--regenerate",
        "--data-file",
        str(path),
        "--runtime-file",
        str(runtime),
        mode,
    ]
    if mode == "--model":
        args.append("org/model")
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert (path.read_bytes(), runtime.read_bytes()) == before


def test_regenerate_cli_repairs_runtime_after_hand_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path, runtime = tmp_path / "table.json", tmp_path / "runtime.json"
    _table(path, {"ref": _record(4e-4), "a": _record(1e-4)})
    runtime.write_text('{"stale": true}')
    monkeypatch.setattr(
        "sys.argv",
        [
            "lr_calibration",
            "--regenerate",
            "--data-file",
            str(path),
            "--runtime-file",
            str(runtime),
        ],
    )
    cli.main()
    table = json.loads(path.read_text())
    assert table["protocol"] == {"name": "p1"}
    assert table["models"]["a"]["multiplier"] == 0.25
    assert json.loads(runtime.read_text()) == runtime_table(table)


def test_package_runtime_table_matches_calibration_table():
    table = json.loads(CALIBRATION_FILE.read_text())
    assert json.loads(RUNTIME_FILE.read_text()) == runtime_table(table)
    assert hyperparam_utils._lr_table() == runtime_table(table)


def test_package_multipliers_match_fits():
    table = json.loads(CALIBRATION_FILE.read_text())
    ref = table["models"][table["reference_model"]]["opt_lr"]
    for name, entry in table["models"].items():
        assert entry["multiplier"] == _round_multiplier(entry["opt_lr"] / ref), name
    assert table["models"][table["reference_model"]]["multiplier"] == 1.0


def test_package_fits_are_reproducible_from_the_recorded_sweeps():
    table = json.loads(CALIBRATION_FILE.read_text())
    for name, entry in table["models"].items():
        sweep = entry["sweep"]
        assert set(sweep["seeds"]) == {str(seed) for seed in table["protocol"]["seeds"]}, name
        by_seed = [dict(zip(s["lrs"], s["eval_bpb"])) for s in sweep["seeds"].values()]
        for lr, combined in zip(sweep["lrs"], sweep["eval_bpb"]):
            values = [seed[lr] for seed in by_seed if lr in seed]
            if combined is None:
                assert None in values, name
            else:
                assert combined == pytest.approx(statistics.median(values), abs=1e-6), name
        good = [(lr, b) for lr, b in zip(sweep["lrs"], sweep["eval_bpb"]) if b is not None]
        fit = fit_curve([g[0] for g in good], [g[1] for g in good])
        assert fit.opt_lr == pytest.approx(entry["opt_lr"], rel=2e-3), name
        assert fit.edge_lr == pytest.approx(entry["edge_lr"], rel=2e-3), name


def test_fit_or_extend_adds_points_where_the_curve_is_incomplete():
    grid = _initial_grid(4e-4)
    # Minimum at the lowest LR: extend downward.
    falling = {lr: _result(lr, 0.5 + 0.01 * i) for i, lr in enumerate(grid)}
    fit, todo = _fit_or_extend(falling)
    assert fit is None and todo == [grid[0] / 2]
    # Loss never rises 1% above the minimum: extend upward, toward the lowest diverged LR.
    flat = {lr: _result(lr, 0.5 - 0.0001 * min(i, 3)) for i, lr in enumerate(grid[:-1])}
    flat[grid[-1]] = _result(grid[-1], None)
    fit, todo = _fit_or_extend(flat)
    assert fit is None and todo == [pytest.approx((grid[-2] * grid[-1]) ** 0.5)]
    # Interior minimum and a clear edge: fit.
    bowl = dict(zip(grid, [0.53, 0.51, 0.502, 0.5, 0.501, 0.52, 0.6]))
    fit, todo = _fit_or_extend({lr: _result(lr, b) for lr, b in bowl.items()})
    assert fit is not None and todo == []
    assert grid[2] < fit.opt_lr < grid[4]


def test_fit_or_extend_treats_a_nonfinite_loss_as_diverged():
    grid = _initial_grid(4e-4)
    bowl = [0.53, 0.51, 0.502, 0.5, 0.501, 0.52, 0.6]
    results = {lr: _result(lr, b) for lr, b in zip(grid, bowl)}
    results[grid[-1]] = RunResult(lr=grid[-1], diverged=False, eval_bpb=math.nan)
    fit, todo = _fit_or_extend(results)
    assert fit is not None and todo == [] and math.isfinite(fit.opt_lr)


@pytest.mark.asyncio
async def test_calibrate_model_gives_up_after_max_rounds(monkeypatch, tmp_path):
    calls = []

    async def always_falling(service_client, model_name, lrs, windows, config, cache):
        calls.append(lrs)
        return [RunResult(lr=lr, diverged=False, eval_bpb=lr) for lr in lrs]

    monkeypatch.setattr(cli, "_sweep_cached", always_falling)
    with pytest.raises(FitError, match="no stable minimum"):
        await calibrate_model(
            cast(tinker.ServiceClient, None), "m", [], CONFIG, 4e-4, {0: RunCache(tmp_path, "m")}
        )
    assert len(calls) == MAX_ROUNDS


def _bowl(lr: float) -> float:
    """Loss with its minimum at 5e-4, a gentle low side and a steep high side."""
    z = math.log(lr / 5e-4)
    return 0.5 * (1 + 0.005 * math.exp(-0.8 * z) + 0.0016 * math.exp(2.5 * z))


def test_combine_takes_the_median_of_the_seeds_that_ran_each_lr():
    runs = {
        0: {1e-4: _result(1e-4, 0.5), 2e-4: _result(2e-4, 0.4), 4e-4: _result(4e-4, 0.6)},
        1: {2e-4: _result(2e-4, 0.5), 4e-4: _result(4e-4, None)},
        2: {2e-4: _result(2e-4, 0.9)},
    }
    curve = _combine(runs)
    assert list(curve) == [1e-4, 2e-4, 4e-4]
    # One bad run (0.9) does not move the curve.
    assert curve[1e-4].eval_bpb == 0.5 and curve[2e-4].eval_bpb == 0.5
    assert curve[4e-4].diverged


def test_refinement_lrs_complete_the_half_octave_window_for_every_seed():
    grid = _initial_grid(4e-4)
    runs: dict[int, dict[float, RunResult]] = {
        0: {lr: _result(lr, _bowl(lr)) for lr in grid},
        1: {},
    }
    below, above = pytest.approx(4e-4 / 2**0.5), pytest.approx(4e-4 * 2**0.5)
    todo = _refinement_lrs(runs)
    assert todo == {0: [below, above], 1: [2e-4, below, 4e-4, above, 8e-4]}
    for seed, lrs in todo.items():
        runs[seed].update({lr: _result(lr, _bowl(lr)) for lr in lrs})
    # 4e-4 * sqrt(2) is now the lowest, so the window moves up by half an octave.
    todo = _refinement_lrs(runs)
    assert todo == {0: [pytest.approx(4e-4 * 2**1.5)], 1: [pytest.approx(4e-4 * 2**1.5)]}
    # A diverged run counts as measured and is not requested again.
    for seed, lrs in todo.items():
        runs[seed].update({lr: _result(lr, None) for lr in lrs})
    assert _refinement_lrs(runs) == {}


@pytest.mark.asyncio
async def test_calibrate_model_refines_the_grid_around_the_minimum(monkeypatch, tmp_path):
    async def sweep(service_client, model_name, lrs, windows, config, cache):
        assert cache is caches[config.seed]
        if config.steps == 0:
            return [_result(lr, 0.6) for lr in lrs]
        return [_result(lr, _bowl(lr) + 1e-5 * config.seed) for lr in lrs]

    monkeypatch.setattr(cli, "_sweep_cached", sweep)
    caches = {seed: RunCache(tmp_path, "m", seed) for seed in (0, 1)}
    fit, runs, base_bpb = await calibrate_model(
        cast(tinker.ServiceClient, None), "m", [], CONFIG, 4e-4, caches
    )
    curve = _combine(runs)
    lowest = min(curve.values(), key=lambda r: r.eval_bpb).lr
    assert lowest == pytest.approx(4e-4 * 2**0.5)
    assert _refinement_lrs(runs) == {}
    # The first seed runs the whole grid; the other only the windows around the minimum.
    assert len(runs[0]) == 10 and sorted(runs[1]) == sorted(runs[0])[2:8]
    assert fit.opt_lr == pytest.approx(5e-4, rel=0.1)
    assert base_bpb == 0.6

    record = cli._record("m", fit, runs, base_bpb)["sweep"]
    assert record["lrs"] == sorted(runs[0]) and set(record["seeds"]) == {"0", "1"}
    assert record["eval_bpb"][3] == pytest.approx(_bowl(record["lrs"][3]) + 5e-6, abs=1e-6)


def test_multipliers_are_rounded_to_two_significant_figures():
    assert [_round_multiplier(v) for v in (0.4827, 1.0, 1.904, 0.0519)] == [0.48, 1.0, 1.9, 0.052]


def test_runs_dir_refuses_runs_from_another_setup(tmp_path):
    _check_runs_dir(tmp_path, "corpus", CONFIG)
    _check_runs_dir(tmp_path, "corpus", CONFIG)
    with pytest.raises(SystemExit, match="another --runs-dir"):
        _check_runs_dir(tmp_path, "corpus", dataclasses.replace(CONFIG, steps=128))
    with pytest.raises(SystemExit, match="another --runs-dir"):
        _check_runs_dir(tmp_path, "other-corpus", CONFIG)
    _check_runs_dir(tmp_path, "corpus", dataclasses.replace(CONFIG, seed=1))
