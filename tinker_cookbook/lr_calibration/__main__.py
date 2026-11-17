"""Calibrate per-model LR multipliers and keep the LR tables up to date.

Usage::

    # Calibrate one model and write its entry
    python -m tinker_cookbook.lr_calibration --model Qwen/Qwen3.5-9B

    # Calibrate every trainable Tinker model that has no entry yet
    python -m tinker_cookbook.lr_calibration --missing

    # Only list the models that would be calibrated
    python -m tinker_cookbook.lr_calibration --missing --dry-run

    # After editing calibration_data.json by hand, rewrite the multipliers and lr_multipliers.json
    python -m tinker_cookbook.lr_calibration --regenerate

Results go to two files. ``calibration_data.json``, next to this module, holds
everything: the protocol, the reference LR and how it was chosen, and each
model's sweep and fit. ``tinker_cookbook/lr_multipliers.json``,
which ``hyperparam_utils`` reads, holds only the reference LR and the published
multipliers. It is regenerated from the calibration table on every write, so
edit the calibration table rather than the runtime one, then run ``--regenerate``;
otherwise the command refuses to run while the two disagree.

A sweep starts with one run per learning rate on a coarse grid. The learning
rates around the optimum are then run once per seed in the protocol's
``seeds``, and the optimum is fitted to their median (see ``fit.py``).

Each finished training run is saved under ``--runs-dir``, so rerunning an
interrupted command only pays for the runs that did not finish. Each model's
entry is written as soon as its sweep is fitted. ``--data-file`` points the
command at another calibration table with the same layout; its ``protocol``
block sets the training setup and the initial LR grid, and no runtime table is
written unless ``--runtime-file`` is given. Calibrate several models in one
process (repeat ``--model``) rather than running several processes on the same
table.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import json
import logging
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import tinker

from tinker_cookbook import cli_utils, hyperparam_utils
from tinker_cookbook.lr_calibration.coverage import (
    CALIBRATION_FILE,
    base_model_name,
    missing_models,
    one_id_per_base_model,
)
from tinker_cookbook.lr_calibration.data import load_corpus
from tinker_cookbook.lr_calibration.fit import (
    EDGE_TOLERANCE,
    CurveFit,
    FitError,
    fit_curve,
    window_centres,
)
from tinker_cookbook.lr_calibration.sweep import RunResult, SweepConfig, run_sweep
from tinker_cookbook.utils.misc_utils import safezip

logger = logging.getLogger(__name__)

RUNTIME_FILE = Path(hyperparam_utils.__file__).with_name("lr_multipliers.json")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tinker-cookbook" / "lr_calibration"
MAX_ROUNDS = 10
"""Rounds of extending the grid or refining it around the minimum before giving up."""
BASE_LR = 0.0
"""Cache key for the zero-step run that measures the base model."""


def _protocol_config(table: dict[str, Any]) -> tuple[str, SweepConfig]:
    p = table["protocol"]
    config = SweepConfig(
        batch_size=p["batch_size"],
        steps=p["steps"],
        lora_rank=p["lora_rank"],
        n_eval=p["n_eval"],
        adam_beta1=p["adam"]["beta1"],
        adam_beta2=p["adam"]["beta2"],
        adam_eps=p["adam"]["eps"],
    )
    return p["corpus"], config


def _initial_grid(center: float) -> list[float]:
    return [center * 2.0**k for k in range(-3, 4)]


def _fit_or_extend(results: dict[float, RunResult]) -> tuple[CurveFit | None, list[float]]:
    """Fit the curve, or return the learning rates needed to make it fittable."""
    failed = {lr for lr, r in results.items() if r.diverged or not math.isfinite(r.eval_bpb)}
    good = sorted((lr, r.eval_bpb) for lr, r in results.items() if lr not in failed)
    diverged = sorted(failed)
    lowest = min(results)

    def one_higher() -> list[float]:
        top = good[-1][0]
        if diverged and diverged[0] > top:
            return [math.sqrt(top * diverged[0])]
        return [top * 2]

    if len(good) < 3:
        return None, [lowest / 4, lowest / 2]
    losses = [g[1] for g in good]
    i = losses.index(min(losses))
    if i < 2:
        return None, [lowest / 2]
    if i > len(good) - 3 or max(losses[i + 1 :]) <= losses[i] * (1 + EDGE_TOLERANCE):
        return None, one_higher()
    return fit_curve([g[0] for g in good], losses), []


def _combine(runs: dict[int, dict[float, RunResult]]) -> dict[float, RunResult]:
    """One curve from every seed's runs: at each learning rate, the median loss of the seeds
    that ran it, so that one bad run does not move the curve. A learning rate counts as diverged
    if any seed diverged there."""
    curve = {}
    for lr in sorted({lr for results in runs.values() for lr in results}):
        at_lr = [results[lr] for results in runs.values() if lr in results]
        failed = any(r.diverged or not math.isfinite(r.eval_bpb) for r in at_lr)
        loss = math.nan if failed else statistics.median(r.eval_bpb for r in at_lr)
        curve[lr] = RunResult(lr=lr, diverged=failed, eval_bpb=loss)
    return curve


def _refinement_lrs(runs: dict[int, dict[float, RunResult]]) -> dict[int, list[float]]:
    """Per seed, the learning rates still to run so that every centre the fit puts its window on
    (the lowest point of the combined curve first), and two neighbours a factor of sqrt(2) apart
    on each side of it, are measured with every seed."""
    curve = {lr: r.eval_bpb for lr, r in _combine(runs).items() if not r.diverged}
    centres = window_centres(list(curve), list(curve.values()))

    measured = sorted({lr for results in runs.values() for lr in results})

    def known(lr: float) -> float:
        return next((have for have in measured if math.isclose(lr, have, rel_tol=1e-6)), lr)

    wanted = sorted({known(centre * 2 ** (k / 2)) for centre in centres for k in range(-2, 3)})
    todo = {seed: [lr for lr in wanted if lr not in results] for seed, results in runs.items()}
    return {seed: lrs for seed, lrs in todo.items() if lrs}


class RunCache:
    """Finished runs of one model and seed, one JSON file per learning rate."""

    def __init__(self, runs_dir: Path, model_name: str, seed: int = 0):
        self.dir = runs_dir / model_name.replace("/", "__")
        # Seed 0 stays at the top level so caches written before multi-seed runs remain valid.
        if seed != 0:
            self.dir = self.dir / f"seed{seed}"

    def _path(self, lr: float) -> Path:
        return self.dir / f"lr{lr:.6e}.json"

    def get(self, lr: float) -> RunResult | None:
        path = self._path(lr)
        return RunResult(**json.loads(path.read_text())) if path.exists() else None

    def put(self, result: RunResult) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self._path(result.lr), json.dumps(dataclasses.asdict(result)))


async def _sweep_cached(
    service_client: tinker.ServiceClient,
    model_name: str,
    lrs: list[float],
    windows: list[str],
    config: SweepConfig,
    cache: RunCache,
) -> list[RunResult]:
    cached = {lr: r for lr in lrs if (r := cache.get(lr)) is not None}
    todo = [lr for lr in lrs if lr not in cached]
    if todo:
        for r in await run_sweep(
            service_client, model_name, todo, windows, config, on_result=cache.put
        ):
            cached[r.lr] = r
    return [cached[lr] for lr in lrs]


async def calibrate_model(
    service_client: tinker.ServiceClient,
    model_name: str,
    windows: list[str],
    config: SweepConfig,
    center: float,
    caches: dict[int, RunCache],
) -> tuple[CurveFit, dict[int, dict[float, RunResult]], float]:
    """Sweep with an adaptive grid until the curve has an interior minimum and a clear edge,
    and the learning rates around the minimum are measured with every seed.

    Args:
        service_client (tinker.ServiceClient): Client to create training clients from.
        model_name (str): Tinker model id to calibrate.
        windows (list[str]): Corpus windows.
        config (SweepConfig): Shared training setup; its seed is replaced by each seed in turn.
        center (float): Middle of the initial learning-rate grid.
        caches (dict[int, RunCache]): Run cache of every seed. The first seed runs the
            initial grid and its extensions.

    Returns:
        The fit of the combined curve, every run's result by seed and learning rate, and the
        base model's held-out bits per byte.
    """
    first = next(iter(caches))
    runs: dict[int, dict[float, RunResult]] = {seed: {} for seed in caches}

    async def run(todo: dict[int, list[float]]) -> None:
        batches = await asyncio.gather(
            *[
                _sweep_cached(
                    service_client,
                    model_name,
                    lrs,
                    windows,
                    dataclasses.replace(config, seed=seed),
                    caches[seed],
                )
                for seed, lrs in todo.items()
            ]
        )
        # Keyed by the requested LR: a cached run can hold the same LR with other rounding.
        for (seed, lrs), batch in safezip(todo.items(), batches):
            runs[seed].update(safezip(lrs, batch))

    todo = {first: _initial_grid(center)}
    fit: CurveFit | None = None
    for _ in range(MAX_ROUNDS):
        await run(todo)
        fit, extension = _fit_or_extend(_combine(runs))
        # A new point can become the lowest, which moves the window; stop when it is complete.
        todo = {first: extension} if fit is None else _refinement_lrs(runs)
        if not todo:
            break
    if fit is None or todo:
        raise FitError(
            f"{model_name}: no stable minimum after {MAX_ROUNDS} rounds of extending and "
            "refining the grid"
        )
    zero_steps = dataclasses.replace(config, steps=0, seed=first)
    (base,) = await _sweep_cached(
        service_client, model_name, [BASE_LR], windows, zero_steps, caches[first]
    )
    return fit, runs, base.eval_bpb


def _bpb(result: RunResult) -> float | None:
    return None if result.diverged else round(result.eval_bpb, 6)


def _record(
    model_name: str, fit: CurveFit, runs: dict[int, dict[float, RunResult]], base_bpb: float
) -> dict[str, Any]:
    curve = _combine(runs)
    return {
        "multiplier": None,
        "edge_lr": fit.edge_lr,
        "opt_lr": fit.opt_lr,
        "date": datetime.date.today().isoformat(),
        "tinker_model_id": model_name,
        "sweep": {
            "lrs": list(curve),
            "eval_bpb": [_bpb(r) for r in curve.values()],
            "base_eval_bpb": round(base_bpb, 6),
            "seeds": {
                str(seed): {
                    "lrs": sorted(results),
                    "eval_bpb": [_bpb(results[lr]) for lr in sorted(results)],
                }
                for seed, results in runs.items()
            },
        },
    }


def _round_multiplier(value: float) -> float:
    """Use two significant figures to avoid implying more precision than the sweep supports."""
    return float(f"{value:.2g}")


def _renormalize(table: dict[str, Any]) -> None:
    """Set every multiplier to opt_lr / opt_lr(reference model), rounded to two significant
    figures. Multipliers stay ``None`` until the reference model is calibrated."""
    ref = table["models"].get(table["reference_model"])
    for entry in table["models"].values():
        entry["multiplier"] = _round_multiplier(entry["opt_lr"] / ref["opt_lr"]) if ref else None


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _check_runs_dir(runs_dir: Path, corpus: str, config: SweepConfig) -> None:
    """Refuse to reuse cached runs that were trained with a different setup."""
    # Runs of every seed share the directory, so the seed is not part of the setup.
    setup = {"corpus": corpus, **dataclasses.asdict(config)}
    del setup["seed"]
    path = runs_dir / "setup.json"
    if path.exists():
        cached = json.loads(path.read_text())
        if cached != setup:
            raise SystemExit(
                f"{runs_dir} holds runs trained with {cached}, but the protocol now asks for "
                f"{setup}. Pass another --runs-dir or rename the protocol."
            )
        return
    runs_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(setup, indent=2))


def runtime_table(table: dict[str, Any]) -> dict[str, Any]:
    """The part of a calibration table that ``hyperparam_utils`` reads: the reference LR and
    every published multiplier."""
    return {
        "reference_lr": table["reference_lr"],
        "multipliers": {
            name: entry["multiplier"]
            for name, entry in sorted(table["models"].items())
            if entry["multiplier"] is not None
        },
    }


def _check_runtime_file(table: dict[str, Any], runtime_file: Path) -> None:
    if runtime_file.exists() and json.loads(runtime_file.read_text()) != runtime_table(table):
        raise SystemExit(
            f"{runtime_file} does not match the calibration table. It is generated from the "
            "calibration table: make the change there and run with --regenerate, or restore "
            "the generated file."
        )


def _write_tables(data_file: Path, runtime_file: Path | None, table: dict[str, Any]) -> None:
    """Renormalize the multipliers, write the table, and regenerate the runtime table from it."""
    if runtime_file is not None and data_file.resolve() == runtime_file.resolve():
        raise SystemExit("--data-file and --runtime-file must be different paths")
    _renormalize(table)
    _atomic_write(data_file, json.dumps(table, indent=2) + "\n")
    if runtime_file is not None:
        _atomic_write(runtime_file, json.dumps(runtime_table(table), indent=2) + "\n")


def _write_record(
    data_file: Path, runtime_file: Path | None, key: str, record: dict[str, Any]
) -> None:
    """Re-read the table, add one record, and write both tables."""
    table = json.loads(data_file.read_text())
    if runtime_file is not None:
        _check_runtime_file(table, runtime_file)
    table["models"][key] = record
    _write_tables(data_file, runtime_file, table)


def _regenerate(data_file: Path, runtime_file: Path | None) -> None:
    """Rewrite the multipliers and the runtime table from a hand-edited calibration table."""
    _write_tables(data_file, runtime_file, json.loads(data_file.read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", action="append", default=[], help="Tinker model id (repeatable)")
    parser.add_argument(
        "--missing", action="store_true", help="calibrate every trainable model without an entry"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the models to calibrate and exit"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="recompute the multipliers and the runtime table from --data-file and exit",
    )
    parser.add_argument(
        "--data-file", type=Path, default=CALIBRATION_FILE, help="calibration table to update"
    )
    parser.add_argument(
        "--runtime-file",
        type=Path,
        default=None,
        help="runtime table to regenerate from --data-file (default: the package's "
        "lr_multipliers.json when --data-file is the package's calibration table, else none)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="where corpus windows are cached"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="where finished runs are saved (default: <runs dir>/lr_calibration/<protocol>)",
    )
    args = parser.parse_args()
    if args.regenerate and (args.dry_run or args.model or args.missing):
        parser.error("--regenerate cannot be combined with --dry-run, --model or --missing")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    table = json.loads(args.data_file.read_text())
    runtime_file = args.runtime_file
    if runtime_file is None and args.data_file.resolve() == CALIBRATION_FILE.resolve():
        runtime_file = RUNTIME_FILE
    if args.regenerate:
        _regenerate(args.data_file, runtime_file)
        print(f"wrote {args.data_file}" + (f" and {runtime_file}" if runtime_file else ""))
        return
    if runtime_file is not None:
        _check_runtime_file(table, runtime_file)
    models = one_id_per_base_model(
        args.model + (missing_models(table, tinker.RestClient()) if args.missing else [])
    )
    if args.dry_run:
        print("\n".join(models))
        return
    if not models:
        parser.error("pass --model or --missing")
    protocol = table["protocol"]
    corpus, config = _protocol_config(table)
    windows = load_corpus(corpus, args.cache_dir)
    runs_dir = args.runs_dir or Path(cli_utils.runs_path("lr_calibration", protocol["name"]))
    _check_runs_dir(runs_dir, corpus, config)
    service_client = tinker.ServiceClient()

    async def one(model_name: str) -> bool:
        try:
            fit, runs, base_bpb = await calibrate_model(
                service_client,
                model_name,
                windows,
                config,
                protocol["grid_center"],
                {seed: RunCache(runs_dir, model_name, seed) for seed in protocol["seeds"]},
            )
        except Exception:
            logger.exception("%s: calibration failed", model_name)
            return False
        print(
            f"{model_name}: opt_lr={fit.opt_lr:.3e} edge_lr={fit.edge_lr:.3e} "
            f"base_bpb={base_bpb:.4f} min_bpb={fit.min_loss:.4f}",
            flush=True,
        )
        _write_record(
            args.data_file,
            runtime_file,
            base_model_name(model_name),
            _record(model_name, fit, runs, base_bpb),
        )
        return True

    async def run_all() -> list[bool]:
        return list(await asyncio.gather(*[one(m) for m in models]))

    failed = [m for m, ok in safezip(models, asyncio.run(run_all())) if not ok]
    print(f"wrote {args.data_file}" + (f" and {runtime_file}" if runtime_file else ""))
    if failed:
        sys.exit(f"calibration failed for {failed}; rerun to resume from {runs_dir}")


if __name__ == "__main__":
    main()
