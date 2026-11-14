"""Compare paint_rl runs (e.g. verifier vs judge) from their metrics.

    python -m tinker_cookbook.recipes.paint_rl.compare_runs \
        runs='{"verifier": "/tmp/paint_rl/verifier", "judge": "/tmp/paint_rl/judge"}' \
        out=/tmp/paint_rl/compare.png

Prints a per-run summary table and writes a figure with the training curves
of the compile-gate pass rate, reward-model score of compiled paintings, the
top-score tie rate, total reward and code length, plus the held-out score
from ``<run>/heldout_eval/metrics.jsonl``. Runs with different rewards
optimize different scales; compare those on the held-out score.
"""

from __future__ import annotations

import json
from pathlib import Path

import chz

# (metrics file relative to the run, key, label)
KEYS: tuple[tuple[str, str, str], ...] = (
    ("heldout_eval/metrics.jsonl", "test/env/all/eval_strong/score", "held-out score"),
    ("metrics.jsonl", "env/all/reward/compile_ok", "compile gate pass rate"),
    ("metrics.jsonl", "env/all/reward/score_if_compiled", "reward score (compiled)"),
    ("metrics.jsonl", "env/all/scorer/top_tie", "top-score tie rate"),
    ("metrics.jsonl", "env/all/reward/total", "total reward"),
    ("metrics.jsonl", "env/all/code/chars", "code length (chars)"),
)


@chz.chz
class Config:
    runs: dict[str, str]
    out: str = "compare.png"
    smooth: int = 3


def load_metrics(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _series(rows: list[dict[str, float]], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in rows:
        if key in row and "step" in row:
            xs.append(int(row["step"]))
            ys.append(float(row[key]))
    return xs, ys


def _smooth(ys: list[float], k: int) -> list[float]:
    if k <= 1:
        return ys
    out = []
    for i in range(len(ys)):
        window = ys[max(0, i - k + 1) : i + 1]
        out.append(sum(window) / len(window))
    return out


def main(config: Config) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(KEYS), figsize=(4.5 * len(KEYS), 3.6))
    print(f"{'run':<12s} {'steps':>5s} " + " ".join(f"{label[:22]:>24s}" for *_, label in KEYS))
    for name, log_dir in config.runs.items():
        files = {f: load_metrics(Path(log_dir) / f) for f, _, _ in KEYS}
        rows = files["metrics.jsonl"]
        cells = []
        for ax, (file, key, label) in zip(axes, KEYS, strict=True):
            xs, ys = _series(files[file], key)
            if xs:
                ax.plot(xs, _smooth(ys, config.smooth), label=name)
                first = sum(ys[: config.smooth]) / min(len(ys), config.smooth)
                last = sum(ys[-config.smooth :]) / min(len(ys), config.smooth)
                cells.append(f"{first:>10.3f} -> {last:<10.3f}")
            else:
                cells.append(f"{'n/a':>24s}")
            ax.set_title(label)
            ax.set_xlabel("step")
        print(f"{name:<12s} {len(rows):>5d} " + " ".join(f"{c:>24s}" for c in cells))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(config.out, dpi=120)
    print(f"Wrote {config.out}")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
