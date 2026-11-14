"""Collect the best paintings of every training step into one folder.

    python -m tinker_cookbook.recipes.paint_rl.progression \
        artifact_dir=/tmp/paint_rl/verifier/paintings out_dir=/tmp/paint_rl/verifier/progression top_k=3

Writes ``<out_dir>/step_XXXX/rank<k>_<prompt id>_r<reward>.png`` (+ the
matching ``.js``), an ``index.csv`` describing every copied painting, and
``progression.png``: a contact sheet with one row per step (best painting
first), so training progress can be seen at a glance.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import chz
from PIL import Image, ImageDraw


@chz.chz
class Config:
    artifact_dir: str
    out_dir: str
    split: str = "train"
    top_k: int = 3
    thumb: int = 192
    every: int = 1  # keep every N-th step in the contact sheet


def collect(config: Config) -> Path:
    artifact_dir = Path(config.artifact_dir)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in (artifact_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if r["split"] == config.split and r.get("png_path"):
            by_step[int(r["iteration"])].append(r)

    index_rows: list[dict[str, object]] = []
    sheet_rows: list[tuple[int, list[tuple[Path, float]]]] = []
    for step in sorted(by_step):
        best = sorted(by_step[step], key=lambda r: -r["reward"])[: config.top_k]
        step_dir = out_dir / f"step_{step:04d}"
        step_dir.mkdir(exist_ok=True)
        thumbs: list[tuple[Path, float]] = []
        for rank, r in enumerate(best, start=1):
            stem = f"rank{rank}_{r['prompt_id']}_r{r['reward']:.2f}"
            png = step_dir / f"{stem}.png"
            shutil.copyfile(artifact_dir / r["png_path"], png)
            if r.get("js_path"):
                shutil.copyfile(artifact_dir / r["js_path"], step_dir / f"{stem}.js")
            thumbs.append((png, float(r["reward"])))
            index_rows.append(
                {
                    "step": step,
                    "rank": rank,
                    "reward": round(float(r["reward"]), 4),
                    "score": round(float(r["score"]), 4),
                    "prompt": r["prompt"],
                    "png": str(png.relative_to(out_dir)),
                }
            )
        if step % config.every == 0:
            sheet_rows.append((step, thumbs))

    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "rank", "reward", "score", "prompt", "png"])
        writer.writeheader()
        writer.writerows(index_rows)

    # contact sheet: one row per step, columns = rank
    t = config.thumb
    label_w = 90
    width = label_w + config.top_k * (t + 6)
    height = max(1, len(sheet_rows)) * (t + 6)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for row_i, (step, thumbs) in enumerate(sheet_rows):
        y = row_i * (t + 6)
        draw.text((6, y + t // 2 - 6), f"step {step}", fill="black")
        for col, (png, reward) in enumerate(thumbs):
            im = Image.open(png).convert("RGB")
            im.thumbnail((t, t))
            x = label_w + col * (t + 6)
            sheet.paste(im, (x, y))
            draw.text((x + 4, y + 2), f"{reward:.2f}", fill="black")
    sheet.save(out_dir / "progression.png")
    return out_dir


def main(config: Config) -> None:
    out = collect(config)
    print(f"Wrote {out}/progression.png and {out}/index.csv")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
