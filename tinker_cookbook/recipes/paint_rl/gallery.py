"""Build a browsable HTML gallery from a paint_rl artifact directory.

    python -m tinker_cookbook.recipes.paint_rl.gallery artifact_dir=/tmp/paint_rl/run/paintings

Writes ``gallery.html`` next to ``index.jsonl``: one section per training
step with every rendered painting, its reward, and its JavaScript, plus a
"progression" strip showing the best painting of each step.
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

import chz

_STYLE = """
body{font-family:system-ui,sans-serif;margin:20px;background:#fafafa;color:#222}
.step{margin-bottom:28px}.grid{display:flex;flex-wrap:wrap;gap:10px}
.card{width:230px;background:#fff;border:1px solid #ddd;border-radius:6px;padding:6px;font-size:12px}
.card img{width:216px;height:216px;object-fit:contain;background:#fff;border:1px solid #eee}
.bad{color:#b00}.strip img{width:150px;height:150px;margin:3px;border:1px solid #ddd}
details pre{max-height:300px;overflow:auto;font-size:10px;background:#f4f4f4;padding:6px}
"""


@chz.chz
class Config:
    artifact_dir: str
    split: str = "train"
    max_per_step: int = 64


def build_gallery(artifact_dir: Path, split: str, max_per_step: int) -> Path:
    index = artifact_dir / "index.jsonl"
    rows = [
        json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows = [r for r in rows if r["split"] == split]
    by_step: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_step[r["iteration"]].append(r)

    parts = [f"<html><head><meta charset='utf-8'><style>{_STYLE}</style></head><body>"]
    parts.append(f"<h1>paint_rl gallery — {html.escape(split)}</h1>")

    parts.append("<h2>Progression (best painting per step)</h2><div class='strip'>")
    for step in sorted(by_step):
        best = max(by_step[step], key=lambda r: r["reward"])
        if best.get("png_path"):
            parts.append(
                f"<a href='{best['png_path']}' title='step {step}: {html.escape(best['prompt'])} reward {best['reward']:.2f}'>"
                f"<img src='{best['png_path']}'/></a>"
            )
    parts.append("</div>")

    for step in sorted(by_step):
        step_rows = sorted(by_step[step], key=lambda r: -r["reward"])[:max_per_step]
        n = len(by_step[step])
        compiled = sum(r["compiled"] for r in by_step[step])
        mean_reward = sum(r["reward"] for r in by_step[step]) / n
        parts.append(
            f"<div class='step'><h2>Step {step}</h2><p>{n} rollouts, {compiled} compiled, "
            f"mean reward {mean_reward:.3f}</p><div class='grid'>"
        )
        for r in step_rows:
            img = (
                f"<img src='{r['png_path']}'/>"
                if r.get("png_path")
                else "<div class='bad'>(no image)</div>"
            )
            code = ""
            if r.get("js_path"):
                js = (artifact_dir / r["js_path"]).read_text(encoding="utf-8")
                code = f"<details><summary>code ({r['code_chars']} chars)</summary><pre>{html.escape(js)}</pre></details>"
            err = (
                f"<div class='bad'>{html.escape(r['render_error'])}</div>"
                if r.get("render_error")
                else ""
            )
            parts.append(
                f"<div class='card'>{img}<div><b>{html.escape(r['prompt'])}</b></div>"
                f"<div>reward {r['reward']:.3f} · score {r['score']:.3f}</div>{err}{code}</div>"
            )
        parts.append("</div></div>")
    parts.append("</body></html>")
    out = artifact_dir / f"gallery_{split}.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main(config: Config) -> None:
    out = build_gallery(Path(config.artifact_dir), config.split, config.max_per_step)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(chz.entrypoint(Config))
