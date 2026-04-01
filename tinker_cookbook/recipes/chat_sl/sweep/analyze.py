"""
Analyze LR sweep results from W&B and generate tables + plots.

Pulls finished runs from a W&B project, groups them by model, and produces:
- Per-model markdown tables (lr, lora_rank, test_nll, wall_time)
- Per-model NLL-vs-step curve plots (one subplot per rank)
- A combined sft_sweep.md ready to commit

Usage::

    # Generate results from the default project
    uv run --with wandb --with matplotlib python -m tinker_cookbook.recipes.chat_sl.sweep.analyze

    # Custom project and output directory
    uv run --with wandb --with matplotlib python -m tinker_cookbook.recipes.chat_sl.sweep.analyze \\
        --wandb-project lr-sweep-2026-03 \\
        --output-dir tinker_cookbook/recipes/chat_sl/results
"""

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    model: str
    rank: int
    lr: float
    test_nll: float
    train_nll: float
    wall_time_min: float
    run_name: str
    # step -> test_nll for NLL curves
    history: list[tuple[int, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# W&B data pulling
# ---------------------------------------------------------------------------

_RUN_NAME_RE = re.compile(r"^tulu3-(.+?)-(\d+)rank-(.+?)lr-(\d+)batch-")


def _parse_run_name(name: str) -> tuple[str, int, float, int] | None:
    """Extract (model, rank, lr, batch_size) from a run name.

    Returns None if the name doesn't match the expected pattern.
    """
    m = _RUN_NAME_RE.match(name)
    if m is None:
        return None
    model = m.group(1)
    rank = int(m.group(2))
    lr = float(m.group(3))
    batch_size = int(m.group(4))
    return model, rank, lr, batch_size


def _display_model_name(slug: str) -> str:
    """Convert raw model slug to a display name.

    Uses the HuggingFace model ID (``org/model``) so the heading matches
    what users actually pass to ``model_name=``.
    """
    return _slug_to_hf_name(slug)


def _slug_to_hf_name(slug: str) -> str:
    """Restore the ``org/model`` HuggingFace name from a run-name slug.

    Run names replace ``/`` with ``-`` (see ``train.py``), making the
    org/model boundary ambiguous for orgs that contain hyphens (e.g.
    ``deepseek-ai``, ``meta-llama``).  We resolve the ambiguity by
    checking known org prefixes longest-first.
    """
    # Sorted longest-first so "deepseek-ai" is tried before "deepseek".
    known_orgs = sorted(
        ["meta-llama", "deepseek-ai", "Qwen", "nvidia", "openai", "moonshotai"],
        key=len,
        reverse=True,
    )
    for org in known_orgs:
        prefix = org + "-"
        if slug.startswith(prefix):
            return org + "/" + slug[len(prefix) :]
    # Unknown org – best-effort: split on first hyphen.
    return slug.replace("-", "/", 1)


def fetch_runs(wandb_project: str) -> list[RunResult]:
    """Fetch all finished runs from a W&B project."""
    import wandb  # type: ignore[import-untyped]

    api = wandb.Api()
    raw_runs = api.runs(wandb_project, filters={"state": "finished"})

    results: list[RunResult] = []
    for r in raw_runs:
        parsed = _parse_run_name(r.name)
        if parsed is None:
            print(f"  Skipping unparseable run: {r.name}")
            continue

        model, rank, lr, _ = parsed
        summary = dict(r.summary)
        test_nll = summary.get("test/nll")
        train_nll = summary.get("train_mean_nll")
        runtime_s = summary.get("_runtime", 0)

        if test_nll is None or train_nll is None:
            print(f"  Skipping run with missing metrics: {r.name}")
            continue

        # Fetch step-level history for NLL curves
        hist = r.history(keys=["test/nll"], pandas=False)
        history_points: list[tuple[int, float]] = []
        for row in hist:
            step = row.get("_step")
            nll = row.get("test/nll")
            if step is not None and nll is not None:
                history_points.append((int(step), float(nll)))
        history_points.sort(key=lambda x: x[0])

        results.append(
            RunResult(
                model=model,
                rank=rank,
                lr=lr,
                test_nll=float(test_nll),
                train_nll=float(train_nll),
                wall_time_min=float(runtime_s) / 60.0,
                run_name=r.name,
                history=history_points,
            )
        )

    print(f"Fetched {len(results)} finished runs from {wandb_project}")
    return results


# ---------------------------------------------------------------------------
# Grouping and analysis
# ---------------------------------------------------------------------------

# Runs with test_nll above this are considered diverged
DIVERGENCE_THRESHOLD = 2.0


def group_by_model(runs: list[RunResult]) -> dict[str, list[RunResult]]:
    by_model: dict[str, list[RunResult]] = defaultdict(list)
    for r in runs:
        by_model[r.model].append(r)
    # Sort within each model by (rank, lr)
    for model_runs in by_model.values():
        model_runs.sort(key=lambda r: (r.rank, r.lr))
    return dict(by_model)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------


def generate_nll_plot(
    model_slug: str,
    runs: list[RunResult],
    output_dir: Path,
) -> str:
    """Generate a NLL-vs-step plot for one model, with one subplot per rank.

    Returns the relative path to the generated PNG.
    """
    import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]

    # Group runs by rank
    by_rank: dict[int, list[RunResult]] = defaultdict(list)
    for r in runs:
        by_rank[r.rank].append(r)

    ranks = sorted(by_rank.keys())
    n_ranks = len(ranks)

    fig, axes = plt.subplots(1, n_ranks, figsize=(5 * n_ranks, 4), squeeze=False)
    fig.suptitle(f"{_display_model_name(model_slug)} — Test NLL vs Step", fontsize=13)

    # Single color, alpha encodes LR: smaller LR = darker, larger LR = lighter.
    # This makes it easy to see at a glance which direction performs better.
    base_color = "#1f77b4"
    non_diverged_lrs = sorted({r.lr for r in runs if r.test_nll < DIVERGENCE_THRESHOLD})
    if len(non_diverged_lrs) > 1:
        import math

        log_lrs = [math.log10(lr) for lr in non_diverged_lrs]
        log_min, log_max = min(log_lrs), max(log_lrs)
        # Map: smallest LR -> alpha 1.0 (darkest), largest LR -> alpha 0.3 (lightest)
        lr_alpha = {
            lr: 1.0 - 0.7 * (math.log10(lr) - log_min) / (log_max - log_min)
            for lr in non_diverged_lrs
        }
    else:
        lr_alpha = dict.fromkeys(non_diverged_lrs, 0.9)

    for col, rank in enumerate(ranks):
        ax = axes[0][col]
        ax.set_title(f"rank={rank}")
        ax.set_xlabel("Step")
        if col == 0:
            ax.set_ylabel("Test NLL")

        for r in sorted(by_rank[rank], key=lambda x: x.lr):
            if not r.history:
                continue
            steps = [h[0] for h in r.history]
            nlls = [h[1] for h in r.history]

            # Skip diverged runs based on final test NLL (not max over history,
            # since step 0 always has high untrained loss)
            if r.test_nll >= DIVERGENCE_THRESHOLD:
                continue

            ax.plot(
                steps,
                nlls,
                label=f"lr={r.lr:.0e}",
                color=base_color,
                linewidth=1.5,
                alpha=lr_alpha.get(r.lr, 0.5),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{model_slug.replace('/', '_')}_nll_curves.png"
    filepath = plots_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return f"plots/{filename}"


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------


def _format_lr(lr: float) -> str:
    return f"{lr:.0e}"


def generate_model_section(
    model_slug: str,
    runs: list[RunResult],
    plot_path: str | None,
) -> str:
    """Generate a markdown section for one model."""
    lines: list[str] = []
    pretty = _display_model_name(model_slug)
    hf_name = _slug_to_hf_name(model_slug)
    lines.append(f"## {pretty}")
    lines.append("")

    # Separate healthy vs diverged
    all_runs = runs  # includes diverged, for inferring full grid
    healthy = [r for r in runs if r.test_nll < DIVERGENCE_THRESHOLD]
    diverged = [r for r in runs if r.test_nll >= DIVERGENCE_THRESHOLD]

    # Infer sweep grid from all runs (including diverged)
    all_lrs = sorted({r.lr for r in all_runs})
    all_ranks = sorted({r.rank for r in all_runs})
    lr_list_str = ", ".join(_format_lr(lr) for lr in all_lrs)
    rank_list_str = ", ".join(str(r) for r in all_ranks)

    # Configuration / reproduction
    lines.append("**Configuration:**")
    lines.append(f"- Model: `{hf_name}`")
    lines.append("- Dataset: tulu3 (train split for training, test split for evaluation)")
    lines.append("- Batch size: 128")
    lines.append(f"- Learning rates: [{lr_list_str}]")
    lines.append(f"- LoRA ranks: [{rank_list_str}]")
    lines.append("- Metric: `test/nll` — negative log-likelihood on held-out test split")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Reproduce</summary>")
    lines.append("")
    lines.append("```bash")
    lines.append("uv run python -m tinker_cookbook.recipes.chat_sl.sweep \\")
    lines.append("    recipe=sft \\")
    lines.append(f"    base.model_name={hf_name} \\")
    lines.append("    base.dataset=tulu3 \\")
    lines.append("    base.batch_size=128 \\")
    lines.append("    metric=test/nll \\")
    lines.append(f"    'learning_rates=[{lr_list_str}]' \\")
    lines.append(f"    'lora_ranks=[{rank_list_str}]'")
    lines.append("```")
    lines.append("")
    lines.append("</details>")
    lines.append("")

    if not healthy:
        lines.append("All runs diverged.")
        lines.append("")
        return "\n".join(lines)

    # Results table
    lines.append("**Results:**")
    lines.append("")
    lines.append("| LR | LoRA Rank | Test NLL | Train NLL | Wall Time (min) |")
    lines.append("|---:|----------:|---------:|----------:|----------------:|")
    for r in sorted(healthy, key=lambda r: (r.rank, r.lr)):
        lines.append(
            f"| {_format_lr(r.lr)} | {r.rank} | {r.test_nll:.4f} | {r.train_nll:.4f} | {r.wall_time_min:.0f} |"
        )
    lines.append("")

    # Best config
    best = min(healthy, key=lambda r: r.test_nll)
    lines.append(
        f"**Best config:** rank={best.rank}, lr={_format_lr(best.lr)}, test_nll={best.test_nll:.4f}"
    )
    lines.append("")

    # Avg wall time
    avg_time = sum(r.wall_time_min for r in healthy) / len(healthy)
    lines.append(f"**Avg wall time per run:** {avg_time:.0f} min")
    lines.append("")

    # NLL curve plot
    if plot_path:
        lines.append(f"![NLL curves for {pretty}]({plot_path})")
        lines.append("")

    # Note diverged runs
    if diverged:
        div_lr_list = ", ".join(sorted({_format_lr(r.lr) for r in diverged}))
        div_rank_list = ", ".join(str(r) for r in sorted({r.rank for r in diverged}))
        lines.append(
            f"> **Note:** {len(diverged)} run(s) diverged (test_nll > {DIVERGENCE_THRESHOLD}) "
            f"at lr={{{div_lr_list}}} with rank={{{div_rank_list}}} and are excluded from the table above."
        )
        lines.append("")

    return "\n".join(lines)


def _heading_to_anchor(heading: str) -> str:
    """Convert a markdown heading to a GitHub-compatible anchor link."""
    return heading.lower().replace(" ", "-").replace("(", "").replace(")", "")


def generate_sft_sweep_md(
    model_names: list[str],
    model_sections: list[str],
) -> str:
    """Generate the full sft_sweep.md document."""
    lines: list[str] = []
    lines.append("# SFT Sweep Results")
    lines.append("")
    lines.append("Empirical hyperparameter sweep results for supervised fine-tuning (SFT).")
    lines.append(
        "Use these as a reference when choosing learning rate and LoRA rank for your model."
    )
    lines.append("")
    lines.append("**Setup:**")
    lines.append("- Dataset: tulu3")
    lines.append("- Batch size: 128")
    lines.append("- Training steps: 780")
    lines.append("- Adapter: LoRA")
    lines.append("")
    lines.append(
        "> **Note:** Wall times are approximate and depend on server load at the time of "
        "the run. They may fluctuate significantly between runs."
    )
    lines.append("")

    # Table of contents
    lines.append("## Table of Contents")
    lines.append("")
    for name in model_names:
        anchor = _heading_to_anchor(name)
        lines.append(f"- [{name}](#{anchor})")
    lines.append("")
    lines.append("---")
    lines.append("")

    for section in model_sections:
        lines.append(section)
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LR sweep results from W&B",
    )
    parser.add_argument(
        "--wandb-project",
        default="lr-sweep-2026-03",
        help="W&B project name (default: lr-sweep-2026-03)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: <script_dir>/results)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (useful if matplotlib is not available)",
    )
    args = parser.parse_args()

    # Default output: tinker_cookbook/recipes/chat_sl/results/
    default_output = Path(__file__).parent.parent / "results"
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching runs from W&B project: {args.wandb_project}")
    runs = fetch_runs(args.wandb_project)

    if not runs:
        print("No runs found.")
        return

    by_model = group_by_model(runs)
    print(f"Found {len(by_model)} models: {list(by_model.keys())}")

    model_sections: list[str] = []
    model_names: list[str] = []
    # Sort models by name for deterministic output
    for model_slug in sorted(by_model.keys()):
        model_runs = by_model[model_slug]
        model_names.append(_display_model_name(model_slug))
        print(f"\n--- {_display_model_name(model_slug)} ({len(model_runs)} runs) ---")

        plot_path: str | None = None
        if not args.no_plots:
            try:
                plot_path = generate_nll_plot(model_slug, model_runs, output_dir)
                print(f"  Plot saved: {plot_path}")
            except Exception as e:
                print(f"  Plot generation failed: {e}")

        section = generate_model_section(model_slug, model_runs, plot_path)
        model_sections.append(section)

        # Also print the table to stdout
        print(section)

    # Write the combined markdown
    md_content = generate_sft_sweep_md(model_names, model_sections)
    md_path = output_dir / "sft_sweep.md"
    md_path.write_text(md_content)
    print(f"\nResults written to {md_path}")


if __name__ == "__main__":
    main()
