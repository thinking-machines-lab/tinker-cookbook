"""
Upload experiment metrics to Weights & Biases.

Reads metrics.jsonl files from experiment log directories and uploads them
to wandb for visualization and comparison.
"""

import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def upload_metrics(
    metrics_file: str,
    project: str = "nemotron-cascade-2",
    run_name: str | None = None,
    tags: list[str] | None = None,
):
    """Upload a metrics.jsonl file to wandb."""
    import wandb

    if run_name is None:
        run_name = os.path.basename(os.path.dirname(metrics_file))

    run = wandb.init(
        project=project,
        name=run_name,
        tags=tags or [],
        config={"metrics_file": metrics_file},
    )

    with open(metrics_file) as f:
        for line in f:
            m = json.loads(line)
            step = m.pop("step", None)
            if step is not None:
                wandb.log(m, step=step)

    run.finish()
    logger.info(f"Uploaded {run_name} to wandb project {project}")


def upload_all(
    log_dir: str = os.path.expanduser("~/data/nemotron-cascade-2/experiment_logs"),
    project: str = "nemotron-cascade-2",
):
    """Upload all experiment logs to wandb."""
    for root, dirs, files in os.walk(log_dir):
        if "metrics.jsonl" in files:
            metrics_file = os.path.join(root, "metrics.jsonl")
            rel_path = os.path.relpath(root, log_dir)
            run_name = rel_path.replace("/", "_")

            # Determine tags
            tags = []
            if "lr_sweep" in rel_path:
                tags.append("lr_sweep")
            elif "full_sft" in rel_path:
                tags.append("full_sft")
            elif "medium_sft" in rel_path:
                tags.append("medium_sft")
            elif "ifrl" in rel_path:
                tags.append("if_rl")

            if "gpt-oss" in rel_path:
                tags.append("gpt-oss-120b")
            elif "Qwen" in rel_path:
                tags.append("qwen3-8b")

            logger.info(f"Uploading: {run_name} (tags: {tags})")
            try:
                upload_metrics(metrics_file, project=project, run_name=run_name, tags=tags)
            except Exception as e:
                logger.error(f"Failed to upload {run_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="Single metrics.jsonl to upload")
    parser.add_argument("--all", action="store_true", help="Upload all experiment logs")
    parser.add_argument("--project", type=str, default="nemotron-cascade-2")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    if args.file:
        upload_metrics(args.file, project=args.project, run_name=args.name)
    elif args.all:
        upload_all(project=args.project)
    else:
        print("Specify --file or --all")
