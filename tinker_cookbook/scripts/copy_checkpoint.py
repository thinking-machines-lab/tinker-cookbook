"""Copy trainable Tinker weights into the currently authenticated account.

This copies a trainable Tinker checkpoint by loading the source weights and
saving them as a new destination-owned checkpoint.

Note: this only works for trainable `weights/...` checkpoints, not sampler-only
`sampler_weights/...` checkpoints.

Usage:
    python -m tinker_cookbook.scripts.copy_checkpoint \\
        --source-path tinker://<run-id>:train:0/weights/<name>
    # Prints: sampler_path: tinker://<new-run-id>:train:0/sampler_weights/<name>

To save into a specific destination project, pass `--destination-project-id`:
    python -m tinker_cookbook.scripts.copy_checkpoint \\
        --source-path tinker://<run-id>:train:0/weights/<name> \\
        --destination-project-id <project-id>

For cross-org copies, pass `--source-access-token` or set
`SRC_TINKER_ACCESS_TOKEN` to a token that can read the source checkpoint:
    python -m tinker_cookbook.scripts.copy_checkpoint \\
        --source-path tinker://<run-id>:train:0/weights/<name> \\
        --source-access-token "$SRC_TINKER_ACCESS_TOKEN" \\
        --destination-project-id <project-id>

By default, this script saves sampler weights; pass `--output-kind training` to
save a trainable `weights/...` checkpoint instead.
"""

import argparse
import importlib.metadata
import os
import re

import tinker

CHECKPOINT_NAME_RE = re.compile(r"/weights/([^/]+)$")
TRAINING_WEIGHTS_PATH_RE = re.compile(r"/weights/[^/]+$")


def copy_checkpoint(
    source_path: str,
    source_access_token: str | None,
    destination_project_id: str | None,
    output_name: str | None,
    output_kind: str,
) -> None:
    if not TRAINING_WEIGHTS_PATH_RE.search(source_path):
        raise SystemExit(
            f"Source path must be a tinker://.../weights/<name> checkpoint, "
            f"got {source_path!r}. Sampler checkpoints cannot be copied "
            "directly; use the matching weights/... path."
        )
    name = output_name or checkpoint_name_from_path(source_path)

    tinker_version = importlib.metadata.version("tinker").split(".")

    if (int(tinker_version[0]), int(tinker_version[1]), int(tinker_version[2])) >= (0, 19, 0):
        destination_client = tinker.ServiceClient(project_id=destination_project_id)
        training_client = destination_client.create_training_client_from_state(
            source_path,
            weights_access_token=source_access_token,
            user_metadata={"copied_from_path": source_path},
        )
    else:
        raise SystemExit(
            "copy_checkpoint requires tinker>=0.19.0. "
            f"Found tinker=={'.'.join(tinker_version)}; upgrade the Tinker SDK and try again."
        )

    if output_kind == "training":
        future = training_client.save_state(name)
        print(f"training_path: {future.result().path}")
    elif output_kind == "sampler":
        future = training_client.save_weights_for_sampler(name)
        print(f"sampler_path:  {future.result().path}")


def checkpoint_name_from_path(source_path: str) -> str:
    match = CHECKPOINT_NAME_RE.search(source_path)
    if match is None:
        raise SystemExit(
            f"Could not derive a checkpoint name from {source_path!r}; pass --output-name."
        )
    return match.group(1)


def main() -> None:
    description = (__doc__ or "").split("\n\n")[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source-path", required=True)
    parser.add_argument(
        "--source-access-token",
        default=os.environ.get("SRC_TINKER_ACCESS_TOKEN"),
        help=(
            "Token that grants access to the source checkpoint. Defaults to "
            "SRC_TINKER_ACCESS_TOKEN; only needed for cross-org copies."
        ),
    )
    parser.add_argument(
        "--destination-project-id",
        "--target-project-id",
        dest="destination_project_id",
        default=None,
        help="Project ID to save the destination checkpoint into.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Checkpoint name to save in the destination account. "
            "Defaults to the source checkpoint name."
        ),
    )
    parser.add_argument(
        "--output-kind",
        choices=["sampler", "training"],
        default="sampler",
        help="What to save in the destination account.",
    )
    args = parser.parse_args()
    copy_checkpoint(
        args.source_path,
        args.source_access_token,
        args.destination_project_id,
        args.output_name,
        args.output_kind,
    )


if __name__ == "__main__":
    main()
