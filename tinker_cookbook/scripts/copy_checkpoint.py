"""Copy trainable Tinker weights into the currently authenticated account.

This copies a trainable Tinker checkpoint by loading the source weights with an
access token and saving them as a new destination-owned checkpoint.

Note: this only works for trainable `weights/...` checkpoints, not sampler-only
`sampler_weights/...` checkpoints.

Usage:
    export SRC_TINKER_ACCESS_TOKEN=...  # grants access to the source checkpoint;
                                        # a Tinker API key works for now

    python -m tinker_cookbook.scripts.copy_checkpoint \\
        --source-path tinker://<run-id>:train:0/weights/<name> \\
        --source-access-token "$SRC_TINKER_ACCESS_TOKEN"
    # Prints: sampler_path: tinker://<new-run-id>:train:0/sampler_weights/<name>

By default, this script saves sampler weights; pass `--output-kind training` to
save a trainable `weights/...` checkpoint instead.
"""

import argparse
import importlib.metadata
import re

import tinker

CHECKPOINT_NAME_RE = re.compile(r"/weights/([^/]+)$")
TRAINING_WEIGHTS_PATH_RE = re.compile(r"/weights/[^/]+$")


def copy_checkpoint(
    source_path: str,
    source_access_token: str,
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

    destination_client = tinker.ServiceClient()
    tinker_version = importlib.metadata.version("tinker").split(".")

    # SDK >=0.18.3 can fetch source checkpoint metadata with weights_access_token and is preferred.
    # Older SDKs need the manual metadata lookup below.
    if (int(tinker_version[0]), int(tinker_version[1]), int(tinker_version[2])) >= (0, 18, 3):
        training_client = destination_client.create_training_client_from_state(
            source_path,
            weights_access_token=source_access_token,
            user_metadata={"copied_from_path": source_path},
        )
    else:
        source_client = tinker.ServiceClient(api_key=source_access_token)
        weights_info = (
            source_client.create_rest_client().get_weights_info_by_tinker_path(source_path).result()
        )
        if weights_info.lora_rank is None:
            raise SystemExit(
                f"Could not determine rank metadata for source checkpoint {source_path!r}; "
                "cannot recreate the destination training run."
            )

        training_client = destination_client.create_lora_training_client(
            base_model=weights_info.base_model,
            rank=weights_info.lora_rank,
            train_mlp=weights_info.train_mlp if weights_info.train_mlp is not None else True,
            train_attn=weights_info.train_attn if weights_info.train_attn is not None else True,
            train_unembed=weights_info.train_unembed
            if weights_info.train_unembed is not None
            else True,
            user_metadata={"copied_from_path": source_path},
        )
        training_client.load_state(source_path, weights_access_token=source_access_token).result()

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
        required=True,
        help="Token that grants access to the source checkpoint. Tinker API key works for now.",
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
        args.output_name,
        args.output_kind,
    )


if __name__ == "__main__":
    main()
