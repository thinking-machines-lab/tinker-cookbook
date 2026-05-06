"""Copy trainable Tinker weights from one account into another.

Usage:
    python -m tinker_cookbook.scripts.copy_checkpoint \\
        --source-path tinker://<run-id>:train:0/weights/<name> \\
        --source-api-key "$SRC_TINKER_API_KEY" \\
        --destination-api-key "$DST_TINKER_API_KEY"

The source path must be a `.../weights/<name>` checkpoint from `save_state`.
Sampler-only checkpoints (`.../sampler_weights/<name>`) cannot be loaded into
a training client.
"""

import argparse
import re

import tinker

CHECKPOINT_NAME_RE = re.compile(r"/weights/([^/]+)$")
TRAINING_WEIGHTS_PATH_RE = re.compile(r"/weights/[^/]+$")


def copy_checkpoint(
    source_path: str,
    source_api_key: str,
    destination_api_key: str,
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

    # TODO: once token-scoped metadata lookup works in the SDK, replace this
    # manual setup with:

    # destination_client = tinker.ServiceClient(api_key=destination_api_key)
    # training_client = destination_client.create_training_client_from_state(
    #     source_path,
    #     weights_access_token=source_api_key,
    #     user_metadata={"copied_from_path": source_path},
    # )

    source_client = tinker.ServiceClient(api_key=source_api_key)
    weights_info = source_client.create_rest_client().get_weights_info_by_tinker_path(
        source_path
    ).result()
    if not weights_info.is_lora or weights_info.lora_rank is None:
        raise SystemExit(
            f"Source checkpoint {source_path!r} is not a LoRA run; "
            "non-LoRA copies are not supported."
        )

    destination_client = tinker.ServiceClient(api_key=destination_api_key)
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

    training_client.load_state(source_path, weights_access_token=source_api_key).result()

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
            f"Could not derive a checkpoint name from {source_path!r}; "
            "pass --output-name."
        )
    return match.group(1)


def main() -> None:
    description = (__doc__ or "").split("\n\n")[0]
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--source-api-key", required=True)
    parser.add_argument("--destination-api-key", required=True)
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
        args.source_api_key,
        args.destination_api_key,
        args.output_name,
        args.output_kind,
    )


if __name__ == "__main__":
    main()
