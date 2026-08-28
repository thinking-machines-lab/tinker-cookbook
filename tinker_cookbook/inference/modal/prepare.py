"""PREPARE: merge a tinker:// checkpoint into a servable model on the Volume.

    modal run -m tinker_cookbook.inference.modal.prepare \\
        --tinker-path tinker://<run-id>/sampler_weights/<name> \\
        --base-model Qwen/Qwen3-8B --name my-finetune

Also callable from another app via prepare.remote(...).
"""

from __future__ import annotations

import os

import modal

from .common import (
    ARTIFACTS_PATH,
    HF_CACHE_PATH,
    MINUTES,
    app,
    artifact_dir,
    artifacts,
    hf_cache,
    prepare_image,
)

# Read from the local env at deploy time. TINKER_API_KEY is required to download
# the checkpoint; HF_TOKEN is optional (only gated base models need it).
secret = modal.Secret.from_dict(
    {
        "TINKER_API_KEY": os.environ.get("TINKER_API_KEY", ""),
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    }
)


@app.function(
    image=prepare_image,
    volumes={ARTIFACTS_PATH: artifacts, HF_CACHE_PATH: hf_cache},
    secrets=[secret],
    gpu="H100",  # merge runs the dequant/requant math on GPU
    cpu=8.0,
    memory=65536,
    timeout=60 * MINUTES,
)
def prepare(*, tinker_path: str, base_model: str, name: str) -> str:
    from tinker_cookbook import weights

    output_path = artifact_dir(name)
    downloaded = weights.download(tinker_path=tinker_path, output_dir="/tmp/checkpoint")
    weights.build_hf_model(base_model=base_model, adapter_path=downloaded, output_path=output_path)

    artifacts.commit()
    print(f"[prepare] merged model ready at {output_path}")
    return output_path


@app.local_entrypoint()
def main(tinker_path: str, base_model: str, name: str) -> None:
    output_path = prepare.remote(tinker_path=tinker_path, base_model=base_model, name=name)
    print(f"\nArtifact on the tinker-artifacts Volume: {output_path}")
    print(
        f"Serve it:\n  FINETUNE={name} MODEL={base_model} modal deploy -m tinker_cookbook.inference.modal.serve"
    )
