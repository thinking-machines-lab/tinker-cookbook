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
    MODEL_REGISTRY,
    app,
    artifact_dir,
    artifacts,
    hf_cache,
    model_config,
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

_PREPARE_FNS: dict[tuple[str, int], modal.Function] = {}


def _merge_checkpoint(*, tinker_path: str, base_model: str, name: str) -> str:
    from tinker_cookbook import weights

    output_path = artifact_dir(name)
    downloaded = weights.download(tinker_path=tinker_path, output_dir="/tmp/checkpoint")
    weights.build_hf_model(base_model=base_model, adapter_path=downloaded, output_path=output_path)

    artifacts.commit()
    print(f"[prepare] merged model ready at {output_path}")
    return output_path


def _prepare_fn(gpu: str, memory_mb: int) -> modal.Function:
    key = (gpu, memory_mb)
    if key in _PREPARE_FNS:
        return _PREPARE_FNS[key]

    @app.function(
        image=prepare_image,
        volumes={ARTIFACTS_PATH: artifacts, HF_CACHE_PATH: hf_cache},
        secrets=[secret],
        gpu=gpu,
        cpu=8.0,
        memory=memory_mb,
        timeout=60 * MINUTES,
    )
    def prepare_variant(*, tinker_path: str, base_model: str, name: str) -> str:
        return _merge_checkpoint(tinker_path=tinker_path, base_model=base_model, name=name)

    _PREPARE_FNS[key] = prepare_variant
    return prepare_variant


for cfg in MODEL_REGISTRY.values():
    _prepare_fn(cfg.gpu, cfg.memory_mb)
prepare = _prepare_fn("H100", 65536)


@app.local_entrypoint()
def main(tinker_path: str, base_model: str, name: str) -> None:
    config = model_config(base_model)
    output_path = _prepare_fn(config.gpu, config.memory_mb).remote(
        tinker_path=tinker_path, base_model=base_model, name=name
    )
    print(f"\nArtifact on the tinker-artifacts Volume: {output_path}")
    print(
        f"Serve it:\n  FINETUNE={name} MODEL={base_model} modal deploy -m tinker_cookbook.inference.modal.serve"
    )
