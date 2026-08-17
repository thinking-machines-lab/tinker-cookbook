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

_PREPARE_COMMON = {
    "image": prepare_image,
    "volumes": {ARTIFACTS_PATH: artifacts, HF_CACHE_PATH: hf_cache},
    "secrets": [secret],
    "cpu": 8.0,
    "timeout": 60 * MINUTES,
}


def _merge_checkpoint(*, tinker_path: str, base_model: str, name: str) -> str:
    from tinker_cookbook import weights

    output_path = artifact_dir(name)
    downloaded = weights.download(tinker_path=tinker_path, output_dir="/tmp/checkpoint")
    weights.build_hf_model(base_model=base_model, adapter_path=downloaded, output_path=output_path)

    artifacts.commit()
    print(f"[prepare] merged model ready at {output_path}")
    return output_path


@app.function(gpu="H100", memory=65536, **_PREPARE_COMMON)
def prepare(*, tinker_path: str, base_model: str, name: str) -> str:
    return _merge_checkpoint(tinker_path=tinker_path, base_model=base_model, name=name)


@app.function(gpu="H100:2", memory=65536, **_PREPARE_COMMON)
def prepare_h100_2(*, tinker_path: str, base_model: str, name: str) -> str:
    return _merge_checkpoint(tinker_path=tinker_path, base_model=base_model, name=name)


@app.function(gpu="H100:4", memory=65536, **_PREPARE_COMMON)
def prepare_h100_4(*, tinker_path: str, base_model: str, name: str) -> str:
    return _merge_checkpoint(tinker_path=tinker_path, base_model=base_model, name=name)


@app.function(gpu="H100:8", memory=65536, **_PREPARE_COMMON)
def prepare_h100_8(*, tinker_path: str, base_model: str, name: str) -> str:
    return _merge_checkpoint(tinker_path=tinker_path, base_model=base_model, name=name)


@app.function(gpu="H100:8", memory=131072, **_PREPARE_COMMON)
def prepare_h100_8_128g(*, tinker_path: str, base_model: str, name: str) -> str:
    return _merge_checkpoint(tinker_path=tinker_path, base_model=base_model, name=name)


_PREPARE_FNS: dict[tuple[str, int], modal.Function] = {
    ("H100:1", 65536): prepare,
    ("H100:2", 65536): prepare_h100_2,
    ("H100:4", 65536): prepare_h100_4,
    ("H100:8", 65536): prepare_h100_8,
    ("H100:8", 131072): prepare_h100_8_128g,
}


@app.local_entrypoint()
def main(tinker_path: str, base_model: str, name: str) -> None:
    config = model_config(base_model)
    try:
        fn = _PREPARE_FNS[(config.gpu, config.memory_mb)]
    except KeyError:
        known = ", ".join(f"{gpu}/{mem}MB" for gpu, mem in sorted(_PREPARE_FNS))
        raise KeyError(
            f"No prepare function for gpu={config.gpu!r} memory_mb={config.memory_mb}. Known: {known}"
        ) from None
    output_path = fn.remote(tinker_path=tinker_path, base_model=base_model, name=name)
    print(f"\nArtifact on the tinker-artifacts Volume: {output_path}")
    print(
        f"Serve it:\n  FINETUNE={name} MODEL={base_model} modal deploy -m tinker_cookbook.inference.modal.serve"
    )
