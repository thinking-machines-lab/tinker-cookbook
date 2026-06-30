"""PREPARE: a tinker:// checkpoint -> a servable artifact on the Volume.

    modal run prepare.py \\
        --tinker-path tinker://<run-id>/sampler_weights/<name> \\
        --base-model Qwen/Qwen3-8B --name my-finetune [--mode adapter|merge]

Omit --mode to use the model's default. Also callable from another app via
prepare.remote(...).
"""

from __future__ import annotations

from typing import Callable

import modal

from .common import (
    HF_CACHE_PATH,
    MINUTES,
    ARTIFACTS_PATH,
    app,
    artifact_dir,
    artifacts,
    hf_cache,
    model_config,
    prepare_image,
    resolve_mode,
)

tinker_secret = modal.Secret.from_name("tinker")  # TINKER_API_KEY
hf_secret = modal.Secret.from_name("huggingface")  # HF_TOKEN for gated base models/faster downloads


def _builder(weights, mode: str) -> Callable[..., None]:
    # Both share the (base_model, adapter_path, output_path) signature
    return weights.build_lora_adapter if mode == "adapter" else weights.build_hf_model


@app.function(
    image=prepare_image,
    volumes={ARTIFACTS_PATH: artifacts, HF_CACHE_PATH: hf_cache},
    secrets=[tinker_secret, hf_secret],
    gpu="H100",  # merge runs the dequant/requant math on GPU (build_hf_model auto-uses cuda)
    cpu=8.0,
    memory=65536,
    timeout=60 * MINUTES,
)
def prepare(
    *,
    tinker_path: str,
    base_model: str,
    name: str,
    mode: str | None = None,
) -> str:
    from tinker_cookbook import weights

    cfg = model_config(base_model)
    resolved = resolve_mode(cfg, mode)
    output_path = artifact_dir(name)

    downloaded = weights.download(tinker_path=tinker_path, output_dir="/tmp/checkpoint")
    build = _builder(weights, resolved)
    build(base_model=base_model, adapter_path=downloaded, output_path=output_path)

    artifacts.commit()
    print(f"[prepare] {resolved} artifact ready at {output_path}")
    return output_path


@app.local_entrypoint()
def main(tinker_path: str, base_model: str, name: str, mode: str = "") -> None:
    output_path = prepare.remote(
        tinker_path=tinker_path, base_model=base_model, name=name, mode=mode or None
    )
    print(f"\nArtifact on the tinker-artifacts Volume: {output_path}")
    print(f"Serve it:\n  FINETUNE={name} MODEL={base_model} modal deploy serve.py")
