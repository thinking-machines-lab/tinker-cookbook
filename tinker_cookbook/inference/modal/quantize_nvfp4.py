"""QUANTIZE: merged BF16 Inkling checkpoint -> NVFP4 experts-only for B200:2 serving.

    modal run --detach -m tinker_cookbook.inference.modal.quantize_nvfp4 \
        --merged-name waldo-v3 --output-name waldo-v3-nvfp4

Reads the merged model from the tinker-artifacts Volume (output of prepare.py),
runs NVIDIA Model Optimizer PTQ with nvfp4_experts_only, and writes the result
back to the Volume.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request

import modal

from .common import (
    ARTIFACTS_PATH,
    HF_CACHE_PATH,
    MINUTES,
    app,
    artifact_dir,
    artifacts,
    hf_cache,
)

HF_PTQ = "/opt/Model-Optimizer/examples/hf_ptq/hf_ptq.py"
INKLING_NVFP4_HF_QUANT_URL = (
    "https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4/resolve/main/hf_quant_config.json"
)

quantize_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "accelerate",
        "safetensors",
        "torch",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/NVIDIA/Model-Optimizer.git /opt/Model-Optimizer",
        "pip install -e '/opt/Model-Optimizer'",
        "pip install --upgrade 'transformers>=5.14.0'",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "HF_TRUST_REMOTE_CODE": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
)


def _copy_inkling_hf_quant_config(output_path: str) -> None:
    """Inkling NVFP4 checkpoints use hf_quant_config.json from the official release."""
    dest = os.path.join(output_path, "hf_quant_config.json")
    if os.path.exists(dest):
        return
    with urllib.request.urlopen(INKLING_NVFP4_HF_QUANT_URL, timeout=120) as resp:
        config = json.load(resp)
    with open(dest, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


@app.function(
    image=quantize_image,
    gpu="H100:8",
    memory=131072,
    cpu=8.0,
    timeout=120 * MINUTES,
    volumes={ARTIFACTS_PATH: artifacts, HF_CACHE_PATH: hf_cache},
)
def quantize_nvfp4(
    *,
    merged_name: str = "waldo-v3",
    output_name: str = "waldo-v3-nvfp4",
    calib_size: int = 256,
    calib_seq: int = 2048,
) -> str:
    merged_path = artifact_dir(merged_name)
    output_path = artifact_dir(output_name)
    if not os.path.isdir(merged_path):
        raise FileNotFoundError(f"Merged model not found at {merged_path}")
    if os.path.exists(output_path):
        raise FileExistsError(
            f"Output already exists: {output_path}. Delete on the volume or pick another name."
        )

    cmd = [
        sys.executable,
        HF_PTQ,
        "--pyt_ckpt_path",
        merged_path,
        "--qformat",
        "nvfp4_experts_only",
        "--export_path",
        output_path,
        "--trust_remote_code",
        "--calib_size",
        str(calib_size),
        "--calib_seq",
        str(calib_seq),
        "--kv_cache_qformat",
        "none",
        "--low_memory_mode",
    ]
    print("[quantize_nvfp4] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    _copy_inkling_hf_quant_config(output_path)

    artifacts.commit()
    print(f"[quantize_nvfp4] NVFP4 model ready at {output_path}")
    return output_path


@app.local_entrypoint()
def main(
    merged_name: str = "waldo-v3",
    output_name: str = "waldo-v3-nvfp4",
    calib_size: int = 256,
    calib_seq: int = 2048,
) -> None:
    output_path = quantize_nvfp4.remote(
        merged_name=merged_name,
        output_name=output_name,
        calib_size=calib_size,
        calib_seq=calib_seq,
    )
    print(f"\nNVFP4 artifact on tinker-artifacts Volume: {output_path}")
