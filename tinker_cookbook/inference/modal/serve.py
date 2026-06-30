"""SERVE: an OpenAI-compatible SGLang endpoint for a prepared artifact

    FINETUNE=my-finetune MODEL=Qwen/Qwen3-8B modal deploy serve.py   # persistent
    FINETUNE=my-finetune MODEL=Qwen/Qwen3-8B modal run    serve.py   # smoke test
"""

from __future__ import annotations

import os
import subprocess
import time

import modal
import modal.experimental

from .common import (
    HF_CACHE_PATH,
    MINUTES,
    ARTIFACTS_PATH,
    app,
    artifact_dir,
    artifacts,
    hf_cache,
    model_config,
    resolve_mode,
    sglang_command,
    sglang_image,
)

PORT = 30000
PROXY_REGION = "us-west"
TARGET_INPUTS = 10

FINETUNE = os.environ["FINETUNE"]
MODEL = os.environ["MODEL"]
MODE_OVERRIDE = os.environ.get("MODE") or None
CONFIG = model_config(MODEL)
MODE = resolve_mode(CONFIG, MODE_OVERRIDE)

serve_image = sglang_image.env(
    {"FINETUNE": FINETUNE, "MODEL": MODEL} | ({"MODE": MODE_OVERRIDE} if MODE_OVERRIDE else {})
)

hf_secret = modal.Secret.from_name("huggingface")

with serve_image.imports():
    import requests


def launch_argv() -> tuple[str, ...]:
    path = artifact_dir(FINETUNE)
    if MODE == "adapter":  # serve base model + attach the adapter
        return sglang_command(
            model_path=CONFIG.base_model, served_name=FINETUNE,
            tp=CONFIG.tp, port=PORT, lora=(FINETUNE, path),
        )
    return sglang_command(  # serve the merged dir directly
        model_path=path, served_name=FINETUNE, tp=CONFIG.tp, port=PORT
    )


def _healthy() -> bool:
    try:
        requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
        return True
    except (requests.ConnectionError, requests.HTTPError):
        return False


def wait_until_ready(proc: subprocess.Popen, timeout: int = 20 * MINUTES) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if (rc := proc.poll()) is not None:
            raise subprocess.CalledProcessError(rc, proc.args)
        if _healthy():
            return
        time.sleep(5)
    raise TimeoutError(f"SGLang server not ready within {timeout}s")


def warmup(rounds: int = 3) -> None:
    payload = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16}
    for _ in range(rounds):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=30
        ).raise_for_status()


@app.cls(
    image=serve_image,
    gpu=CONFIG.gpu,
    volumes={ARTIFACTS_PATH: artifacts, HF_CACHE_PATH: hf_cache},
    secrets=[hf_secret],
    scaledown_window=10 * MINUTES,
    startup_timeout=20 * MINUTES,
)
@modal.experimental.http_server(port=PORT, proxy_regions=[PROXY_REGION], exit_grace_period=15)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class Server:
    @modal.enter()
    def start(self) -> None:
        self.proc = subprocess.Popen(launch_argv(), env=os.environ.copy())
        wait_until_ready(self.proc)
        warmup()

    @modal.exit()
    def stop(self) -> None:
        self.proc.terminate()


@app.local_entrypoint()
async def test(prompt: str = "What is Modal?", timeout: int = 10 * MINUTES) -> None:
    import asyncio
    import json

    import aiohttp

    url = (await Server._experimental_get_flash_urls.aio())[0]
    headers = {"Modal-Session-ID": "0"}
    body = {"model": FINETUNE, "messages": [{"role": "user", "content": prompt}], "stream": True}
    deadline = time.time() + timeout

    async with aiohttp.ClientSession(base_url=url, headers=headers) as session:
        while time.time() < deadline:
            try:
                async with session.post("/v1/chat/completions", json=body) as resp:
                    resp.raise_for_status()
                    async for raw in resp.content:
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        delta = (json.loads(data)["choices"] or [{}])[0].get("delta", {})
                        if chunk := delta.get("content"):
                            print(chunk, end="", flush=True)
                    print()
                    return
            except aiohttp.ClientResponseError as exc:
                if exc.status == 503: 
                    await asyncio.sleep(1)
                    continue
                raise
    raise TimeoutError(f"No response within {timeout}s")
