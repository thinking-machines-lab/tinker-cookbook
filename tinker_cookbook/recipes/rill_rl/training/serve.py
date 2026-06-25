"""Serve the RILL agent app backed by a Tinker policy, with live checkpoint switching.

Runs the sampling proxy (Tinker-backed) and the production agent app together, and adds two
endpoints to the app so you can A/B test checkpoints live in the UI:

    GET  /api/checkpoints        list the project's RILL checkpoints (+ the base model)
    POST /api/select_checkpoint  point the served policy at a chosen checkpoint

The agent app stays model-agnostic (it just calls OpenAI at OPENAI_BASE_URL, which we point
at the proxy); this launcher is the training-side control plane that knows about Tinker.

    python -m tinker_cookbook.recipes.rill_rl.training.serve \\
        --project 8a4e2d8a-1b44-446e-88a7-ff9eaef978c9 --app-port 8300
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
import time

import tinker
import uvicorn
from pydantic import BaseModel

from tinker_cookbook import model_info, renderers
from tinker_cookbook.recipes.rill_rl.agent_app import server as appserver
from tinker_cookbook.recipes.rill_rl.training.checkpoints import list_project_checkpoints
from tinker_cookbook.recipes.rill_rl.training.proxy import SamplingProxy
from tinker_cookbook.tokenizer_utils import get_tokenizer

BASE = "base"  # sentinel for "serve the untrained base model"


class SelectRequest(BaseModel):
    tinker_path: str
    label: str = ""


class _Server:
    def __init__(self, model: str, project_id: str | None, temperature: float, max_tokens: int):
        self.model = model
        self.project_id = project_id
        self.temperature = temperature
        self.service = tinker.ServiceClient()
        renderer = renderers.get_renderer(
            model_info.get_recommended_renderer_name(model), get_tokenizer(model)
        )
        self.proxy = SamplingProxy(renderer, default_max_tokens=max_tokens)
        self.current = {"label": "", "tinker_path": ""}

    def select(self, tinker_path: str, label: str) -> None:
        if tinker_path == BASE:
            sc = self.service.create_sampling_client(base_model=self.model)
        else:
            sc = self.service.create_sampling_client(model_path=tinker_path, base_model=self.model)
        self.proxy.set_policy(sc, temperature=self.temperature)
        self.current = {"label": label, "tinker_path": tinker_path}

    def options(self) -> list[dict]:
        opts = [{"label": f"base ({self.model})", "tinker_path": BASE, "base_model": self.model}]
        if self.project_id:
            opts += list_project_checkpoints(self.project_id)
        return opts


def _serve_proxy(proxy: SamplingProxy, port: int) -> None:
    threading.Thread(
        target=lambda: uvicorn.run(proxy.app, host="127.0.0.1", port=port, log_level="warning"),
        daemon=True,
    ).start()
    time.sleep(1.5)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the RILL agent with live checkpoint A/B.")
    ap.add_argument("--project", default=os.environ.get("RILL_PROJECT_ID"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--checkpoint", default=BASE, help="initial tinker_path or 'base'")
    ap.add_argument("--app-port", type=int, default=8300)
    ap.add_argument("--proxy-port", type=int, default=8301)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    srv = _Server(args.model, args.project, args.temperature, args.max_tokens)
    _serve_proxy(srv.proxy, args.proxy_port)
    srv.select(args.checkpoint, "base" if args.checkpoint == BASE else args.checkpoint)

    # Point the agent app's default OpenAI client at the proxy.
    os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{args.proxy_port}/v1"
    os.environ["OPENAI_API_KEY"] = "tinker"

    app = appserver.app

    @app.get("/api/checkpoints")
    async def checkpoints():
        opts = await asyncio.to_thread(srv.options)
        return {"current": srv.current, "checkpoints": opts}

    @app.post("/api/select_checkpoint")
    async def select_checkpoint(req: SelectRequest):
        await asyncio.to_thread(srv.select, req.tinker_path, req.label or req.tinker_path)
        return {"ok": True, "current": srv.current}

    print(
        f"RILL agent on :{args.app_port} (proxy :{args.proxy_port}); "
        f"project={args.project}; serving {srv.current['label']}",
        flush=True,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.app_port, log_level="info")


if __name__ == "__main__":
    main()
