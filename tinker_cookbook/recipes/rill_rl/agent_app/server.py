"""FastAPI backend + UI for the RILL coding agent.

Endpoints:
- ``GET  /``              the web UI
- ``GET  /api/examples``  demo task prompts for the UI
- ``POST /solve``         run the agent on one task, return the final program + output
- ``POST /api/run``       same, but stream each attempt to the browser over SSE

``/solve`` accepts a per-request model backend (``openai_base_url`` / ``openai_api_key`` /
``model``), so the app can be pointed at any OpenAI-compatible endpoint. That's how
post-training drives it: each rollout calls ``/solve`` pointed at a sampling proxy.

Run it::

    export OPENAI_API_KEY=sk-...            # and OPENAI_BASE_URL for a non-OpenAI endpoint
    python -m tinker_cookbook.recipes.rill_rl.agent_app.server   # http://localhost:8000
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import RillAgent
from .examples import EXAMPLE_PROMPTS

_STATIC = Path(__file__).parent / "static"
app = FastAPI(title="RILL Coding Agent")


class SolveRequest(BaseModel):
    prompt: str
    model: str = "gpt-5.5"
    max_turns: int = 3
    temperature: float | None = None
    max_completion_tokens: int | None = None
    # Per-request backend override (bring-your-own OpenAI-compatible endpoint).
    openai_base_url: str | None = None
    openai_api_key: str | None = None


def _agent(req: SolveRequest) -> RillAgent:
    return RillAgent(
        model=req.model,
        max_turns=req.max_turns,
        base_url=req.openai_base_url,
        api_key=req.openai_api_key,
        temperature=req.temperature,
        max_completion_tokens=req.max_completion_tokens,
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/api/examples")
async def examples() -> dict:
    return {"prompts": EXAMPLE_PROMPTS}


@app.post("/solve")
async def solve(req: SolveRequest) -> dict:
    result = await _agent(req).solve(req.prompt)
    return {
        "prompt": result.prompt,
        "program": result.program,
        "output": result.output,
        "error": result.error,
        "ran_clean": result.ran_clean,
        "turns": result.turns,
        "api_error": result.api_error,
        "transcript": result.transcript,
    }


@app.post("/api/run")
async def run(req: SolveRequest) -> StreamingResponse:
    async def gen():
        async for ev in _agent(req).iter_solve(req.prompt):
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


def main() -> None:
    import os

    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("RILL_HOST", "0.0.0.0"),
        port=int(os.environ.get("RILL_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
