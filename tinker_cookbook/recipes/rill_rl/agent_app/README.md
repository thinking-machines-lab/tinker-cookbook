# RILL Coding Agent (standalone app)

A self-contained AI coding agent for **RILL**, a small out-of-distribution DSL. Given a
task, the agent asks an OpenAI chat model to write a RILL program, runs it through the
reference interpreter, and — if it fails to run — shows the model the interpreter error
and asks it to fix it, up to a few attempts.

It's a plain chat app: configured with `OPENAI_BASE_URL` / `OPENAI_API_KEY`, depends only
on `openai` + `lark` + `fastapi`, and imports nothing from tinker-cookbook. The agent uses
the interpreter as a tool to self-correct on errors; it has no answer key and computes no
reward. That's deliberate — it's a stand-in for an agent already in production.

## Run it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
# For a non-OpenAI / self-hosted model, also set OPENAI_BASE_URL.
./run.sh          # serves http://localhost:8000
```

In the UI: pick an example task (or write your own), hit **Run agent**, and watch each
attempt stream in — the program it wrote, the interpreter's output or error, and the fix
request it gets before the next attempt.

## Endpoints

- `GET /` — the web UI.
- `GET /api/examples` — demo task prompts.
- `POST /solve` — run the agent on one task; returns the final program, its interpreter
  output, whether it ran cleanly, and the turn count. Accepts a per-request model backend
  (`openai_base_url` / `openai_api_key` / `model`).
- `POST /api/run` — same as `/solve` but streams each attempt over Server-Sent Events.

The per-request backend on `/solve` is the integration seam: point it at any
OpenAI-compatible endpoint to drive a different model. The post-training setup in
[`../training/`](../training/) uses exactly this — each RL rollout calls `/solve` pointed
at a sampling proxy that serves the policy being trained.

## What's inside

- `agent.py` — `RillAgent`: the chat loop with interpreter-driven self-correction.
- `program.py` — extract the RILL program from a model message.
- `prompts.py` — the RILL language reference shown to the model.
- `rill_lang/` — the vendored RILL interpreter and grammar (the language spec).
- `server.py` — FastAPI backend; `static/index.html` — the UI.
