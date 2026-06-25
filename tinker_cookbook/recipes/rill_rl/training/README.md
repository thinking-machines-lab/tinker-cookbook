# RILL post-training (GRPO via a sampling proxy)

This package post-trains an open model **for the production agent app** in
[`../agent_app/`](../agent_app/) without reimplementing the agent. It drives the app
through its HTTP endpoints and serves the policy through an OpenAI-compatible proxy that
records tokens. See the [recipe README](../README.md) for the architecture diagram.

## Pieces

- **`proxy.py`** — `SamplingProxy`: an OpenAI-compatible endpoint backed by a Tinker
  `SamplingClient`. It renders chat requests with the model's renderer, samples, returns a
  `ChatCompletion`, and records `(prompt_tokens, sampled_tokens, logprobs)` per
  `rollout_id` (route: `POST /v1/<rollout_id>/chat/completions`). The trainer runs it
  embedded and controls it in-process (`set_policy`, `pop_captures`).
- **`train.py`** — the GRPO loop. Each step: point the proxy at the current policy, fire
  `group_size` rollouts per task at the app's `/solve` (each with a unique proxy URL), pull
  the captured tokens, grade the final programs against held-out expected outputs, center
  rewards within each group, build one datum per turn weighted by the rollout's advantage,
  and run `forward_backward` + `optim_step`.
- **`grading.py`** — the shaped reward, run against the task's expected output.
- **`tasks.py`** — task-family generators (reference solutions run through the interpreter)
  and the disjoint train/eval family split.
- **`eval.py`** — score the app on held-out families via `/solve`; works for the frontier
  baseline or a served checkpoint.

## Run

The app is the production service; start it first (see `../agent_app/README.md`). Then:

```bash
pip install 'tinker-cookbook[rill-rl]'
python -m tinker_cookbook.recipes.rill_rl.training.train \
    model_name=Qwen/Qwen3.5-4B app_url=http://127.0.0.1:8000 \
    group_size=8 groups_per_batch=32 max_turns=3
```

Requires `TINKER_API_KEY` (and network) for the training client. The proxy binds
`proxy_host:proxy_port` (default `127.0.0.1:8100`); the app must be able to reach it.
