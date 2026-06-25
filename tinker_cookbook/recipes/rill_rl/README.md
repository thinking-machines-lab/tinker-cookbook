# Post-training a model for an agent you already have

You have a coding agent in production. It's an ordinary app built on OpenAI chat
completions: a system prompt, a loop that writes code, runs it, and fixes errors. Now you
want to drop in an open-weights model and have it match or beat the frontier model **in
that exact app**. The wrong way is to rebuild the agent inside a training environment.
The right way, and what this recipe shows: **point the app at a sampling proxy and train
through its own endpoints.**

The testbed is **RILL**, a small deterministic DSL built to be out-of-distribution —
ordinary semantics under unfamiliar surface syntax (`5 -> x`, `when/elsewhen/otherwise`,
`sustain`, `walk x across xs`, `forge`/`give`, `emit`, `~` comments). A model can't lean
on memorized patterns, so the gap between a frontier model and a small open model is real
and the climb is measurable.

## The integration: a proxy, not an import

```
                 OPENAI_BASE_URL = http://proxy/v1/<rollout_id>
   ┌──────────────────┐   /solve     ┌──────────────────┐  chat/completions  ┌──────────────────┐
   │     trainer      │ ───────────▶ │  production app  │ ─────────────────▶ │  sampling proxy  │
   │ (GRPO, this dir) │              │   (agent_app/)   │                    │  (tinker policy) │
   └──────────────────┘ ◀─────────── └──────────────────┘ ◀───────────────── └──────────────────┘
        ▲   reward            final program          OpenAI ChatCompletion        records tokens
        └──────────────── pulls captured tokens ◀──────────────────────────────────────┘
```

- The **app** ([`agent_app/`](./agent_app/)) is a normal standalone chat app. It imports
  nothing from tinker-cookbook and has no notion of reward, "step", or episodes. It's
  configured with `OPENAI_BASE_URL` / `OPENAI_API_KEY` like any OpenAI app.
- The **sampling proxy** ([`training/proxy.py`](./training/proxy.py)) is OpenAI-compatible.
  It renders each chat request with the model's renderer, samples from the Tinker policy
  being trained, returns a `ChatCompletion`, and **records the exact prompt tokens,
  sampled tokens, and logprobs**. Requests are routed per rollout (`/v1/<rollout_id>`), so
  every turn of a rollout is grouped without the app knowing anything.
- The **trainer** ([`training/train.py`](./training/train.py)) never imports the agent
  loop. It triggers rollouts by POSTing to the app's `/solve` endpoint, pulls the captured
  tokens from the proxy, grades each rollout's final program against the held-out expected
  output, computes GRPO advantages within each task's group, and steps the Tinker training
  client on the captured tokens.

Because the policy is trained on the exact tokens the app generated through the proxy,
training is perfectly aligned with how the model is actually used in production.

## Why RILL is a clean RL target

Deterministic (no clocks/RNG/IO, hard `max_steps`), so a bad sample can't stall a rollout.
Each task asks for a function `forge solve(...)`, and the grader calls it on **hidden
inputs not shown in the prompt** ([`training/tasks.py`](./training/tasks.py),
[`training/grading.py`](./training/grading.py)) — so a program that just emits a constant
fails (the Experiment 2 fix for the reward hack found in Experiment 1; see the
[logbook](./results/)). The reward is **shaped** (parses -> runs clean -> emits -> fraction
of hidden inputs correct), giving gradient before the program is fully right. Tasks are
built by running a reference `solve` on each hidden input for the exact expected output,
and we hold out **disjoint task families** for eval to measure transfer. The trainer holds
the hidden inputs and expected outputs; the app never sees them.

## Quickstart

### 1. Run the production agent

```bash
pip install -r tinker_cookbook/recipes/rill_rl/agent_app/requirements.txt
export OPENAI_API_KEY=sk-...
python -m tinker_cookbook.recipes.rill_rl.agent_app.server   # http://localhost:8000
```

Open the UI and watch a frontier model (default `gpt-5.5`) struggle on RILL even with
interpreter feedback. Or score it on the held-out families:

```bash
python -m tinker_cookbook.recipes.rill_rl.training.eval --app-url http://127.0.0.1:8000 --model gpt-5.5
```

### 2. Post-train an open model with GRPO

Keep the app running (it's the production service). Then:

```bash
pip install 'tinker-cookbook[rill-rl]'
python -m tinker_cookbook.recipes.rill_rl.training.train \
    model_name=Qwen/Qwen3.5-4B \
    app_url=http://127.0.0.1:8000 \
    group_size=8 groups_per_batch=32 \
    learning_rate=1e-5 lora_rank=32 \
    max_turns=3
```

The trainer starts the sampling proxy itself and drives the app over HTTP. `Qwen/Qwen3.5-9B`
gives more headroom at higher cost.

### 3. Score the trained checkpoint in the same app

Serve the checkpoint behind an OpenAI-compatible endpoint, then point the app at it (set
the app's `OPENAI_BASE_URL`, or pass `--openai-base-url` to the eval) and rerun the eval
from step 1. Same app, same harness, head-to-head numbers.

## Results

Measured on the held-out families (`gcd`, `nth_fib`, `palindrome`, n=120; temperature 1.0,
single sample, up to 3 turns of interpreter self-correction). The base model barely handles
the OOD language; 30 GRPO steps take it to near-perfect, and the gains transfer to families
it never trained on.

| Model | pass@1 (held-out families) | mean shaped reward |
|-------|----------------------------|--------------------|
| Frontier baseline (`gpt-5.5`)       | _pending an API key_ | _pending_ |
| `Qwen/Qwen3.5-4B` (before training) | 0.100 | 0.270 |
| `Qwen/Qwen3.5-4B` (after 30 steps)  | **0.967** | **0.979** |

Per-family pass@1, before → after:

| Family | before | after |
|--------|--------|-------|
| `gcd` (n=60)        | 0.100 | 1.000 |
| `nth_fib` (n=30)    | 0.133 | 0.867 |
| `palindrome` (n=30) | 0.067 | 1.000 |

Training config for the "after" row: `group_size=8`, `groups_per_batch=8`, 30 batches,
`learning_rate=4e-5`, `lora_rank=32`, `max_turns=2`. The training metric climbed from
pass@1 ≈ 0.41 (first 5 batches) to ≈ 0.98 (last 5).

> **Caveat — the policy reward-hacked (Experiment 1).** That 0.97 is *correct output*, not
> *learned to code*. Inspecting the rollouts
> ([`results/experiment_1/sample_rollouts.md`](./results/experiment_1/sample_rollouts.md)),
> 98% of trained completions are hardcoded constant emits (`emit 1`, `emit "yes"`): the
> model computes the answer in its reasoning and prints the literal. Each task is a single
> fixed instance with its inputs in the prompt and the reward only checks output, so
> emitting the constant is the cheapest win. The fix (grade a `solve(...)` against hidden
> inputs) is Experiment 2. The full running record is the logbook in
> [`results/`](./results/).

## Files

- [`agent_app/`](./agent_app/) — the standalone production agent (chat loop, interpreter,
  prompts, FastAPI server, UI). Its own README explains how to run it.
- [`training/proxy.py`](./training/proxy.py) — the OpenAI-compatible sampling + token-capture
  proxy backed by a Tinker policy.
- [`training/train.py`](./training/train.py) — the standalone GRPO trainer.
- [`training/eval.py`](./training/eval.py) — score the app on held-out tasks via `/solve`.
- [`training/tasks.py`](./training/tasks.py), [`training/grading.py`](./training/grading.py)
  — task families (with the answer key) and the shaped reward.
- [`rill_rl_test.py`](./rill_rl_test.py) — offline tests for every piece.

## Tuning the OOD pressure

The difficulty knob is how much surface syntax you swap. Harder: rename more builtins,
change block delimiters, alter how `walk` iterates, then regenerate the reference outputs.
Easier: move back toward mainstream syntax. Semantics stay fixed; only the surface moves.
