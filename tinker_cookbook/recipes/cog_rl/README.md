# Post-training a model for an agent you already have

You have a coding agent in production. It's an ordinary app built on OpenAI chat
completions: a system prompt, a loop that writes code, runs it, and fixes errors. Now you
want to drop in an open-weights model and have it match or beat the frontier model **in
that exact app**. The wrong way is to rebuild the agent inside a training environment.
The right way, and what this recipe shows: **point the app at a sampling proxy and train
through its own endpoints.**

The testbed is **Cog**, a small deterministic DSL built to be out-of-distribution —
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

## Why Cog is a clean RL target

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
pip install -r tinker_cookbook/recipes/cog_rl/agent_app/requirements.txt
export OPENAI_API_KEY=sk-...
python -m tinker_cookbook.recipes.cog_rl.agent_app.server   # http://localhost:8000
```

Open the UI and watch a frontier model (default `gpt-5.5`) struggle on Cog even with
interpreter feedback. Or score it on the held-out families:

```bash
python -m tinker_cookbook.recipes.cog_rl.training.eval --app-url http://127.0.0.1:8000 --model gpt-5.5
```

### 2. Post-train an open model with GRPO

Keep the app running (it's the production service). Then:

```bash
pip install 'tinker-cookbook[cog-rl]'
python -m tinker_cookbook.recipes.cog_rl.training.train \
    model_name=Qwen/Qwen3.5-4B \
    app_url=http://127.0.0.1:8000 \
    group_size=8 groups_per_batch=32 \
    learning_rate=1e-5 lora_rank=32 \
    max_turns=3
```

The trainer starts the sampling proxy itself and drives the app over HTTP. `Qwen/Qwen3.5-9B`
gives more headroom at higher cost. Pass `project_id=<id>` (and optionally `run_label=...`)
to group runs under a Tinker project so the app can list them (below).

### 3. Serve a trained checkpoint and A/B test live

`training/serve.py` runs the proxy + the app and adds a checkpoint picker to the UI. Point
it at a Tinker project and it lists that project's Cog checkpoints (tagged via
`user_metadata`); pick one to serve, run a prompt, switch to another, run the same prompt —
live A/B in the browser.

```bash
python -m tinker_cookbook.recipes.cog_rl.training.serve \
    --project <project-id> --app-port 8300 \
    --checkpoint tinker://<run>:train:0/sampler_weights/<name>   # or 'base'
```

Manage the project's checkpoints with `training/checkpoints.py` (copies never expire):

```bash
# copy a prior run's trainable weights into the project, tagged + permanent
python -m tinker_cookbook.recipes.cog_rl.training.checkpoints copy \
    --project <project-id> --label exp3 --source tinker://<run>:train:0/weights/final
# list what's in the project
python -m tinker_cookbook.recipes.cog_rl.training.checkpoints list --project <project-id>
```

To score a checkpoint on the held-out set, point the eval at the running app's proxy (or
any OpenAI-compatible endpoint) via `--openai-base-url` and rerun step 1's eval.

## Results

Two experiments so far; the full running record (config, rollouts, takeaways) is the
[logbook in `results/`](./results/). `Qwen/Qwen3.5-4B`, 30 GRPO steps, held-out disjoint
families, temperature 1.0.

**Experiment 1 — output-match reward.** Held-out pass@1 0.10 → 0.97 — but the policy
**reward-hacked**: 98% of trained completions are hardcoded constant emits (`emit 1`,
`emit "yes"`). Each task's inputs were fixed in the prompt and the reward only checked
output, so printing the literal answer was the cheapest win. Not learned-to-code.
([`results/experiment_1/`](./results/experiment_1/))

**Experiment 2 — grade `solve(...)` on hidden inputs.** Tasks now ask for a function and
the grader runs it on inputs the model never sees, so a constant can't win.

| `Qwen/Qwen3.5-4B` | held-out pass@1 | mean reward | trained programs that are real functions |
|---|---|---|---|
| before training | 0.200 | 0.340 | — |
| after 30 steps  | **0.550** | **0.671** | **20/20** (0 constant emits) |

Lower than Experiment 1's *fake* 0.97 because it now measures generalizing programs, not
memorized literals. The model genuinely learned Cog syntax — e.g. it switched from an
invalid `c = chars(w)` to a correct `chars(w) -> c` with `w @ i` indexing.
([`results/experiment_2/`](./results/experiment_2/))

**Experiment 3 — async off-policy loop, 50 steps, 10 held-out families.**

| `Qwen/Qwen3.5-4B` | held-out pass@1 | mean reward | real functions |
|---|---|---|---|
| before training | 0.250 | 0.418 | — |
| after 50 steps  | **0.750** | **0.838** | **40/40** (0 constants) |

The Exp-2 weak spots improved (`reverse_text` 0.00→1.00, `gcd` 0.00→0.75, `is_sorted`
0.00→0.75). The async loop (`max_steps_off_policy`) overlaps sampling with the optimizer
step and is bounded-correct, but at this scale a LoRA step is as fast as sampling, so the
sampler never ran ahead (`lag` stayed 0) — i.e. effectively on-policy. The staleness path
would engage with a slower trainer or longer generations.
([`results/experiment_3/`](./results/experiment_3/))

**Experiment 4 — enrich the train mix for the lagging families.** Added 5 structure-matched
train families (list reduce, accumulate/while-with-branch, two-index text scan; eval still
disjoint), 60 steps. The matched held-out families lifted — `list_max` 0.50→**1.00**,
`palindrome` 0.50→**1.00**, `lcm` 0.50→**0.75**, `is_sorted` 0.75→**1.00** — and mean reward
rose 0.838→**0.904**, but overall pass@1 was flat (0.75→**0.775**): the held-out set is only
n=4/family, so unrelated families swing within noise (and `nth_fib` stayed hardest). Lesson:
structure-matched data transfers to matched structure, but read per-family deltas at n=4
with caution — a bigger eval is the real next step.
([`results/experiment_4/`](./results/experiment_4/))

**Experiment 5 — two-state recurrence family + a stable (16-sample) eval.** Added a `lucas`
train family (same two-state recurrence as the held-out `nth_fib`, different seeds) and
re-measured at 16 samples/family (n=160). On the stable eval, **overall pass@1 0.275 →
0.944** (mean reward 0.419 → 0.974), with every completion a real `solve(...)` (0 constants).
`nth_fib` moved 0.19 → 0.62 (the matched recurrence transferred; still the hardest), and the
Exp-4 "regressions" (`gcd`, `power`, `reverse_text`) turned out to be n=4 noise — all 0.94–1.00
here. This is the recipe's headline result. ([`results/experiment_5/`](./results/experiment_5/))

The frontier baseline (`gpt-5.5`) row is still _pending an API key_.

## Files

- [`agent_app/`](./agent_app/) — the standalone production agent (chat loop, interpreter,
  prompts, FastAPI server, UI). Its own README explains how to run it.
- [`training/proxy.py`](./training/proxy.py) — the OpenAI-compatible sampling + token-capture
  proxy backed by a Tinker policy.
- [`training/train.py`](./training/train.py) — the standalone GRPO trainer.
- [`training/eval.py`](./training/eval.py) — score the app on held-out tasks via `/solve`.
- [`training/tasks.py`](./training/tasks.py), [`training/grading.py`](./training/grading.py)
  — task families (with the answer key) and the shaped reward.
- [`cog_rl_test.py`](./cog_rl_test.py) — offline tests for every piece.

## Tuning the OOD pressure

The difficulty knob is how much surface syntax you swap. Harder: rename more builtins,
change block delimiters, alter how `walk` iterates, then regenerate the reference outputs.
Easier: move back toward mainstream syntax. Semantics stay fixed; only the surface moves.
