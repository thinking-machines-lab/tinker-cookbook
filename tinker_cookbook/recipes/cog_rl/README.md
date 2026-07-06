# Post-training a model for an agent you already have

You have a coding agent in production. It's an ordinary app built on OpenAI chat
completions: a system prompt, a loop that writes code, runs it, and fixes errors. Now you
want to drop in an open-weights model and have it hold its own against much larger models
**in that exact app**. The wrong way is to rebuild the agent inside a training
environment. The right way, and what this recipe shows: point the app at a sampling proxy
and train through its own endpoints.

The testbed is **Cog**, a small deterministic DSL built to be out-of-distribution:
ordinary semantics under unfamiliar surface syntax (`5 -> x`, `when/elsewhen/otherwise`,
`sustain`, `walk x across xs`, `forge`/`give`, `emit`, `~` comments). A model can't lean
on memorized patterns, so the gap between a frontier model and a small open model is real
and the climb is measurable. Everything here (interpreter, grading, app, trainer) is
self-contained and runs offline except the Tinker calls.

## The integration: a proxy, not an import

```
                 OPENAI_BASE_URL = http://proxy/v1/<rollout_id>
   +------------------+   /solve     +------------------+  chat/completions  +------------------+
   |     trainer      | -----------> |  production app  | -----------------> |  sampling proxy  |
   | (GRPO, this dir) |              |   (agent_app/)   |                    |  (tinker policy) |
   +------------------+ <----------- +------------------+ <----------------- +------------------+
        ^   reward            final program          OpenAI ChatCompletion        records tokens
        +---------------- pulls captured tokens <---------------------------------------+
```

- The **app** ([`agent_app/`](./agent_app/)) is a normal standalone chat app. It imports
  nothing from tinker-cookbook and has no notion of reward, "step", or episodes. It's
  configured with `OPENAI_BASE_URL` / `OPENAI_API_KEY` like any OpenAI app.
- The **sampling proxy** ([`training/proxy.py`](./training/proxy.py)) is OpenAI-compatible.
  It renders each chat request with the model's renderer, samples from the Tinker policy
  being trained, returns a `ChatCompletion`, and records the exact prompt tokens, sampled
  tokens, and logprobs. Requests are routed per rollout (`/v1/<rollout_id>`), so every turn
  of a rollout is grouped without the app knowing anything.
- The **trainer** ([`training/train.py`](./training/train.py)) never imports the agent
  loop. It triggers rollouts by POSTing to the app's `/solve` endpoint, pulls the captured
  tokens from the proxy, grades each rollout's final program, computes GRPO advantages
  within each task's group, and steps the Tinker training client on the captured tokens.

Because the policy is trained on the exact tokens the app generated through the proxy,
training is perfectly aligned with how the model is used in production. Nothing about the
pattern is Cog-specific: any OpenAI-chat-completions agent with a triggerable endpoint and
a gradeable outcome can be trained this way.

## The reward: hidden inputs, shaped credit

Each task asks for a function `forge solve(...)` and the grader runs it on inputs the
model never sees ([`training/grading.py`](./training/grading.py)). This matters: our first
run graded visible output and the policy promptly reward-hacked it, emitting hardcoded
constants for a fake 0.97 (the full story is in the [logbook](./results/)). Hidden inputs
make constants worthless. The reward is shaped (parses, runs clean, emits, fraction of
hidden inputs correct) so there's gradient before a program is fully right, and Cog is
deterministic with a hard step budget, so a bad sample can't stall a rollout.

Tasks come from two sources ([`training/tasks.py`](./training/tasks.py)):

- **Hand-authored families** with reference solutions run through the interpreter for
  exact expected outputs, split into disjoint train/eval families to measure transfer.
- **A corpus source with zero hand-written Cog**
  ([`training/corpus_tasks.py`](./training/corpus_tasks.py)): take MBPP and HumanEval,
  execute each problem's *Python* reference on its own test inputs, keep the problems
  whose values Cog can represent, and grade Cog programs against that expected I/O
  ([`training/cog_format.py`](./training/cog_format.py) matches the interpreter's printing
  byte for byte). The verifier doesn't care what language the reference was written in,
  which is what makes the task supply scale.

## Quickstart

### 1. Run the production agent

```bash
pip install -r tinker_cookbook/recipes/cog_rl/agent_app/requirements.txt
export OPENAI_API_KEY=sk-...
python -m tinker_cookbook.recipes.cog_rl.agent_app.server   # http://localhost:8000
```

Open the UI and watch a frontier model work through Cog with interpreter feedback, or
score it on the held-out set:

```bash
python -m tinker_cookbook.recipes.cog_rl.training.eval \
    --app-url http://127.0.0.1:8000 --model gpt-5.5 --task-source corpus --repeat 5
```

### 2. Post-train an open model with GRPO

Keep the app running (it's the production service). Then:

```bash
pip install 'tinker-cookbook[cog-rl]'
python -m tinker_cookbook.recipes.cog_rl.training.train \
    model_name=Qwen/Qwen3.5-9B \
    task_source=corpus \
    app_url=http://127.0.0.1:8000 \
    group_size=8 groups_per_batch=16 \
    learning_rate=8e-5 lora_rank=32 \
    max_turns=3
```

The trainer starts the sampling proxy itself and drives the app over HTTP. Notes that
save real time (details in [`results/FINDINGS.md`](./results/FINDINGS.md)):

- **Warm-start before RL.** Harvest verified solutions from the model itself with
  [`training/curate.py`](./training/curate.py) (rejection sampling against the grader),
  SFT on them with [`training/sft.py`](./training/sft.py), then pass that checkpoint as
  `init_state_path=` to `train.py`. RL from a warm start spends its steps on correctness
  instead of syntax discovery.
- **Learning rates are per-method.** `hyperparam_utils.get_lr` is right for SFT and
  on-policy distillation but GRPO collapses there; ~8e-5 worked for GRPO on these models.
- `task_source=corpus+he` adds the HumanEval-derived tasks to training; the held-out eval
  set stays fixed.

### 3. Serve a trained checkpoint and A/B test live

`training/serve.py` runs the proxy + the app and adds a checkpoint picker to the UI. Point
it at a Tinker project and it lists that project's checkpoints; pick one, run a prompt,
switch, run the same prompt again: live A/B in the browser.

```bash
python -m tinker_cookbook.recipes.cog_rl.training.serve \
    --project <project-id> --app-port 8300 \
    --checkpoint tinker://<run>:train:0/sampler_weights/<name>   # or 'base'
```

Manage the project's checkpoints with `training/checkpoints.py` (copy with no expiry,
list with created-at timestamps).

### 4. Squeeze the deployed agent at test time

Two agentic techniques, both flags on the eval and both honest (the shown example is
excluded from grading):

```bash
python -m tinker_cookbook.recipes.cog_rl.training.eval \
    --app-url http://127.0.0.1:8300 --model my-checkpoint --task-source corpus \
    --show-example --best-of 4
```

`--show-example` puts one input/output pair in the prompt (fixes task-misreads);
`--best-of N` retries until the program passes that visible example (self-verification).
Together they lifted the trained 9B from 0.626 to 0.768 on the hidden-only protocol. They
lift frontier models about equally, so treat them as product technique, not a training
result.

## Results

Held-out corpus problems (disjoint from training), 5 samples per task. The full ladder,
the per-method learning-rate findings, and the failure analysis are in
[`results/FINDINGS.md`](./results/FINDINGS.md); the experiment-by-experiment record
(including the reward hack and its fix) is the [logbook](./results/).

| model | held-out pass@1 |
|---|---|
| Qwen3.5-4B base | 0.283 |
| Qwen3.5-4B trained (self-distilled SFT, then GRPO) | 0.513 |
| Qwen3.5-9B base | 0.347 |
| Qwen3.5-9B trained (SFT, on-policy distillation, then GRPO) | **0.618** |
| Kimi K2.6 untrained (reference) | 0.616 |
| Qwen3.5-397B untrained (reference) | 0.620 |
| Qwen3.5-397B trained with this recipe | 0.673 |
| gpt-5.5 (frontier reference) | 0.751 |

The trained 9B lands level with untrained 400B-class open models in this app, and with
`--show-example --best-of 4` the deployed agent scores what plain gpt-5.5 scores (0.768 on
that protocol) at a fraction of the serving cost.

Training is fully self-contained: the SFT data is rejection-sampled from the model being
trained and verified by the interpreter, so no frontier model appears anywhere in the
training path (frontier numbers above are evaluation references).

## Files

- [`agent_app/`](./agent_app/) — the standalone production agent (chat loop, vendored Cog
  interpreter, prompts, FastAPI server, UI). Its own README explains how to run it.
- [`training/proxy.py`](./training/proxy.py) — the OpenAI-compatible sampling + token-capture
  proxy backed by a Tinker policy.
- [`training/train.py`](./training/train.py) — the standalone GRPO trainer (sync or async
  off-policy via `max_steps_off_policy`; warm-start via `init_state_path`).
- [`training/tasks.py`](./training/tasks.py) / [`training/corpus_tasks.py`](./training/corpus_tasks.py)
  / [`training/cog_format.py`](./training/cog_format.py) — task sources and the Cog-value
  adapter.
- [`training/grading.py`](./training/grading.py) — the hidden-input shaped reward.
- [`training/curate.py`](./training/curate.py) / [`training/sft.py`](./training/sft.py) —
  rejection-sampled gold harvesting and the SFT warm-start.
- [`training/eval.py`](./training/eval.py) — score any backend through the app, with the
  test-time flags.
- [`training/serve.py`](./training/serve.py) / [`training/checkpoints.py`](./training/checkpoints.py)
  — live checkpoint A/B in the UI and project checkpoint management.
- [`cog_rl_test.py`](./cog_rl_test.py) — offline tests for every piece (no API key needed).

## Tuning the OOD pressure

The difficulty knob is how much surface syntax you swap. Harder: rename more builtins,
change block delimiters, alter how `walk` iterates, then regenerate the reference outputs.
Easier: move back toward mainstream syntax. Semantics stay fixed; only the surface moves.
