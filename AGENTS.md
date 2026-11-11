# Tinker Cookbook Agent Guide

Quick reference for agents working on `tinker-cookbook`. Detailed guidance is in the skills under `skills/`.

`tinker-cookbook` is a client library with training and eval code built on the Tinker service (hosted by Thinking Machines Lab) and the Tinker SDK (a separate repo with just the API). You author training/eval loops that run on a CPU machine; Tinker executes the heavy GPU work.

**Skills:** This repo ships four Claude Code skills in `skills/`: `research` (SFT, RL, DPO, distillation, evaluation, model selection, experiment methodology), `debug` (performance, correctness, renderer, and error triage), `inkling` (thinking effort, rendering, post-training, and multimodal input for Inkling series models, including Inkling 2 Small), and `rollout-ui` (rollout content, messages, tool calls, transcript comparisons, and browser validation; metrics stay in W&B). Install via `/plugin marketplace add thinking-machines-lab/tinker-cookbook`, then install the `tinker` plugin from Discover (`/plugin`). Use `/tinker:research`, `/tinker:debug`, `/tinker:inkling`, or `/tinker:rollout-ui`.

## Composing Types

Agents often struggle with the nested type hierarchy.

**Core types:**

- `Datum` = `model_input` (ModelInput) + `loss_fn_inputs` (dict of TensorData)
- `ModelInput` = list of chunks (EncodedTextChunk, ImageChunk)
- `TensorData` = wrapper for numpy/torch arrays with shape info

**Helper functions** (use these instead of manual construction):

- `datum_from_model_input_weights(model_input, weights, max_length)` - SL datum creation (`supervised/common.py`)
- `conversation_to_datum(messages, renderer, max_length, train_on_what)` - Full pipeline (`supervised/data.py`)
- `renderer.build_supervised_example(messages)` - Returns (ModelInput, weights)
- `ModelInput.from_ints(tokens)` - Create from token list
- `TensorData.from_numpy(arr)` / `TensorData.from_torch(tensor)` - Wrap arrays

---

## Architecture

**Builder pattern:** Config objects are `chz` dataclasses (SupervisedDatasetBuilder, RLDatasetBuilder, EnvGroupBuilder). They expose `.build()`/`__call__()` returning runtime objects.

**Key code locations:**

- SL: `tinker_cookbook/supervised/train.py`
- RL: `tinker_cookbook/rl/train.py`
- DPO: `tinker_cookbook/preference/train_dpo.py`
- Renderers: `tinker_cookbook/renderers/`
- Completers: `tinker_cookbook/completers.py`
- RL types: `tinker_cookbook/rl/types.py`
- Rollout strategies: `tinker_cookbook/rl/rollout_strategy.py` (FailFast, RetryOnFailure)
- Logging: `tinker_cookbook/utils/logtree.py`, `tinker_cookbook/rl/rollouts.py`
- Recipes: `tinker_cookbook/recipes/`

**Training outputs:** RL and SL training write human-readable HTML reports and machine-readable JSON files (metrics, rollout transcripts, per-trajectory summaries) to `log_path`. Point agents at a `log_path` directory to analyze training runs — `metrics.jsonl` for scalar metrics, `*_rollout_summaries.jsonl` for per-trajectory data, and `*_logtree.json` for full rollout transcripts including model responses.

**Default output paths:** Recipes default `log_path` (and generated data, sweeps, eval outputs) to `cli_utils.runs_path(...)`, which resolves under `$TINKER_COOKBOOK_RUNS_DIR` or `~/tinker-runs`. Never default outputs to `/tmp`: many machines clear it on reboot. Re-downloadable caches go under `~/.cache/tinker-cookbook/`.

---

## Conventions

**Subscript suffixes** for tensor names: `_P` (problems), `_G` (groups), `_T` (tokens), `_D` (datums). Example: `tokens_P_G_T[p][g][t]`

**Code style:**

- Explicit typing; avoid `Any` / `type: ignore`
- Use `safezip`, `timed`, `scope` helpers
- `@chz.chz` decorator for config serialization
- `ml_log.log_metrics` for metrics; `logtree` for transcripts

**Env lifecycle:** `Env` objects are single-use (no reset). Create via `EnvGroupBuilder`.

---

## Public-repo discipline

`tinker-cookbook` is a public repository. Keep internal context out of commits (even intermediate commits), and write with an external audience in mind.

---

## Finding supported models

For an identifier to pass to the SDK, call `tinker.RestClient().get_server_capabilities().result().supported_models` — authoritative, includes `:peft:` long-context variants, but requires `TINKER_API_KEY`. For a read-only lookup with pricing and context length (no auth), browse <https://tinker-docs.thinkingmachines.ai/tinker/models/>.

---

## Common Pitfalls

1. **Sequential API calls:** The #1 performance mistake. Submit calls back-to-back before awaiting their `APIFuture`s. Use `asyncio.gather` for concurrent evaluation — never sequential loops over API calls. Tinker is designed for high request concurrency from a single Python process; when exact limits matter, check the current SDK/client configuration rather than inventing small caps. For very high parallelism, shard work across multiple Python processes and pass pickled sampling clients where appropriate.

2. **Client-side timeouts and retries:** Don't wrap Tinker requests in your own timeouts (e.g. `asyncio.wait_for` around `sample`) or retry loops. The SDK and backend already retry transient failures, and the SDK detects stuck requests. Tinker is optimized for throughput over latency, so request latency varies with system load and generation length; there is no correct fixed timeout. Under load, aggressive timeouts turn slow-but-progressing steps into total failure (every request in a batch times out, retries just add load), which especially bites RL sampling loops that bound per-batch sampling time. Submit requests concurrently and wait for them; slow steps resolve on their own. Reserve `RetryOnFailure` / `per_rollout_timeout` for non-Tinker failures (flaky sandboxes, tool errors), not as a bound on sampling latency. If sampling speed is dramatically blocking your training jobs, email support (<https://tinker-docs.thinkingmachines.ai/support/>) with your session ID.

3. **Sampler desync:** Create a **new** sampling client after saving weights. A stale client silently samples from old weights.

4. **LoRA LR:** Use `hyperparam_utils.get_lr(model_name)` where calibrated. Inkling models require an explicit, task-calibrated learning rate; the helper deliberately raises for them.

5. **Renderer mismatch:** Use `model_info.get_recommended_renderer_name()` — never hardcode renderer names. Pass effort explicitly where supported.

   **Never call `tokenizer.encode(prompt)` directly on a chat-tuned model** (gpt-oss, Llama-3-Instruct, Qwen-Instruct, etc.). Raw encoding skips the chat template, producing OOD prompt tokens. The sampler and trainer then take subtly different code paths on those OOD inputs, and per-token sampler/trainer logprob KL can inflate by 5×+ (max ratios in the tens), silently breaking PPO/CISPO/GRPO importance ratios. Use `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)` (take `["input_ids"]` from the returned `BatchEncoding`) or a cookbook renderer for the prompt. Raw `encode` is correct only for base / continued-pretraining / NLL-eval workflows where there is no conversation.

6. **Context length:** Be mindful of the context length of the model you pick. Each model's maximum sequence length is listed on <https://tinker-docs.thinkingmachines.ai/tinker/models/>, and sequences that exceed it get truncated or rejected. For long-horizon tasks (multi-turn agentic RL, long tool-use trajectories, long documents), always prefer the extended-context variant of the model if one is available — these are the `:peft:<length>` model IDs in `get_server_capabilities().result().supported_models` (e.g. `...:peft:131072` for 128K, `...:peft:262144` for 256K). Also check trajectory caps like `max_trajectory_tokens` and datum `max_length` — a model with a large context window still truncates if the loop's limits are smaller.

7. **Type construction:** Use helper functions, not manual dict construction. See `supervised/data.py` and `supervised/common.py`.

8. **Group semantics:** RL advantages are centered within each group.

9. **DPO:** Start with `dpo_beta=0.1`, LR~1e-5.

---

## Running training experiments

Defaults to follow when launching training runs for a user. Deviating from any of them is fine, but say so explicitly and get agreement before launch rather than letting it happen silently.

**Set up Weights & Biases before launching.** Logging to W&B is the easiest way to watch a run and to compare the runs in a sweep. Before launch:
- Authenticate (`wandb login`).
- Set `wandb_project` and `wandb_name` in the training config.
- Use one project per experiment, with consistent run names, so sweep runs overlay cleanly.
- Make sure the logged config includes the model, initialization, dataset and split, LR, batch and group sizes, seed, and `max_steps_off_policy`.
- Confirm the first few rows reach W&B before leaving the run unattended.
- Keep training metrics (`env/...`) and held-out metrics (the evaluator's prefix, e.g. `test/env/...`) visually separate.
- Plot held-out reward against step, cumulative cost and wall-clock time.

**Metrics to watch, roughly in priority order.** Exact keys come from the RL trainer; env-specific keys depend on what the environment logs.
1. **Held-out reward**, `<eval_name>/env/all/reward/total` (e.g. `test/env/all/reward/total`). This is the main quality signal. Compare runs on it, not on training reward.
2. **Training reward**, `env/all/reward/total`. If it rises while held-out reward stays flat, suspect overfitting or reward exploitation (eval noise and distribution gaps can also explain a gap), and check transcripts. If it falls, investigate.
3. **Learning signal**, `env/all/by_group/frac_mixed`, along with `frac_all_good` and `frac_all_bad`. With group-centered advantages and a binary task reward, only mixed groups produce a nonzero advantage. Continuous or shaped rewards can still differ inside all-good or all-bad groups. If most groups are all-good or all-bad under a binary reward, the task is too easy or too hard to learn from.
4. **Response length**, `env/all/ac_tokens_per_turn`. Sudden growth often means looping or hitting the length cap; collapse toward zero often means degenerate outputs.
5. **Format or validity rate**, if the environment logs one (env-specific keys under `env/all/`). Falling validity is usually the first sign of a collapse.
6. **Sampling entropy**, `optim/entropy`. A steady fall toward zero is a warning to inspect output diversity, reward and transcripts. On its own it doesn't prove collapse; deterministic correct answers are legitimate.
7. **Sampler/trainer mismatch**, `optim/kl_sample_train_v1` and `optim/kl_sample_train_v2`. These measure the logprob gap between sampler and trainer on the same tokens, not how far the model has drifted from its starting point. Large values point to a rendering or prompt-encoding problem, or to stale data.
8. **Staleness**, `async/staleness_mean` and `async/staleness_max` (async runs). These show how old the sampled data actually is relative to `max_steps_off_policy`.
9. **Cost**, `cost/cumulative_usd` and `cost/step_usd`. These are estimates, not a bill. Use them as the x-axis for reward-per-dollar comparisons.
10. **Time and progress**, `time/total` (seconds per step), the per-phase `time/<span>` timings the tracer logs, and `progress/done_frac`. Use them to find the bottleneck and estimate time to completion. Derive throughput yourself: completed updates, rollouts, or `tokens/sampled` per elapsed second. Keep per-step counts separate from cumulative ones (e.g. `tokens/cumulative/sampled`).

1. **Train on the intended dataset.** Use the full training split unless the user asked for a subset. Don't silently train on a tiny repeated subset; label diagnostic runs as diagnostics. Before launch, report train and held-out row counts, unique prompts (and pairs or groups, if the recipe builds them), train/held-out overlap (should be zero), updates per pass, and the total budget.

2. **Track held-out performance from the start.** Enable the recipe's periodic held-out evaluation, including at the initial and final checkpoints. Evaluate a fixed, deterministic subset so points are comparable, sized so eval cost stays small relative to training, and run the full held-out set at the endpoint. Log held-out reward against checkpoint step, cumulative estimated training cost, and elapsed training time, using the values at the checkpoint snapshot rather than when the eval finished. Report eval cost separately, and mark unknown values as unavailable instead of estimating them. Confirm the metrics actually reach the logger early in the run.

3. **Capture all rollouts.** Log every group's transcripts by default rather than a representative sample, and verify the actual coverage. If full capture is too expensive, discuss the tradeoff first.

4. **Use async RL with `stable_reinforce`** by default, including for small diagnostics. Size concurrency from the SDK's capacity instead of inventing small caps. Keep off-policyness bounded: async should overlap sampling with training, not sample the whole dataset from the initial weights. `AsyncConfig.max_steps_off_policy` (call it `m`) admits at most `groups_per_batch * (m + 1)` groups that are started but not yet trained. It controls admission and doesn't cap sample age, since slow rollouts are still trained when they finish.
   - Default to a high pipeline depth: as large as the service and workload can usefully keep busy, so sampling and training overlap fully.
   - The one constraint: the window `groups_per_batch * (m + 1)` should stay small next to the number of training groups in a pass, so the weights get refreshed many times over the pass. Never let it sample the whole dataset from the initial weights.
   - Pick the starting `m` from the dataset size, rollout latency, and service capacity, not from a fixed number.
   - Increase it until wall-clock throughput plateaus, or until measured staleness starts to hurt held-out reward or validity.
   - Measure actual data age (`async/staleness_mean` and `async/staleness_max`) and wall-clock time, rather than assuming the age equals `m`.

5. **Use precise units.** Distinguish groups per batch, `group_size`, and total datums per update; step index vs. completed updates; per-batch vs. cumulative values. Always include average decode tokens per turn in training metrics.

6. **Be explicit about initialization and continuity.** State whether a run starts from the base model or a checkpoint, and whether optimizer state is fresh. When stopping, resuming, or restarting, say what training and optimizer state carries over. Don't call a run healthy from a heartbeat alone; check that steps and metrics are advancing.

---

## Testing

```bash
# Unit tests (no API needed, colocated *_test.py files)
pytest tinker_cookbook/

# Smoke tests (requires TINKER_API_KEY + network)
pytest tests/
```

For debugging, shrink workloads via `n_batches`, `batch_size`, `group_size` in dataset builders.
