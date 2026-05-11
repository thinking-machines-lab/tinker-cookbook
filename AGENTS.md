# Tinker Cookbook Agent Guide

Quick reference for agents working on `tinker-cookbook`. Detailed guidance is in the skills under `skills/`.

`tinker-cookbook` is a client library with training and eval code built on the Tinker service (hosted by Thinking Machines Lab) and the Tinker SDK (a separate repo with just the API). You author training/eval loops that run on a CPU machine; Tinker executes the heavy GPU work.

**Skills:** This repo ships two Claude Code skills in `skills/`: `research` (SFT, RL, DPO, distillation, evaluation, model selection, experiment methodology) and `debug` (performance, correctness, renderer, and error triage). Install via `/plugin marketplace add thinking-machines-lab/tinker-cookbook`, then use `/tinker:research` or `/tinker:debug`.

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

## Common Pitfalls

1. **Sequential API calls:** The #1 performance mistake. Always use `_async` variants and submit calls back-to-back before awaiting. Use `asyncio.gather` for concurrent evaluation — never sequential loops over API calls.

2. **Sampler desync:** Create a **new** sampling client after saving weights. A stale client silently samples from old weights.

3. **LoRA LR:** Use `hyperparam_utils.get_lr(model_name)` - LoRA needs ~10x higher LR than full fine-tuning.

4. **Renderer mismatch:** Use `model_info.get_recommended_renderer_name()` — never hardcode renderer names.

5. **Type construction:** Use helper functions, not manual dict construction. See `supervised/data.py` and `supervised/common.py`.

6. **Group semantics:** RL advantages are centered within each group.

7. **DPO:** Start with `dpo_beta=0.1`, LR~1e-5.

---

## Testing

```bash
# Unit tests (no API needed, colocated *_test.py files)
pytest tinker_cookbook/

# Smoke tests (requires TINKER_API_KEY + network)
pytest tests/
```

For debugging, shrink workloads via `n_batches`, `batch_size`, `group_size` in dataset builders.
