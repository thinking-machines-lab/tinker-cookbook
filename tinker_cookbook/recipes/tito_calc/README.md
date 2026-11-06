# tito_calc — Parity demo: cookbook renderer vs `apply_chat_template`

A tiny recipe that runs the cookbook's existing [`math_rl`](../math_rl/) arithmetic
recipe **twice** — once with `renderer_name="llama3"` (the cookbook's hand-coded
`Llama3Renderer`), once with `renderer_name="apply_chat_template"` (the
model-agnostic `TitoRenderer` that delegates to
`tokenizer.apply_chat_template`) — and prints a per-step diff of the headline
metrics.

The point of the demo: for any model whose chat template is prefix-preserving
for tool messages, the cookbook does not need a per-family renderer. One
generic renderer that calls `apply_chat_template` plugs into the existing RL
loop and trains as well as the hand-coded fork.

Background, motivation, and the property test this relies on: **TITO blog
post —** [_Agentic RL: Token-In, Token-Out Done Right_][tito]
(see §6 *Prefix preservation* and §7 *Do you need a renderer for this?*).

[tito]: https://qgallouedec-tito.hf.space/

## Run

```bash
python -m tinker_cookbook.recipes.tito_calc.run_compare \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-steps 5 --batch 20 --group 4
```

## Result we measured

`meta-llama/Llama-3.1-8B-Instruct`, `math_rl` arithmetic, 5 steps:

```
COMPARISON: A = cookbook Llama3Renderer    B = apply_chat_template (TitoRenderer)
step  | env/all/correct  | reward/total | kl_sample_train_v1 | entropy
  0 A:     0.738         |   0.733      |    0.001541        |  0.364
  0 B:     0.238         |   0.211      |   -0.003513        |  0.241
  1 A:     0.988         |   0.988      |   -0.000319        |  0.042
  1 B:     1.000         |   1.000      |    0.000102        |  0.002
  2-4 both: 1.000        |   1.000      |    ~0              |  ~0
```

Both arms converge to 100% reward by step 2. `kl_sample_train_v1` stays
small in both (`~0.003` worst case, ~0 once converged). The step-0 gap
reflects different bytes the model sees (the cookbook's simplified
rendering omits HF's `Cutting Knowledge Date:` preamble; the
`apply_chat_template` path includes it). The model adapts in one step.

## Published checkpoints (this run)

The final sampler LoRA adapters for both arms are published on Tinker:

- Run A (`renderer_name=llama3`): `tinker://2c10608f-fa00-5833-9c73-2eb6b652ed65:train:0/sampler_weights/final`
- Run B (`renderer_name=apply_chat_template`): `tinker://5632da7c-4ae4-5eda-96f1-c57cdc0e0bfd:train:0/sampler_weights/final`

Download with `tinker checkpoint download <path>`.

## Files

| file | role |
|---|---|
| `run_compare.py` | driver: calls `math_rl.train.cli_main` twice with different `renderer_name` |
| `README.md` | this file |
| `__init__.py` | package marker |

The `TitoRenderer` class itself lives upstream in
[`tinker_cookbook/renderers/apply_chat_template.py`](../../renderers/apply_chat_template.py)
and is wired into `get_renderer` under the name `"apply_chat_template"`, so
**any cookbook recipe** can use it via `renderer_name="apply_chat_template"`.
This recipe is just a demonstration that the parity holds on a real Tinker
training run.
