# Hyperparameter Selection

Detailed guide for choosing training hyperparameters across SL, RL, DPO, and distillation.

## Reference

- `tinker_cookbook/hyperparam_utils.py` — calibrated per-model LRs and LoRA parameter counts
- `tinker_cookbook/lr_multipliers.json` — the reference LR and per-model multipliers that `get_lr` reads
- `tinker_cookbook/lr_calibration/calibration_data.json` — the calibration protocol and each model's sweep and fit,
  written by `python -m tinker_cookbook.lr_calibration`, which regenerates `lr_multipliers.json` from it

## Learning rate

> [!IMPORTANT]
> **These are starting points, not guarantees.** `get_lr` and every rule in this guide
> give a starting point for an LR sweep, not a tuned optimum for your task. How well
> they hold depends on your data, model and setup. Sweep nearby learning rates on your
> task and pick the value with held-out evaluation.

### Reference LR and model multipliers

`get_lr` reads `reference_lr` and the per-model multipliers from `lr_multipliers.json`.
That file is generated from `lr_calibration/calibration_data.json`, the source of truth,
so make changes there and run `python -m tinker_cookbook.lr_calibration --regenerate`
rather than editing `lr_multipliers.json`. Currently, Qwen3-8B is
the reference model, with `model_multiplier = 1`.
Import `get_lr`, `get_lora_lr_multiplier` and `transfer_lr` from
`tinker_cookbook.hyperparam_utils`; do not copy the model table into training code
or these guides. Models without a calibrated entry raise `ConfigurationError`
and need direct tuning.

```text
get_lr(model) = reference_lr * model_multiplier(model)
```

The model ratios are measured with short AdamW LoRA sweeps on plain pretraining
text. The absolute `reference_lr` is a practical SFT starting point for Qwen3-8B:
rank 32, about 128 examples per update, several hundred steps and linear decay to
zero.

This is how we calibrated the table, using `python -m tinker_cookbook.lr_calibration`.
You can adapt that tool to your own workload, but we recommend sweeping the LR
directly on your task instead (see
[Choose a starting point, then tune](#choose-a-starting-point-then-tune)).

### Multiplier rules

These names describe adjustments in this guide, not `get_lr` arguments. Apply a
rule only within its scope and only for a setting that differs from your baseline.

| Name | Adjustment | Basis and scope |
|---|---|---|
| `model_multiplier` | Look up the value with `get_lora_lr_multiplier(model)`; for transfer, use the new/old model ratio | Measured AdamW LoRA ratios on pretraining text; check transfer on your task |
| `objective_multiplier` | SFT baseline; other objectives use the next table | RL has limited evidence; DPO and on-policy distillation are heuristics |
| `optimizer_multiplier` | AdamW baseline; Dimuon about 30 times a matched tuned AdamW LR | Approximate short-pretraining-text conversion; not validated for chat-SFT, RL, DPO or on-policy distillation. See [measurement details](#how-these-numbers-were-measured) |
| `batch_multiplier` | Adjust examples per update with the [combined scaling formula](#scale-or-transfer-a-tuned-lr) | Empirical AdamW and Dimuon SFT approximation in the small-batch regime |
| `training_tokens_multiplier` | Adjust total input-token budget with the [combined scaling formula](#scale-or-transfer-a-tuned-lr) | Empirical AdamW and Dimuon SFT approximation |
| `schedule_multiplier` | Linear-decay baseline; constant LR about 0.5 times the corresponding linear-decay LR | Additional heuristic, not a conversion inferred from schedule step-efficiency gains; keep the budget fixed and tune when changing schedules. No calibrated cosine conversion |

LoRA rank is a separate adjustment; see [LoRA rank](#lora-rank). Dataset and loss
changes have no multiplier; tune again.

| Training objective | `objective_multiplier` | Basis |
|---|---|---|
| SFT, including off-policy distillation | 1 | The setting `reference_lr` is chosen for. A coarse Tulu3 SFT sweep found task-dependent optima |
| RL | about 0.1 | Limited MATH evidence; the conservative end of the measured range. See [measurement details](#how-these-numbers-were-measured) |
| DPO | 0.025 | Not measured; approximates the DPO recipe's LR scale |
| On-policy distillation | 0.25 | Not measured; approximates the distillation recipes' LR scale |

### Choose a starting point, then tune

For AdamW, start with `get_lr(model) * objective_multiplier`. `get_lr` has no
defined training-token budget, so apply the batch and token formula to an LR you
tuned, not to `get_lr`. The Dimuon factor converts a tuned AdamW LR and has only
been checked on pretraining text. Tune other optimizer and task settings directly.

Sweep cheaply with the intended data, optimizer, loss and schedule, using fixed
held-out evaluation. Record the tuned model, LR, batch size and units, total
training tokens, rank, schedule and optimizer configuration.

### Scale or transfer a tuned LR

For AdamW or Dimuon SFT, with data, loss, optimizer, rank and schedule unchanged
when scaling batch size or token budget. The model-transfer step uses
AdamW-calibrated ratios; for Dimuon, tune on the target model directly:

```text
lr_scaled = tuned_lr * (B_new / B_tuned) ** 0.5 * (T_new / T_tuned) ** -0.25
lr_new_model = lr_scaled * model_multiplier(new_model) / model_multiplier(tuned_model)
```

Use `transfer_lr(lr_scaled, tuned_model, new_model)` for the second line. If the
model, batch size or token budget is unchanged, its adjustment is 1. The tuned LR
already includes its objective, optimizer and schedule: do not apply those factors
again or restart from `get_lr`. Recheck held-out performance after scaling.

Here, `B` is examples per optimizer update in the standard SFT dataset builders,
not tokens. `T` is actual total input tokens processed during training after
truncation, including prompts and repeated passes, but excluding evaluation.
Use the same units for the tuned and new runs. Do not count only loss-bearing
tokens as `T`, or assume different tokenizers produce the same token budget.

At fixed steps and unchanged example lengths, increasing batch also increases
`T`: the combined rule becomes `(B_new / B_tuned) ** 0.25`. This is not an
additional multiplier; do not apply it alongside both terms above. These are SFT
rules, not validated general RL scaling laws; RL groups and rollout counts are
[defined separately](#rl-batch-size-and-group-size).

## LoRA rank

- **Default**: 32 for most tasks
- **Higher rank** (64-128): More capacity for complex tasks or large models
- **Lower rank** (8-16): Faster, sufficient for simple adaptations
- **LR at other ranks**: Treat a tuned LR as a starting point when increasing rank,
  then check whether another sweep is needed. Very low ranks sometimes prefer lower
  LRs in the [rank checks](#rank-checks), but the effect is task-dependent; do not
  apply a universal rank correction.

```python
from tinker_cookbook.hyperparam_utils import get_lora_param_count
params = get_lora_param_count("Qwen/Qwen3.5-9B-Base", lora_rank=32)
```

## Batch size

### SL batch size
- Standard SFT dataset builders measure `batch_size` in **examples**, not tokens.
  Check custom builders before assuming the same units; record tokens per update separately.
- Start with 128 examples per update.
- Use the [combined batch/token formula](#scale-or-transfer-a-tuned-lr). At larger
  batches, which can already include 128, the best LR rises more slowly than the
  square-root rule, so retune rather than extrapolate.
- Aim for at least 100 training steps (best results with 1000+)

### RL batch size and group size
- `groups_per_batch`: Number of trajectory groups per optimizer batch; commonly one
  group per problem. Some recipes call this `batch_size`; check the builder.
- `group_size`: Rollouts per group, with advantages centered within the group.
- `total_rollouts = groups_per_batch * group_size`; do not multiply by problems again.
- With `group_size` fixed, ratios of group counts and rollout counts coincide.
  Changing `group_size` also changes the learning signal, not just the batch size;
  do not assume the SFT square-root rule still applies.
- Start small for debugging: `groups_per_batch=4, group_size=2`

## Learning rate schedule

- `"linear"` — Linear decay to 0 (most common)
- `"cosine"` — Cosine annealing
- `"constant"` — No decay

The [schedule multiplier](#multiplier-rules) adjusts the starting/peak LR when
changing schedule type. It is distinct from the per-step decay applied by the
training loop. Tune using the intended schedule rather than changing it after a
sweep whenever possible.

## `num_substeps` (RL)

- `num_substeps=1` (default): One update per batch
- `num_substeps>1`: Splits batch into mini-batches. Requires PPO objective.
- Start with 2-4 if experimenting; decrease LR with higher values

## DPO-specific

- `dpo_beta=0.1` — Well-tested default
- Lower beta = more aggressive; higher beta = closer to reference

## Distillation-specific

- `kl_penalty_coef=1.0` — Weight of KL penalty from teacher
- `kl_discount_factor=0.0` — No discounting (increase for long sequences)

## Quick-start recommendations

Choose the LR with [Learning rate](#learning-rate). These are example starting
configurations; check your dataset builder's units.

| Scenario | Model | Batch | LoRA rank |
|---|---|---|---|
| SFT on chat data | Qwen3.5-9B-Base | 128 examples | 32 |
| Math GRPO | Qwen3.5-9B (thinking) | 128 groups x 16 rollouts/group | 32 |
| DPO | Qwen3.5-9B-Base | 256 pairs | 32 |
| On-policy distillation | Qwen3.5-9B-Base | 1024 prompts x 4 samples/prompt | 128 |
| Multi-turn RL | Kimi-K2.6 | 8 groups x 4 rollouts/group | 32 |

No separate multi-turn objective factor has been measured. Smaller RL groups or
rollout batches do not automatically justify applying the SFT batch-scaling rule.

## Pitfalls

- `get_lr(model)` is a starting point for an SFT sweep, not a tuned LR for your task.
  Read [Learning rate](#learning-rate) before applying an objective or optimizer adjustment.
- A tuned LR already includes the training objective. Apply only changes relative to
  that run; do not double-count batch/token scaling or reset its scale with `get_lr`.
- RL can be destabilized by aggressive updates. Tune and check held-out reward on
  your task; model transfer and the SFT scaling rules are not guaranteed for RL.
- Monitor the named KL metrics in RL, but do not infer stability from one universal
  threshold. Sampler/trainer mismatch and reference-policy drift are different
  diagnostics; also check held-out reward, output validity and entropy.


## How these numbers were measured

### Model calibration and uncertainty

Each model is calibrated with short LoRA runs on pretraining text: a fresh rank-32
adapter, 64 steps of 8 windows of about 4,000 characters, linear decay to zero,
and AdamW with beta1=0.9, beta2=0.95 and epsilon=1e-8. A coarse grid with 2x
spacing brackets the optimum, then a finer sqrt(2) grid around it is run with five
seeds. A quadratic in log LR fitted to the median held-out loss gives each model's
optimum, and its multiplier is that optimum divided by Qwen3-8B's. The protocol,
curves and fits are in `tinker_cookbook/lr_calibration/calibration_data.json`, and
the fit is in `lr_calibration/fit.py`. The absolute `reference_lr` is a practical
SFT choice explained in the JSON's `reference_lr_note`, not a measured default.

Leaving out one seed at a time and refitting both the model and Qwen3-8B reference
gives a jackknife standard error of the log optimum-LR ratio of about 4% RMS
across the 23 models, up to about 10% for Kimi-K2.6. This estimates seed uncertainty
of the five-seed estimator, not raw leave-one-out movement or dataset uncertainty.
A separate one-seed check (FineWeb-Edu, 128 steps, 16 windows per update, constant
LR) gave about 14% RMS relative disagreement across its original 21-model cohort,
including the reference. Here the error is `calibrated_ratio / followup_ratio - 1`,
with each ratio normalized to Qwen3-8B. Individual models differed by more; this
is not a per-model bound. The follow-up brackets used prior calibration predictions,
so it is a separate-run check, not blind validation. Neither check measures
transfer to your task.

### Rank checks

Repeating the short pretraining protocol at other ranks for Qwen3-8B, gpt-oss-20b
and Nemotron-3.5-Lightning, with three seeds each, gave optimum-LR ratios relative
to rank 32 of 0.85-0.96 at rank 8, 0.7-0.9 at rank 4 and about 0.5-0.6 at rank 1.
Ranks 64 and 128, each tested on one model, matched rank 32 within seed noise.
The effect depends on the task: at constant LR on short pretraining examples,
rank 1 wanted 0.67-0.83 of the rank-32 LR; on a one-token-answer classification
task at batch 32, rank 1 wanted 0.83-0.94 and rank 4 wanted 0.89-0.94.

### Batch and training-token checks

Pretraining-text sweeps with Qwen3-8B and gpt-oss-20b used constant LRs and
batches from 1 to 1,024 sequences. At fixed training-example count, optimum LR
grew with the square root of batch size in the small-batch regime, consistent
with the square-root scaling rule for AdamW (Malladi et al., 2022). Beyond that regime,
the median log-log slope was about 0.18. For roughly 1,000-token sequences, the
range where a larger batch cost no loss ended at 4-64 sequences within the first
2,000 examples; the boundary depends on the data and training progress.

At fixed steps, the observed batch-size exponent was about 0.15-0.25. At fixed
batch size, optimum LR fell with run length as about steps^-0.2 under linear
decay and steps^-0.3 to steps^-0.4 under a constant LR with small batches. The
combined scaling formula uses an approximate exponent within this observed
range; it should not be extrapolated indefinitely.

The SFT studies in the learning rate guide
also compare AdamW and Dimuon: fixed-example batch sweeps on Tulu3 and fixed-batch
training-budget sweeps on OpenThoughts3 support the same scaling starting rules
for both optimizers in those settings. This does not establish a universal
AdamW-to-Dimuon LR conversion or calibrate the model ratios for Dimuon.

### Objective and optimizer checks

The RL factor comes from a MATH check on three models (32 groups of 16 rollouts
per update, 41 steps), where the optima were 0.1-0.4 times `get_lr`; the starting
rule chooses the conservative low end. The DPO and on-policy distillation factors
were not measured: they were chosen to approximate the LR scales of the
[DPO recipe](../../../tinker_cookbook/recipes/preference/dpo/train.py) (`1e-5`) and
[on-policy distillation recipe](../../../tinker_cookbook/recipes/distillation/on_policy_distillation.py)
(`1e-4`). They do not reproduce those fixed defaults for every model.

The constant-versus-linear schedule factor is also a starting heuristic: the
calibration record does not provide a paired LR-conversion measurement. A gain
in step efficiency does not itself determine the optimum-LR ratio.

For Dimuon, matched sweeps on seven models under the calibration protocol put the
optimum at around 30x the AdamW LR, ranging from about 25x to 38x across models.
This was tested only on pretraining text, not on chat SFT, RL, DPO or on-policy
distillation. Dimuon's best loss was slightly worse on all seven models, so the
factor says nothing about which optimizer is better.

### Loss reduction in the supervised trainer

The calibration uses the same AdamW settings as `supervised/train.py`, but sums
loss over tokens, while the standard SFT data builder averages it per example
(`reduction="mean"`). AdamW largely cancels a uniform loss rescaling, but
per-example averaging also weights short examples more than long ones.
