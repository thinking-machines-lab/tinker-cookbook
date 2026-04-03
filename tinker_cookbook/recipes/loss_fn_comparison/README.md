# Comparing RL Loss Functions

Tinker provides four built-in RL loss functions: `importance_sampling`, `ppo`, `cispo`, and `dro`. This recipe trains the **same task** with each loss function so you can see how they differ in practice.

## Tinker loss functions vs. named algorithms

An RL training algorithm has two parts:
1. **Advantage estimation** — how you turn rewards into per-token training signal
2. **Loss objective** — how you use those advantages to compute gradients

Traditional PPO (Schulman 2017) uses a **critic network** (learned value function) for advantage estimation and a **clipped surrogate objective** for the loss. It requires 4 models in memory: policy, critic, reference policy, and reward model.

[GRPO](https://arxiv.org/abs/2402.03300) (DeepSeekMath) replaces the critic with a simpler scheme: sample a group of completions per prompt, score them with a verifiable reward (e.g., "is the math answer correct?"), and compute advantages by mean-centering rewards within the group (`A = R - mean(R)`). No critic, no separate reward model. GRPO keeps PPO's clipped loss objective.

Tinker separates these two parts. All four `loss_fn` options in this recipe use **GRPO-style group-relative advantages** (no critic). They differ only in the loss objective:

| Tinker `loss_fn` | Loss objective | Named algorithm |
|---|---|---|
| `ppo` | PPO's clipped surrogate | **GRPO** — this is standard GRPO |
| `importance_sampling` | Unclipped IS ratio × advantage | REINFORCE with group-relative baseline |
| `cispo` | Detached clipped ratio × log p_θ | [CISPO](https://arxiv.org/abs/2506.13585) (MiniMax-M1) |
| `dro` | log p_θ × advantage − quadratic KL penalty | DRO |

**In the results below, `ppo` = standard GRPO.** The other three are variants that replace GRPO's clipping with alternative mechanisms.

## How the loss functions differ

All four losses solve the same problem: given a batch of rollouts sampled from policy `q`, update parameters `θ` to increase the probability of high-reward completions. The key quantity is the **importance sampling ratio**:

```
ratio = p_θ(x) / q(x)
```

This ratio corrects for the mismatch between the policy that generated the data (`q`, the sampler) and the policy being trained (`p_θ`, the learner). When `ratio > 1`, the learner assigns higher probability than the sampler did; when `ratio < 1`, lower.

The four losses differ in how they use this ratio to compute gradients. These differences determine **how aggressively the policy can change in a single step**, which directly affects convergence speed, stability, and what tokens receive gradient signal.

### importance_sampling (REINFORCE)

Standard REINFORCE with importance sampling correction:

```
L = -(p_θ(x) / q(x)) * A(x)
```

The gradient of this loss is `∇L = -A(x) * ∇ log p_θ(x)` — the ratio disappears in the gradient because `∇(p_θ/q) = (p_θ/q) * ∇ log p_θ`. This means **every token gets a gradient proportional to its advantage**, regardless of how much the policy has shifted.

The catch: if `p_θ` diverges far from `q` (large ratio), the loss landscape becomes steep and updates can overshoot. This is fine for single-step on-policy training but can destabilize multi-step updates.

### ppo (GRPO / Proximal Policy Optimization)

Clips the ratio to prevent destructively large updates:

```
L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

Default clip range: `[0.8, 1.2]`. The `min` selects the **pessimistic** estimate — if the unclipped objective looks better than the clipped one, PPO uses the clipped version instead. This creates a "trust region" where the policy can move freely, but updates beyond `±ε` are suppressed.

**The problem for reasoning:** When a token is rare under `q` but important for the correct answer (e.g., a "Wait, let me reconsider" correction token), its ratio quickly exceeds the clip threshold. Once clipped, that token receives **zero gradient** — PPO completely drops it from the update. For short outputs this rarely matters, but for long chain-of-thought reasoning, these "fork" tokens (where the model decides to backtrack or try a different approach) are exactly the tokens that matter most.

### cispo (Clipped Importance Sampling Policy Optimization)

From the [MiniMax-M1 paper](https://arxiv.org/abs/2506.13585). Clips the ratio but uses it as a **detached coefficient** on `log p_θ`:

```
L = -sg(clip(ratio, 1-ε_low, 1+ε_high)) * A(x) * log p_θ(x)
```

where `sg()` is stop-gradient (PyTorch `detach()`). The key difference from PPO:

- **PPO:** gradient flows through `ratio * A` → when ratio is clipped, gradient is exactly zero
- **CISPO:** gradient flows through `log p_θ` → the clipped ratio is just a scalar weight, and `∇ log p_θ` always contributes

This means **every token always contributes gradients**, even when the ratio is far from 1. The clipped ratio controls *how much* each token contributes (downweighting tokens where the policy has already moved far from the sampler), but never drops tokens entirely.

In practice, CISPO uses **asymmetric clipping**: `ε_low` is set large (permissive downweighting) while `ε_high` is tuned (caps upweighting). This allows the policy to freely reduce probability of bad tokens but limits how fast it can concentrate probability on good ones.

### dro (Distributionally Robust Optimization)

Replaces hard clipping with a **soft quadratic penalty** on policy divergence:

```
L = -(log p_θ(x) * A(x) - 0.5 * β * (log(p_θ(x)/q(x)))²)
```

The gradient is `∇L = -(A(x) - β * log(p_θ/q)) * ∇ log p_θ`, which means the effective advantage is **reduced** when the policy moves far from the sampler. Unlike PPO's hard wall at `ratio = 1±ε`, DRO's penalty grows smoothly — large deviations are increasingly expensive but never fully blocked.

`β` controls the tradeoff: higher β = more conservative updates. At `β = 0`, DRO reduces to plain REINFORCE (`-log p_θ * A`). At very high β, the penalty dominates and the policy barely moves.

**When DRO helps:** The quadratic penalty ensures the policy stays close to the distribution that generated the training data. This is valuable when the data is **off-policy** (generated by an older version of the policy, or a different policy entirely). In that regime, IS and PPO can make updates that look good under the stale data but perform poorly under the actual current policy. DRO prevents this by keeping updates conservative. On-policy (fresh rollouts each step), this conservatism is unnecessary and just slows learning.

## Quick start

### Run a single loss function

The existing `math_rl` recipe already supports `loss_fn` as a parameter:

```bash
# Importance sampling (default)
python -m tinker_cookbook.recipes.math_rl.train loss_fn=importance_sampling

# PPO with default clipping
python -m tinker_cookbook.recipes.math_rl.train loss_fn=ppo

# CISPO
python -m tinker_cookbook.recipes.math_rl.train loss_fn=cispo

# DRO
python -m tinker_cookbook.recipes.math_rl.train loss_fn=dro

# PPO with custom clipping thresholds
python -m tinker_cookbook.recipes.math_rl.train loss_fn=ppo 'loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}'
```

### Run all four sequentially

```bash
# Quick comparison on arithmetic
python -m tinker_cookbook.recipes.loss_fn_comparison.sweep

# On GSM8K with a larger model
python -m tinker_cookbook.recipes.loss_fn_comparison.sweep \
    env=gsm8k \
    model_name="Qwen/Qwen3-8B" \
    max_tokens=512 \
    groups_per_batch=64 \
    group_size=16 \
    learning_rate=2e-5 \
    max_steps=100

# Run only a subset
python -m tinker_cookbook.recipes.loss_fn_comparison.sweep loss_fns=ppo,cispo
```

### Analyze results

```bash
# Compare metrics from all runs
python -m tinker_cookbook.recipes.loss_fn_comparison.analyze --log-dir /tmp/tinker-examples/math_rl
```

## Results

### GSM8K (Qwen3-8B, 100 steps, lr=2e-5, group_size=16, groups_per_batch=64)

#### Test accuracy over training (averaged over 2 seeds where available)

In the tables below, **IS** = `importance_sampling` (REINFORCE), **GRPO** = `ppo` (standard GRPO with clipping).

| Step | IS | GRPO | CISPO | DRO |
|------|---:|---:|---:|---:|
| 0 | 7.8% | 7.2% | 7.9% | 8.1% |
| 10 | 12.7% | 13.1% | 12.2% | 6.6% |
| 20 | 47.8% | 51.2% | 47.4% | 7.4% |
| 30 | 89.6% | 86.7% | 88.2% | 8.7% |
| 40 | 93.3% | 93.4% | 93.7% | 10.1% |

All three IS-based methods converge to **~93-94% test accuracy** by step 40. DRO barely learns.

#### Reproducibility (seed-to-seed variance at step 20)

| Loss | seed=0 | seed=1 | |Δ| |
|------|--------|--------|-----|
| IS | 42.9% | 52.6% | 9.7% |
| GRPO | 51.3% | 51.1% | **0.2%** |
| CISPO | 46.1% | 48.7% | 2.6% |

GRPO is the **most reproducible** — its clipping constrains the update direction, producing nearly identical learning curves across seeds (|Δ|=0.2% at step 20). IS has the highest variance (|Δ|=9.7%). By step 40 all three converge to within 1% regardless of seed.

#### Per-token training cost (controlled for sequence length)

Measured at steps 0-4 when all methods generate ~520K tokens per batch:

| Loss | µs/token | Relative cost |
|------|---------|--------------|
| IS | 30.5 | 1.0x |
| CISPO | 32.1 | 1.1x |
| DRO | 33.3 | 1.1x |
| GRPO | 80.2 | **2.6x** |

GRPO is **2.6x more expensive per token** than IS/CISPO/DRO. IS, CISPO, and DRO have essentially identical per-token GPU cost. GRPO's overhead comes from computing both the clipped and unclipped objectives and taking their `min`.

As training progresses, the gap widens further: methods that learn faster produce shorter responses, making each step cheaper. At step 80-99, CISPO averages 203K tokens/step (8.8s train time) while GRPO averages 236K tokens/step (23.4s train time) — a combined 2.7x wall-clock difference from both algorithmic cost and learned efficiency.

#### Entropy (policy diversity)

| Step | IS | GRPO | CISPO | DRO |
|------|---:|---:|---:|---:|
| 0 | 0.170 | 0.171 | 0.171 | 0.170 |
| 20 | 0.177 | 0.090 | 0.168 | 0.233 |
| 50 | 0.103 | 0.030 | 0.076 | 0.289 |
| 99 | 0.059 | 0.035 | 0.085 | 0.271 |

GRPO collapses entropy fastest (0.035 at step 99). CISPO maintains 2.4x more entropy than GRPO (0.085 vs 0.035) while matching accuracy. Theoretically, this is because CISPO never zeros out gradients — all tokens contribute, preserving diverse reasoning strategies. DRO barely reduces entropy because it barely learns.

#### DRO with multiple gradient steps (num_substeps=4)

With 4 gradient updates per rollout batch (creating mild off-policy conditions), DRO improves substantially:

| Step | DRO (substeps=1) | DRO (substeps=4) | IS (substeps=1) | IS (substeps=4) |
|------|---:|---:|---:|---:|
| 10 | 6.6% | 9.5% | 11.3% | 79.9% |
| 20 | 7.4% | 15.7% | 42.9% | 93.0% |
| 30 | 8.7% | 37.5% | 88.7% | 93.6% |
| 40 | 10.1% | 75.9% | 93.3% | 93.6% |

Multi-step updates help DRO (75.9% vs 10.1% at step 40) but IS benefits even more. The `num_substeps` parameter effectively multiplies gradient updates per batch; IS with substeps=4 reaches 93% at step 20 — equivalent to IS substeps=1 at step 40, as expected.

### Why these results make sense

**IS, GRPO, CISPO converge to the same final accuracy** because GSM8K is learnable at this scale. The loss function affects the learning trajectory, not the destination. This is consistent with published work: [a minimalist approach to LLM reasoning](https://arxiv.org/abs/2504.11343) shows even simple rejection sampling achieves competitive performance with GRPO, and [Stabilizing RL with LLMs](https://arxiv.org/abs/2512.01374) finds that "once training is stabilized, prolonged optimization consistently yields comparable final performance." Note that on harder benchmarks (MATH-500, AIME), the loss function choice may matter more for final accuracy — [CISPO benchmarks on Qwen2.5-7B](https://arxiv.org/abs/2602.06717) show it achieving higher asymptotic reward than DAPO and GRPO on harder math problems.

**GRPO is the most reproducible** (not the fastest) because clipping constrains both the magnitude and direction of each update. This reduces variance across seeds but does not accelerate learning on average. While [Henderson et al. (2018)](https://arxiv.org/abs/1709.06560) established that seed variance is a major problem across all RL methods, our finding that GRPO's clipping specifically reduces this variance in LLM RL is novel — we are not aware of prior work measuring seed-to-seed variance of GRPO vs IS/CISPO in this setting. Caveat: 2 seeds is a limited sample; more seeds would strengthen this claim.

**CISPO maintains higher entropy** because it never zeros out the gradient for any token. In the CISPO gradient, the clipped ratio acts as a scalar weight on `∇ log p_θ` — even tokens with extreme importance ratios contribute. In GRPO, the `min` operation completely drops tokens whose ratio exceeds the clip threshold, removing them from the gradient. Over many steps, this causes GRPO to concentrate probability faster (lower entropy). This entropy preservation is directly confirmed by the [MiniMax-M1 paper](https://arxiv.org/abs/2506.13585) which introduced CISPO to address this token-dropout problem. [DAPO](https://arxiv.org/abs/2503.14476) independently identified the same issue and introduced "Clip-Higher" (asymmetric clipping) as a mitigation.

**GRPO's per-token GPU cost is 2.6x higher.** In published work, full PPO's ~2x overhead is typically attributed to the critic (value) network — [ReMax](https://arxiv.org/abs/2310.10505) reports >2x GPU memory, and [GRPO](https://arxiv.org/abs/2402.03300) saves ~50% by eliminating the critic. Our setup already uses GRPO (no critic), so the 2.6x overhead comes purely from the clipped loss computation itself (evaluating both clipped and unclipped objectives and taking their `min`). This overhead may be implementation-specific to Tinker's server-side loss evaluation.

**DRO is too conservative for on-policy learning** because the quadratic penalty `0.5β(log p_θ/q)²` directly resists policy change. To go from 10% to 94% accuracy, the policy must make large distributional shifts that DRO penalizes. With `num_substeps=4`, each rollout batch gets reused for 4 gradient steps — by the 4th step, the data is mildly off-policy, and DRO's conservatism becomes more appropriate. DRO's design for off-policy robustness is well-established theoretically ([DRO survey](https://arxiv.org/abs/2411.02549)), and [Distributionally Robust RLHF](https://arxiv.org/abs/2503.00539) shows improvements on out-of-distribution data. However, no published work tests DRO as a direct policy gradient loss for on-policy LLM math training — our finding that it fails dramatically in this regime is empirically novel.

### MATH-500 (Qwen3-8B, 100 steps, lr=2e-5, group_size=16, groups_per_batch=64)

MATH-500 is substantially harder than GSM8K (starting accuracy ~0% vs ~8%). This reveals a dynamic hidden on easy tasks:

| Step | IS | GRPO | CISPO |
|------|---:|---:|---:|
| 30 | 6.6% | **10.6%** | 5.2% |
| 40 | 20.6% | **32.8%** | 21.6% |
| 50 | 49.0% | **59.2%** | 44.8% |
| 60 | 67.2% | **74.0%** | 66.6% |
| 70 | 74.6% | 74.2% | **74.8%** |
| 80 | 75.0% | 75.4% | 72.6% |
| 90 | 75.2% | **76.4%** | 74.0% |

**GRPO converges fastest on hard problems** — leading by 10-15 percentage points during mid-training (steps 40-60). On GSM8K this lead was within seed variance; on MATH it's substantial. GRPO's clipping prevents the policy from overshooting on complex multi-step reasoning, which matters more when problems are harder and the reward signal is sparser.

**All three converge to ~75% by step 70-90.** The loss function affects the trajectory, not the ceiling. The 2% gap at step 90 (GRPO 76.4%, IS 75.2%, CISPO 74.0%) may be noise or a small real advantage for GRPO on hard tasks — more seeds would clarify.

**CISPO's entropy preservation is most dramatic on MATH:** 0.180 at step 99 vs GRPO's 0.039 (4.6x). Despite this diversity, CISPO reaches 74.8% at step 70, matching GRPO's 74.2% at the same step. The entropy difference could matter for downstream tasks that benefit from diverse reasoning strategies (e.g., best-of-N sampling, where more diverse policies generate better candidate pools).

### Arithmetic (Llama-3.2-1B, 50 steps)

On this toy task (single-token addition answers), all four loss functions converge within 3-12 steps. The per-step wall time is identical (~11s) and per-token training cost is identical because sequences are too short (~5 tokens) for the loss function computation to matter.

### Cross-benchmark summary

| Property | GSM8K (easy) | MATH-500 (hard) |
|----------|-------------|-----------------|
| Final accuracy | All ~94% | All ~75% |
| GRPO mid-training lead | Within seed noise | Real: 10-15% at steps 40-60 |
| Convergence step | ~40 | ~70 |
| CISPO entropy vs GRPO | 2.4x | 4.6x |
| GRPO per-token cost | 2.6x IS/CISPO | 2.6x IS/CISPO |

**The harder the task, the more GRPO's clipping helps during training** — but the final accuracy converges regardless. CISPO's entropy preservation grows more pronounced on harder tasks. IS is the simplest and cheapest option when you don't need GRPO's stability or CISPO's diversity.

## Choosing a loss function

| Scenario | Recommended (`loss_fn`) | Why |
|----------|------------|-----|
| Default / getting started | `importance_sampling` | Simple, effective, cheapest per token |
| Hard tasks + fastest convergence | `ppo` (= GRPO) | 10-15% ahead mid-training on MATH; 2.6x GPU cost |
| Maximize policy diversity | `cispo` | 4.6x more entropy than GRPO; same GPU cost as IS |
| Off-policy / stale rollout data | `dro` | Quadratic penalty prevents distributional collapse |
| Best-of-N sampling downstream | `cispo` | Diverse policies produce better candidate pools |

### Limitations

- **2 seeds** is limited for reproducibility claims. Henderson et al. recommend 5+ seeds.
- **GRPO's 2.6x per-token cost** may be specific to Tinker's server-side implementation. Other frameworks may have different overhead profiles for the clipping computation.
- **DRO's beta default** is undocumented. An extensive beta sweep might find a setting that performs better on on-policy tasks.
- **AIME-level problems** were not tested. GRPO's mid-training advantage may grow further on competition math.

## Configuration reference

All RL loss functions accept the same `loss_fn_inputs` in each Datum:
- `target_tokens`: The token sequence to compute logprobs on
- `logprobs`: Sampling logprobs from the policy that generated the completion
- `advantages`: Per-token advantages (group-relative, mean-centered: `R - mean(R)`, no std normalization)

Optional `loss_fn_config` parameters:

| Loss | Parameter | Default | Description |
|------|-----------|---------|-------------|
| `ppo` | `clip_low_threshold` | 0.8 | Lower clip bound for IS ratio |
| `ppo` | `clip_high_threshold` | 1.2 | Upper clip bound for IS ratio |
| `cispo` | `clip_low_threshold` | 0.8 | Lower clip bound for IS weight |
| `cispo` | `clip_high_threshold` | 1.2 | Upper clip bound for IS weight |
| `dro` | `beta` | — | Strength of quadratic KL penalty (try 0.01–0.1) |

## References

Papers referenced in this recipe and how they relate to our findings:

- **[MiniMax-M1](https://arxiv.org/abs/2506.13585)** — Introduced CISPO. Shows PPO drops "fork tokens" (rare correction/backtracking tokens) from the gradient, while CISPO preserves them via detached clipped weights. Our entropy measurements (CISPO 2.4-4.6x PPO) quantify what this paper describes qualitatively.
- **[DAPO](https://arxiv.org/abs/2503.14476)** — Independently identified PPO's entropy collapse and proposed "Clip-Higher" (asymmetric clipping) as mitigation. Confirms the token-dropout mechanism we observe.
- **[GRPO](https://arxiv.org/abs/2402.03300)** (DeepSeekMath) — Eliminates the critic network from PPO, saving ~50% compute. Our setup uses critic-free PPO; the 2.6x overhead we measure comes from the loss computation alone, not the critic.
- **[A Minimalist Approach to LLM Reasoning](https://arxiv.org/abs/2504.11343)** — Shows even simple rejection sampling matches GRPO/PPO final performance. Consistent with our finding that IS/PPO/CISPO converge to the same accuracy.
- **[Stabilizing RL with LLMs](https://arxiv.org/abs/2512.01374)** — Finds "prolonged optimization consistently yields comparable final performance." Matches our cross-benchmark result that loss function affects trajectory, not ceiling.
- **[Henderson et al. (2018)](https://arxiv.org/abs/1709.06560)** — Established that seed variance is a major problem in RL. Our finding that GRPO's clipping reduces seed variance in LLM RL (|Δ|=0.2% vs IS 9.7%) extends this to the LLM setting.
- **[REINFORCE++](https://arxiv.org/abs/2501.03262)** — Argues PPO's stability benefits are "less of a concern for LLMs." Our MATH-500 results suggest this understates PPO's value on hard tasks, where its mid-training lead is 10-15%.
- **[ReMax](https://arxiv.org/abs/2310.10505)** — Reports >2x GPU cost for full PPO (with critic). Our 2.6x cost for GRPO (critic-free, clipping only) comes from the loss computation alone — a distinct overhead source.
- **[Distributionally Robust RLHF](https://arxiv.org/abs/2503.00539)** — Applies DRO to RLHF reward models and shows improvements on OOD data. Our work tests DRO as a direct policy gradient loss, finding it fails on-policy but partially recovers with multi-step updates.
- **[DRO Survey](https://arxiv.org/abs/2411.02549)** — Comprehensive survey confirming DRO is designed for worst-case distributional robustness, consistent with our finding that it needs off-policy conditions.
- **[F-GRPO](https://arxiv.org/abs/2602.06717)** — Shows CISPO achieves higher asymptotic reward than DAPO/GRPO on harder math problems (Qwen2.5-7B). Suggests our equal-accuracy finding on GSM8K/MATH may not hold on competition-level benchmarks (AIME).
- **[ASPO](https://arxiv.org/abs/2510.06062)** — Argues IS ratio mismatches lead to "premature convergence" on difficult problems. May explain why IS slightly trails GRPO on MATH-500.
- **[DISPO](https://arxiv.org/abs/2602.00983)** — Reports even better entropy preservation than CISPO. Positions CISPO's entropy as moderate, not maximal.
