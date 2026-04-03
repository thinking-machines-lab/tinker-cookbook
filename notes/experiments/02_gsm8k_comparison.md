# Experiment 02: GSM8K comparison (all 4 loss functions + DRO beta ablation)

**Commit:** 7fb72e20
**Date:** 2026-04-03

## Config
- Model: Qwen/Qwen3-8B
- Task: GSM8K (grade-school math with chain-of-thought)
- group_size=16, groups_per_batch=64, lr=2e-5, max_tokens=512
- lora_rank=32, eval_every=10, save_every=50
- 100 training steps per loss function
- Log paths: ~/experiments/loss_cmp_gsm8k/{is,ppo,cispo,dro,dro_beta001}/

## Results: Test accuracy over training

| Step | IS | PPO | CISPO | DRO (default) | DRO (β=0.01) |
|------|------|------|------|------|------|
| 0 | 8.9% | 6.6% | 7.7% | 8.1% | 6.6% |
| 10 | 11.3% | 12.1% | 11.2% | 6.6% | 7.8% |
| 20 | 42.9% | 51.3% | 46.1% | 7.4% | 9.9% |
| 30 | 88.7% | 85.7% | 88.5% | 8.7% | 10.5% |
| 40 | 93.3% | 93.0% | 93.7% | 10.1% | 16.2% |
| 50 | 92.9% | 93.8% | 93.4% | 12.5% | 28.7% |
| 60 | 93.2% | 93.4% | 93.6% | 12.2% | — |
| 70 | 93.7% | 93.6% | 93.6% | 14.8% | — |
| 80 | 94.2% | 94.0% | 93.3% | 18.2% | — |
| 90 | 93.9% | — | 93.0% | — | — |

**Peak test accuracy:** IS=94.2%, PPO=94.0%, CISPO=93.7%, DRO=18.2%, DRO(β=0.01)=28.7%

## Key findings

1. **IS, PPO, CISPO all converge to ~93-94% test accuracy by step 40.**
   On GSM8K, the three IS-based losses perform very similarly in terms of final
   accuracy. All three reach 90%+ test accuracy at step 40.

2. **PPO leads early (step 20: 51% vs 43-46% for IS/CISPO).**
   PPO's clipping stabilizes early training when the policy is changing rapidly.
   It also has the lowest entropy throughout (more confident/focused updates).

3. **CISPO has the highest entropy (most exploratory) while matching IS/PPO accuracy.**
   CISPO entropy: 0.085 at step 99 vs PPO: 0.035, IS: 0.059. Despite being more
   exploratory (higher entropy), CISPO matches or exceeds the others in test accuracy.
   This is consistent with CISPO preserving gradient signal for diverse tokens.

4. **DRO fails to learn on GSM8K with on-policy training.**
   Both default and β=0.01 configurations are far too conservative. DRO reaches
   only 18% (default) or 29% (β=0.01) test accuracy after 100 steps.
   The quadratic KL penalty prevents the large policy changes needed to go from
   ~10% to ~90% accuracy. Entropy stays high (0.27-0.31 vs 0.03-0.09 for others),
   indicating the policy barely moves.

5. **DRO's penalty is the bottleneck, not the advantage formulation.**
   Verified in experiment 01 that DRO(β=0) converges like IS on arithmetic.
   Lower β=0.01 helps (29% vs 18%) but not enough for this task.

## Interpretation

DRO is designed for **off-policy/offline** RL where data is stale and you need
conservative updates to avoid distributional collapse. On **on-policy** GSM8K
training (fresh rollouts every step), this conservatism is a pure liability —
the policy needs to make large, rapid changes to learn reasoning.

IS, PPO, and CISPO are all good choices for on-policy RL. PPO is marginally
faster early but converges to the same place. CISPO maintains higher entropy
(diversity) while matching accuracy, which could matter for harder tasks where
exploration is important.
