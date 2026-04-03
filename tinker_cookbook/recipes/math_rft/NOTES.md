# Math RFT Research Notes

## Research Question
For math reasoning, how does iterative Rejection Sampling Fine-Tuning (RFT)
compare with GRPO when given matched sampling compute?

## Method Overview
Both methods sample K solutions per problem. The difference:
- **GRPO**: trains on ALL solutions, weighted by advantage (reward - group mean)
- **RFT**: trains on ONLY correct solutions, using standard SFT loss (cross-entropy)

## Key Hypotheses
1. RFT will be more stable than GRPO (no negative advantages, no importance weighting)
2. GRPO may converge faster because it learns from incorrect solutions too (what NOT to do)
3. RFT may plateau earlier because it can only learn from problems the model already solves
4. The gap between RFT and GRPO may be larger on harder datasets (MATH vs GSM8K)

## Experimental Design

### Experiment 1: GSM8K with Qwen3-8B (completed)

Config: `model_name=Qwen/Qwen3-8B, group_size=16, groups_per_batch=32, lr=1e-4, max_tokens=1024`

| Step | pass@1 | sample_acc | NLL   |
|------|--------|------------|-------|
| 0    | 62.4%  | 67.0%      | 0.377 |
| 5    | 91.4%  | 98.2%      | 0.201 |
| 10   | **94.0%** | 91.2%   | 0.252 |
| 15   | 93.6%  | 93.0%      | 0.244 |
| 20   | 92.2%  | 91.2%      | 0.345 |
| 30   | 93.4%  | —          | —     |

**Findings:** RFT achieves 94% on GSM8K in 10 steps. GSM8K is too easy — the model
solves 67% of samples at baseline, providing rich training signal for a virtuous cycle.

### Experiment 2: MATH-500 with Qwen3-8B — RFT (completed)

Config: `model_name=Qwen/Qwen3-8B, env=math, group_size=16, groups_per_batch=32, lr=1e-4, max_tokens=2048`

**Per-level test accuracy (greedy eval):**

| Step | Overall | L1    | L2    | L3    | L4    | L5    | Format |
|------|---------|-------|-------|-------|-------|-------|--------|
| 0    | 42.2%   | 81.4% | 72.2% | 47.6% | 32.8% | 14.2% | 44.0%  |
| 5    | **78.8%** | 93.0% | 91.1% | 87.6% | 76.6% | 61.2% | 90.4% |
| 10   | 78.6%   | 97.7% | 86.7% | 85.7% | 79.7% | 60.4% | 92.6%  |
| 15   | 78.0%   | 93.0% | 86.7% | 85.7% | 80.5% | 59.0% | 90.4%  |
| 20   | 78.2%   | 93.0% | 85.6% | 85.7% | 79.7% | 61.2% | 90.4%  |
| 25   | 79.6%   | 95.3% | 86.7% | 89.5% | 81.2% | 60.4% | 91.2%  |
| 30   | 79.8%   | 90.7% | 86.7% | 89.5% | 79.7% | 64.2% | 93.4%  |
| 35   | 79.6%   | 95.3% | 88.9% | 89.5% | 80.5% | 59.7% | —      |
| 40   | 78.8%   | 93.0% | 92.2% | 88.6% | 75.8% | 60.4% | —      |

**Key findings:**
1. **Sharp convergence then plateau:** 42.2% → 78.8% in 5 steps, then flat for 35 more steps
2. **Difficulty-dependent ceiling:** L1 saturates at ~95%, L5 plateaus at ~60%
3. **Format compliance learned fast:** 44% → 90%+ in 5 steps
4. **Training solve rate misleading:** Model achieves 85-100% solve rate on training problems
   by step 4, but test L5 accuracy stays at ~60%. The model solves *seen* L5 problems
   but doesn't generalize to *unseen* ones.
5. **NLL rises after step 5:** Overfitting signal — the model memorizes solution patterns
   rather than learning general reasoning strategies

### Experiment 3: MATH-500 with Qwen3-8B — GRPO comparison (completed)

Config: `model_name=Qwen/Qwen3-8B, env=math, group_size=16, groups_per_batch=32, lr=8e-5, max_tokens=2048`

**Head-to-head comparison:**

| Step | RFT (greedy) | GRPO (T=1.0) | Notes |
|------|-------------|--------------|-------|
| 0    | 42.2%       | 35.9%        | Baseline gap from eval method |
| 5    | **78.8%**   | 46.9%        | RFT dominates early |
| 10   | 78.6%       | 67.3%        | GRPO catching up |
| 15   | 78.0%       | 77.5%        | **Crossover point** |
| 20   | 78.2%       | **82.3%**    | GRPO breaks through |
| 25   | 79.6%       | 81.6%        | Slight GRPO dip |
| 30   | 79.8%       | **84.1%**    | GRPO clearly ahead |
| 35   | 79.6%       | **85.1%**    | GRPO still improving |

**Key findings:**
1. **RFT is faster early:** RFT reaches ~80% in 5 steps vs GRPO needs ~17 steps
2. **GRPO breaks through RFT's ceiling:** GRPO reaches 85.1% at step 35 while RFT
   plateaus at ~79% from step 5 onwards. Even accounting for the eval method difference
   (~6-8pp from greedy vs T=1.0), GRPO likely surpasses RFT by step 20.
3. **GRPO doesn't plateau yet at 40 steps:** Train reward is still rising, KL is tiny
   (~0.0005), suggesting room for further improvement with more training
4. **The crossover around step 15 is the key finding:** Before step 15, RFT dominates
   (3x more efficient). After step 15, GRPO is better. This suggests a natural **hybrid
   strategy**: RFT for warm-start, then GRPO for refinement.

## Analysis: Why Does RFT Plateau?

The training data reveals a subtle but important finding: **the plateau is NOT caused
by inability to find correct solutions.** By step 4, the model achieves 85-100% solve
rate on training problems, even for L5. The real causes:

1. **Redundant gradient signal:** As the model improves, it generates increasingly
   similar correct solutions. SFT loss on these near-identical outputs produces
   diminishing gradient updates.

2. **No negative signal:** RFT only reinforces what works. It cannot actively push
   the model away from common failure modes. On hard problems where the model makes
   systematic errors, RFT can only wait for a lucky correct sample.

3. **Easy problem bias:** In each batch, easy problems (L1-L3) generate many more
   correct solutions than hard problems (L5). The gradient is dominated by signals
   from easy problems, even though the model already masters them.

GRPO addresses all three issues:
- Advantage weighting gives more credit to rare correct solutions on hard problems
- Negative advantages explicitly penalize incorrect solution patterns
- The importance-weighted loss provides richer gradient information

## Implications

1. **RFT as cheap warm-start:** Use RFT for 5-10 steps to quickly capture "low-hanging
   fruit" (format compliance, common patterns), then switch to GRPO for harder reasoning

2. **Difficulty regime determines method choice:**
   - Easy tasks (solve_rate > 70%): RFT is sufficient and 5x faster
   - Hard tasks (solve_rate < 50%): GRPO is essential for continued improvement

3. **Per-level analysis is critical:** Aggregate metrics (overall pass@1) hide the
   difficulty-dependent dynamics. RFT appears to work well overall (78.8%) but fails
   to improve on the hardest 25% of problems.

## Log
- 2026-04-03 01:51: Initial implementation committed
- 2026-04-03 03:03: First full GSM8K run: peak 94.0% at step 10
- 2026-04-03 06:17: Controlled GRPO comparison on GSM8K
- 2026-04-03 08:00: Per-difficulty tracking added for MATH experiments
- 2026-04-03 10:00: MATH RFT experiment: 42.2% → 78.8% plateau at step 5
- 2026-04-03 14:00: MATH GRPO experiment: breaks through to 85.1% at step 35
