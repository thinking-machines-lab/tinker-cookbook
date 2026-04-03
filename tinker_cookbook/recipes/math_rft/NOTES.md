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

### Experiment 4: RFT→GRPO Hybrid (warm-start, completed)

Config: 5 steps RFT (lr=1e-4), then 35 steps GRPO (lr=8e-5) from RFT checkpoint.

**Three-way comparison (all at T=1.0 eval):**

| GRPO step | Pure GRPO | Hybrid (RFT+GRPO) | Effective total |
|-----------|-----------|-------------------|-----------------|
| 0         | 35.9%     | 77.1%             | RFT5 + GRPO0    |
| 5         | 46.9%     | 75.8%             | RFT5 + GRPO5    |
| 10        | 67.3%     | 78.3%             | RFT5 + GRPO10   |
| 15        | 77.5%     | 78.5%             | RFT5 + GRPO15   |
| 20        | **82.3%** | 78.6%             | RFT5 + GRPO20   |
| 25        | 81.6%     | 78.7%             | RFT5 + GRPO25   |
| 30        | **84.1%** | **79.3%**         | RFT5 + GRPO30   |
| 35        | **85.1%** | --                | --              |

**Surprising negative result:** The naive RFT→GRPO pipeline *underperforms* pure GRPO.
The hybrid peaks at 79.3%, barely above pure RFT's plateau (79.8% greedy ≈ ~73% T=1.0)
and far below pure GRPO (85.1%).

**Why does the warm-start hurt?** Three possible explanations:

1. **Entropy collapse:** RFT (SFT on correct solutions) dramatically reduces the model's
   output entropy. GRPO needs exploration diversity — if the model is already very
   confident in its (sometimes wrong) solutions, GRPO's importance weights become
   near-uniform, providing weak gradient signal.

2. **Local optimum trapping:** RFT pushes the model into a narrow region of weight
   space that generates correct-looking outputs. GRPO's small per-step updates
   (KL ~0.0005) can't escape this basin. In contrast, GRPO from scratch traverses
   a broader region of weight space.

3. **Distribution shift:** GRPO's importance-weighted loss assumes the current policy
   is close to the reference. After 5 steps of RFT with lr=1e-4, the model has shifted
   significantly, potentially destabilizing the importance weights early on.

This finding has important implications for multi-stage training pipelines like those
used by DeepSeek-R1, which employs RFT as a warm-up stage before RL. Our result
suggests this works only with careful tuning of the transition (e.g., LR warmup,
gradual mixing) rather than a naive checkpoint hand-off.

## Key Takeaways

1. **RFT is a "fast ceiling" method:** Rapid convergence (5 steps) to a ceiling
   determined by the model's initial capability and the task difficulty. Ideal for
   easy-to-moderate tasks where the ceiling is acceptable.

2. **GRPO is a "slow breakthrough" method:** Slower convergence but continues improving
   beyond RFT's ceiling. Essential for hard tasks requiring reasoning improvement.

3. **Naive warm-starting hurts:** Initializing GRPO from an RFT checkpoint does NOT
   combine their strengths. RFT's entropy collapse prevents GRPO from exploring
   effectively. A more careful transition is needed.

4. **The crossover point depends on task difficulty:**
   - GSM8K (easy): RFT dominates — GRPO never catches up in 30 steps
   - MATH (hard): GRPO overtakes RFT at step ~15 and pulls away

5. **Per-level analysis is essential:** Aggregate metrics hide difficulty-dependent
   dynamics. On MATH, RFT's 79% overall hides that L5 is stuck at 60%.

## Follow-Up Research Ideas

### Idea 1: RFT with Entropy Regularization
Add a KL penalty to the RFT loss: `L = CE(correct_solutions) + β * KL(π || π_ref)`.
This preserves exploration ability while training on correct solutions, potentially
avoiding the local optimum that prevents GRPO from improving after RFT.
**Expected impact:** If this works, it could combine RFT's speed with GRPO's ceiling.

### Idea 2: Frontier-Focused GRPO
Only apply GRPO updates to "frontier" problems where solve rate is between 20-80%.
Easy problems (solve_rate ≈ 1) contribute near-zero useful gradient. Impossible
problems (solve_rate ≈ 0) also contribute noise. Focusing on the frontier maximizes
the information per gradient step.
**Expected impact:** Could make GRPO 2-3x more sample-efficient.

### Idea 3: Characterizing the Local Optimum
Measure the output entropy and solution diversity of RFT-trained vs GRPO-trained
models. If RFT collapses entropy, this explains the hybrid failure and suggests
entropy-preserving modifications. Compare the KL divergence from the base model
for both methods at the same test accuracy.
**Expected impact:** Theoretical understanding of why methods differ.

### Idea 4: STaR-Style Rationalization for Unsolvable Problems
For problems where all K samples are wrong, provide the correct answer and ask the
model to generate a step-by-step solution (rationalization). Train on these
rationalizations alongside naturally correct solutions. This directly addresses
RFT's blind spot on unsolvable problems.
**Expected impact:** Could break through RFT's ceiling without needing full RL.

### Idea 5: Interleaved RFT-GRPO
Instead of sequential (RFT then GRPO), alternate within each batch:
- For problems with solve_rate > 80%: apply RFT loss (efficient on easy problems)
- For problems with solve_rate < 80%: apply GRPO loss (learns from failures)
**Expected impact:** Gets benefits of both methods without the warm-start problem.

### Idea 6: Difficulty-Aware Curriculum
Train on problems ordered by difficulty (L1→L5). As the model masters easy
problems, shift the distribution toward harder ones. This is natural for RFT
because the solve rate determines whether a problem contributes training signal.
**Expected impact:** More efficient use of training compute.

## Log
- 2026-04-03 01:51: Initial implementation committed
- 2026-04-03 03:03: First full GSM8K run: peak 94.0% at step 10
- 2026-04-03 06:17: Controlled GRPO comparison on GSM8K
- 2026-04-03 08:00: Per-difficulty tracking added for MATH experiments
- 2026-04-03 10:00: MATH RFT experiment: 42.2% → 78.8% plateau at step 5
- 2026-04-03 14:00: MATH GRPO experiment: breaks through to 85.1% at step 35
- 2026-04-03 18:00: Hybrid RFT→GRPO experiment: surprising negative result (79.3% < 85.1%)
- 2026-04-03 19:00: Analysis and follow-up ideas written up
