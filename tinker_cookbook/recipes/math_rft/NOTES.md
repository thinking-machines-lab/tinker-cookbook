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

### Experiment 1: GSM8K Smoke Test
- Model: Qwen/Qwen3-8B
- Dataset: GSM8K (7473 train, 1319 test)
- group_size=4, groups_per_batch=4, max_tokens=512
- Goal: Verify pipeline works end-to-end

### Experiment 2: GSM8K Full Run (RFT)
- Model: Qwen/Qwen3-8B
- group_size=16, groups_per_batch=64
- max_tokens=1024, learning_rate=2e-5
- Eval every 5 batches, ~117 total batches

### Experiment 3: GSM8K Full Run (GRPO baseline)
- Same model and sampling budget as Experiment 2
- Use existing math_rl recipe

### Experiment 4: MATH dataset comparison (if time permits)
- Same setup but on Hendrycks MATH (harder problems)
- Expect larger gap between methods on harder data

## Key Metrics
- test/correct (pass@1 accuracy on test set)
- train/sample_accuracy (fraction of correct samples during training)
- train/solve_rate (fraction of problems with at least 1 correct solution)
- train/mean_nll (SFT loss on correct solutions)

## Log
- 2026-04-03: Initial implementation committed. Running smoke test.
