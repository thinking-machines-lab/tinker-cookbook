# Loss Function Comparison Recipe — Research Plan

## Research question

How do Tinker's built-in RL loss functions compare on the same task, and when should a user choose one over another?

## Status

### Done
- [x] Experiment 01: Arithmetic baseline (all 4 loss functions, 50 steps)
  - Result: All converge quickly. DRO 3-4x slower. Task too easy to differentiate.

### Open questions

1. **Does a harder task show meaningful differences?**
   GSM8K with longer CoT would test CISPO's gradient preservation for rare tokens
   and give DRO more room to show its conservative but robust behavior.

2. **Does DRO's slowness persist with tuned beta?**
   We used default beta. Lower beta = weaker penalty = faster convergence. The
   tradeoff between speed and robustness is the interesting thing to show.

3. **Is the recipe valuable without a harder experiment?**
   The arithmetic results + README explaining the theory might be sufficient for a
   cookbook recipe. Users can run GSM8K themselves. The recipe's value is:
   - Showing that loss_fn is a one-line config change
   - Providing the analysis script
   - Documenting when to use each loss function with formulas

## Possible next experiments

### Option A: GSM8K comparison (expensive but informative)
- Model: Qwen/Qwen3-8B, group_size=16, groups_per_batch=64, max_tokens=512
- 100 steps per loss function (4 runs)
- Would take ~2-4 hours total
- Pros: real task, longer outputs, actual reasoning
- Cons: expensive, slow iteration

### Option B: DRO beta sweep on arithmetic (cheap, focused)
- Test beta in {0.01, 0.05, 0.1, 0.5} to show the speed/robustness tradeoff
- 20 steps each, arithmetic
- Quick to run, shows how to tune DRO

### Option C: Ship as-is with arithmetic results + good docs
- The README already explains the theory well
- Arithmetic results confirm the basic behavior matches theory
- Users who need more can run GSM8K themselves
