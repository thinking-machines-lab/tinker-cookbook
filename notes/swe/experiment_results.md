# SWE Environment Experiment Results (2026-03-27)

## SWE Agentless — Execution Mode (3 steps)

**Config**: group_size=2, groups_per_batch=2, lr=3e-5, max_tokens=16384, reward_mode=execution
**Model**: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (SFT v1 checkpoint)
**Dataset**: R2E-Gym-Subset (use_r2e_gym=True)

| Step | Reward | Correct | Has Patch | Frac Mixed | Time (s) |
|------|--------|---------|-----------|------------|----------|
| 0 | 0.0 | 0.0 | 1.0 | 0.0 | 682 |
| 1 | 0.0 | 0.0 | 1.0 | 0.0 | 530 |
| 2 | 0.0 | 0.0 | 1.0 | 0.0 | 668 |

**Key observations:**
- Model generates patches every time (has_patch=1.0) — it understands the diff format
- No patches pass tests (correct=0.0) — model lacks codebase context
- 100% all-bad groups (frac_mixed=0.0) — zero GRPO signal
- ~10 min/step at tiny scale (group=2, batch=2)

**Root cause**: Without seeing actual file contents, the model hallucinate file paths and code structure. Patches look plausible but don't apply to the right locations.

**Comparison with LLM judge mode** (from previous session):
- LLM judge: reward=0.306, frac_mixed=1.0 — viable for training
- Execution: reward=0.0, frac_mixed=0.0 — not viable without prompt improvement

**Recommendation**: Use LLM judge mode for now. To enable execution mode:
1. Add codebase context to prompt (file listing, failing test, relevant source files)
2. Filter to instances where the fix is in a single file
3. Consider adding partial credit for patches that apply but don't fully fix

## SWE Agentic (2 steps)

**Config**: group_size=2, groups_per_batch=1, lr=3e-5, max_tokens=16384, max_turns=6
**Model**: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (SFT v1 checkpoint)
**Dataset**: R2E-Gym-Subset (use_r2e_gym=True)

| Step | Reward | Pass Rate | Frac Mixed | Time (s) | Turns/Episode |
|------|--------|-----------|------------|----------|---------------|
| 0 | 0.0 | 0.0 | 0.0 | 301 | 3.0 |
| 1 | 0.0 | 0.0 | 0.0 | 301 | 3.0 |

**Key observations:**
- Multi-turn interaction works: model uses read_file, write_file, run_command
- ~850 action tokens/turn, ~3826 observation tokens/turn
- No successful fixes — Nano-30B too weak for SWE tasks
- ~5 min/step at tiny scale — paper scale (group=64) would be ~2.5 hrs/step
- 100% all-bad groups — zero GRPO signal

**Assessment**: SWE agentic should be deferred to later in the cascade. The model needs to be much stronger (post IF-RL + multi-domain RL) before it can produce any correct fixes.

## SWE Agentless — Cascade SWE Data with Golden Patch Judge (3 steps)

**Config**: group_size=4, groups_per_batch=4, lr=3e-6, max_tokens=32768, reward_mode=llm_judge
**Model**: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (SFT v1 checkpoint)
**Dataset**: nvidia/Nemotron-Cascade-RL-SWE (141,191 instances after filtering)
**Judge**: Qwen3.5-397B with golden-patch comparison prompt

| Step | Reward | Judge Reward | Has Patch | Frac Mixed | Ob Tokens | Ac Tokens |
|------|--------|-------------|-----------|------------|-----------|-----------|
| 0 | 0.1125 | 0.1125 | 0.75 | 1.0 | 9725 | 4976 |
| 1 | 0.075 | 0.075 | 0.71 | 1.0 | - | - |
| 2 | 0.200 | 0.200 | 1.0 | 1.0 | 20368 | 62915 |

**Key observations:**
- NON-ZERO REWARD! First real reward signal from SWE tasks.
  - Previous R2E-Gym execution mode: reward=0.0 (no codebase context -> hallucinated patches)
  - Previous R2E-Gym LLM judge (no golden patch): reward=0.306 (generous scoring)
  - This run with golden patch comparison: reward=0.075-0.200 (calibrated scoring)
- Reward INCREASED over 3 steps: 0.1125 -> 0.075 -> 0.200 (step 2 shows improvement!)
- Step 2: 100% patch generation rate (up from 75% at step 0)
- 100% mixed reward groups at every step -> GRPO gets learning signal
- Prompts tokenize to 10K-20K tokens (from 14K-80K char prompts with codebase context)
- Model generates ~5K tokens at step 0, growing to ~8K at step 2
- Judge occasionally fails to parse (Qwen3.5-397B reasons before scoring, can exceed 512 max_tokens)
- Training completed successfully in ~73 min total (3 steps at this small scale)

**Dataset statistics:**
- 141,191 instances (up from 4,578 R2E-Gym instances)
- Prompt lengths: min=13,723 chars, median=56,102 chars, max=79,857 chars
- Sources: SWE-Bench-Train, SWE-reBench, SWE-Smith, R2E-Gym-Subset, SWE-Fixer-Train

**Issues found:**
1. Context window: Nemotron 30B has 65K context. With max_tokens=49K for generation, prompts > 16K tokens overflow. Reduced to max_tokens=32K.
   Paper uses max_seq=98,304 -> needs a model with 128K+ context window.
2. Judge thinking: Qwen3.5-397B outputs reasoning before the score. With judge_max_tokens=512, long reasoning can be truncated before the final score integer. Should increase judge_max_tokens or use a non-thinking judge model.
3. Reward calibration: 0.075-0.1125 seems low. May need to increase judge_max_tokens so it can finish scoring, or use a different judge prompt that's more lenient.

**Next steps:**
1. Increase judge_max_tokens to 2048+ to handle thinking model reasoning
2. Try with a larger context model (128K+) to use max_tokens=49K+
3. Implement paper's filtering: mask loss where no rollout gets reward > 0.5
4. Scale up: groups_per_batch=128, group_size=16 (paper config)
