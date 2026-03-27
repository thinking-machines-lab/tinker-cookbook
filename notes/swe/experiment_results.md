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
