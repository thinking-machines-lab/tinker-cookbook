# Overall Status (2026-03-27, evening)

## Key Milestones Achieved
- All 8 SFT data subsets downloaded (24.5M examples, 553GB)
- 9 RL environments built, 7 producing non-zero reward
- 8 benchmark evaluations available
- SFT v2 LR sweep running (rank=64, 49K tokens)
- Paper-matched RL sweep running (group=16, batch=32, 49K tokens)
- R2E-Gym integration analysis complete

## SFT v2 LR Sweep (50K sample, rank=64, 49K tokens)
| LR | Steps | Min NLL | Status |
|----|-------|---------|--------|
| 1e-4 | 496 | 0.426 | Running |
| 3e-4 | 498 | **0.424** | Best |
| 5e-4 | 475 | 0.426 | Running |

Decision: lr=3e-4 or 5e-4 for full SFT v2 (both very close).

## RL Environments (with fixes)
| Env | Reward | Mixed Groups | Status |
|-----|--------|-------------|--------|
| IF-RL | 0.69-0.79 | 100% | Working, lr=3e-5 best |
| MCQA (fixed) | 0.36→0.40 | Yes | Improving! Fix worked |
| StructOut (jsonschema) | 0.56→0.79 | Yes | Strong learning signal |
| Long-ctx (fixed) | 0.51-0.61 | Yes | 6x improvement from judge fix |
| Code RL (g=16) | 0.06-0.09 | 100% | Works with enough rollouts |
| RLHF (fixed) | 0.008 | Testing g=16 | GenRM bugs fixed |
| Workbench | -0.006 | 100% | Tool format working, mock data issue |
| SWE Agentless | 0.0 | — | Needs R2E-Gym Docker |
| SWE Agentic | 0.0 | — | Needs R2E-Gym Docker |

## Key Findings
1. LoRA LR = 10x paper's full-FT LR (SFT: 3-5e-4, RL: 3e-5)
2. group_size=16 is critical — creates mixed groups even for hard tasks
3. Long-ctx judge needed 512 tokens (was 32) — thinking models need reasoning space
4. MCQA needed <think> stripping before answer extraction
5. StructOut needed real jsonschema validation (was too easy before)
6. RLHF had two bugs: wrong API param + wrong dataset field
7. R2E-Gym Docker images can solve SWE dependency issues

## Wandb
https://wandb.ai/thinking-machines-lab-inc/nemotron-cascade-2-replication

## Next Steps
1. Finish SFT v2 sweep → pick lr → launch full SFT v2 (24.5M examples, rank=64, 49K)
2. Run full RL cascade: SFT v2 → IF-RL (180 steps) → Multi-domain (70 steps)
3. Benchmark after each stage
4. Implement R2E-Gym Docker integration for SWE envs
5. Ground-truth-seeded mocks for Workbench
