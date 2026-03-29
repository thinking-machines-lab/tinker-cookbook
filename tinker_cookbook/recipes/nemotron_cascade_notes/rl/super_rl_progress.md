# RL Environment Validation on Super 120B

## Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144
## Baseline: No SFT checkpoint (base model)

### Results Summary

| Env | Reward | frac_mixed | max_tokens | Notes |
|---|---|---|---|---|
| IF-RL | 0.77-1.0 | 1.0/0.0 | 49K | Easy for base model |
| MCQA | 1.0 | 0.0 | 8K | All correct after extraction fix |
| Structured Output | 0.67-0.89 | 1.0 | 49K | Good variance |
| Code RL | 0.75-1.0 | 1.0 | 118K | MBPP easy for Super |
| Long-Context | 0.60-0.65 | 1.0 | 49K | Stable |
| RLHF | 0.50 | 1.0 | 16K | Random baseline (pairwise judge) |
| Workbench | 0.44 | 1.0 | 49K | Ground-truth seeded mocks |
| SWE Agentless (LLM judge) | 0.12-0.46 | 1.0 | 98K | Cascade data better than R2E-Gym |

### Key Findings
1. Super base model is very strong -- IF-RL and MCQA get near-perfect scores
2. All envs produce frac_mixed=1.0 (good training signal) except IF-RL (too easy) and MCQA (too easy)
3. SWE Agentless works best with Cascade SWE dataset (has codebase context)
4. Renderer auto-resolves to `nemotron3`

### SFT Progress
- Step 169/33K, NLL: 0.687 -> 0.547 (20% improvement)
- ~5.5 min/step, 11.5M tokens/step
