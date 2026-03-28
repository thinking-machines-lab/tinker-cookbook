# RL Environment Progress on Super 120B (2026-03-28)

## Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144
## Base model (no SFT checkpoint)

### Results Summary

| Env | Reward | Mixed | max_tokens | Status | Fix |
|---|---|---|---|---|---|
| IF-RL | 0.77-1.0 | 1.0/0.0 | 49K | ✅ Working | — |
| MCQA | 1.0 | 0.0 (all good) | 8K | ✅ Fixed | Expanded answer extraction + overlong partial credit |
| Structured Output | 0.67-0.89 | 1.0 | 49K | ✅ Working | — |
| Code RL | 0.75-1.0 | 1.0 | 118K | ✅ Working | — |
| Long-Context | 0.60-0.65 | 1.0 | 49K | ✅ Working | — |
| RLHF | 0.50 | 1.0 | 16K | ✅ Working | — |
| Workbench | 0.44 | 1.0 | 49K | ✅ Fixed | Ground-truth seeded mocks + partial credit |
| SWE Agentless (LLM judge) | 0.12-0.20 | 1.0 | 98K | ✅ Working | Cascade SWE dataset |
| SWE Agentless (execution) | TBD | — | 98K | ⚠️ Sandbox errors | Modal sandbox timeouts on pandas |

### Key Findings
1. Super base model is very strong — IF-RL and MCQA get near-perfect scores
2. MCQA needed 8K+ max_tokens (4K caused overlong) + better answer extraction
3. SWE Agentless works with Cascade data + LLM judge (paper's approach)
4. Execution mode has Modal sandbox reliability issues

### SFT Progress
- Step 169/33K, NLL: 0.687 → 0.547 (20% improvement)
- ~5.5 min/step, 11.5M tokens/step
- Running via nohup, persistent to ~/experiments/
