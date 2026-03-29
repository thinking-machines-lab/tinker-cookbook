# Nemotron Cascade 2 Replication Notes

Investigation notes for the Nemotron-Cascade-2 replication. See `status.md` for current progress.

## Index

| File | Topic |
|---|---|
| `status.md` | Current status, env results, next steps |
| `model_decision.md` | Why Super 120B over Nano 30B |
| `data/sft_download.md` | SFT data download tracking (8 subsets, 24.5M examples) |
| `eval/paper_evals.md` | Paper's benchmark suite and our coverage |
| `rl/shared_findings.md` | Cross-environment RL findings (LR, group size, `<think>` handling) |
| `rl/super_rl_progress.md` | Super base model validation results across all envs |
| `rl/if_rl.md` | IF-RL: programmatic IFEval reward |
| `rl/mcqa.md` | MCQA: answer extraction fixes, partial credit |
| `rl/structured_output.md` | Structured Output: jsonschema validation |
| `rl/code_rl.md` | Code RL: MBPP execution, newline bug fix |
| `rl/longctx_rl.md` | Long-Context: LLM judge token limit fix |
| `rl/rlhf.md` | RLHF: GenRM pairwise reward, API bug fixes |
| `rl/workbench.md` | Workbench: ground-truth seeded mocks, partial credit |
| `rl/swe_rl.md` | SWE Agentless: LLM judge vs execution, Cascade SWE dataset |
| `rl/swe_agentic.md` | SWE Agentic: R2E-Gym Docker integration |
| `rl/r2e_gym.md` | R2E-Gym: dataset analysis, Docker image approach |
| `swe/research_findings.md` | How the paper does SWE RL, Cascade SWE dataset discovery |
| `swe/docker_integration.md` | R2E-Gym Docker images via Modal, bugs fixed |

## References
- Paper: arxiv:2603.19220
- Data: nvidia/Nemotron-Cascade-2-SFT-Data, nvidia/Nemotron-Cascade-2-RL-data, nvidia/Nemotron-Cascade-RL-SWE
- Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144
- Wandb: (internal — see team Wandb project)
