# Nemotron-Cascade-2 Replication Notes

Organized investigation notes for the Nemotron-Cascade-2 replication project.

## Directory Structure

```
notes/
├── README.md                  # This file
├── status.md                  # Current overall status and next steps
├── data/
│   ├── sft_download.md        # SFT data download tracking
│   └── rl_data.md             # RL data analysis and cleaning
├── sft/
│   ├── hyperparams.md         # SFT hyperparameter findings
│   └── training_runs.md       # SFT run logs and results
├── rl/
│   ├── shared_findings.md     # Cross-environment RL findings
│   ├── if_rl.md               # IF-RL environment notes
│   ├── mcqa.md                # MCQA environment notes
│   ├── structured_output.md   # Structured output env notes
│   ├── workbench.md           # Workbench tool calling notes
│   ├── rlhf.md                # RLHF with GenRM notes
│   ├── code_rl.md             # Code RL notes
│   ├── longctx.md             # Long-context RL notes
│   ├── swe_agentless.md       # SWE agentless notes
│   └── swe_agentic.md         # SWE agentic notes
└── eval/
    ├── benchmarks.md          # Benchmark setup and results
    └── paper_evals.md         # Paper's eval suite analysis
```

## Key References
- Paper: arxiv:2603.19220
- Data: nvidia/Nemotron-Cascade-2-SFT-Data, nvidia/Nemotron-Cascade-2-RL-data
- Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- Wandb: https://wandb.ai/thinking-machines-lab-inc/nemotron-cascade-2-replication
- Branch: nemotron-cascade-2-replication on fork
