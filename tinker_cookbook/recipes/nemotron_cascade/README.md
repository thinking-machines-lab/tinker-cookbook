# Nemotron Cascade 2 Replication

Replicates NVIDIA's [Nemotron-Cascade 2](https://arxiv.org/abs/2603.19220) post-training pipeline using the Tinker API.

## Model

`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144` — MoE (12B active), LoRA rank 64, 262K context.

## Pipeline

### 1. Supervised Fine-Tuning

Trains on 8 data domains from `nvidia/Nemotron-Cascade-2-SFT-Data` using `InterleavedChatDatasetBuilder`.

```bash
python -m tinker_cookbook.recipes.nemotron_cascade.run_super_sft
```

- **Data**: ~24.5M examples across math, science, chat, IF, safety, conversational agent, SWE, terminal agent
- **Config**: batch=2048, lr=3e-4 cosine, rank=64, β₂=0.98, 33K steps
- **Paper reference**: Table 7, Appendix B

### 2. Stage 1: IF-RL (Instruction Following)

```bash
python -m tinker_cookbook.recipes.nemotron_cascade.train_rl \
    env=if_rl \
    load_checkpoint_path=<sft_checkpoint> \
    group_size=16 groups_per_batch=32 \
    max_tokens=49152 max_steps=180
```

- **Reward**: Programmatic (48 IFEval instruction types)
- **Paper**: ~180 steps, batch=128×16, lr=3e-6 (we use 3e-5 for LoRA)

### 3. Stage 2: Multi-Domain RL

Run each environment independently:

```bash
# MCQA (55% of multi-domain blend)
python -m tinker_cookbook.recipes.nemotron_cascade.train_rl \
    env=mcqa load_checkpoint_path=<ifrl_checkpoint> \
    group_size=16 groups_per_batch=32 max_tokens=8192 max_steps=70

# Workbench / Tool-calling (30%)
python -m tinker_cookbook.recipes.nemotron_cascade.train_rl \
    env=workbench load_checkpoint_path=<ifrl_checkpoint> \
    group_size=16 groups_per_batch=32 max_tokens=49152 max_steps=70

# Structured Output (15%)
python -m tinker_cookbook.recipes.nemotron_cascade.train_rl \
    env=structured_output load_checkpoint_path=<ifrl_checkpoint> \
    group_size=16 groups_per_batch=32 max_tokens=49152 max_steps=70
```

### 4. Additional RL Stages

```bash
# Code RL (118K context — paper-matched)
python -m tinker_cookbook.recipes.nemotron_cascade.train_rl \
    env=code_rl max_tokens=118000 context_window=262144

# RLHF (16K context, KL=0.03)
python -m tinker_cookbook.recipes.nemotron_cascade.train_rl \
    env=rlhf max_tokens=16384

# Long-Context QA (49K context)
python -m tinker_cookbook.recipes.nemotron_cascade.train_rl \
    env=longctx_rl max_tokens=49152

# SWE Agentless (98K context, LLM judge)
python -m tinker_cookbook.recipes.nemotron_cascade.train_rl \
    env=swe_rl max_tokens=98304 context_window=262144

# SWE Agentic (262K context, execution-based)
python -m tinker_cookbook.recipes.nemotron_cascade.train_rl \
    env=swe_agentic max_tokens=262144 context_window=262144
```

### 5. Full Cascade (Orchestrated)

```bash
python -m tinker_cookbook.recipes.nemotron_cascade.run_cascade \
    --sft-checkpoint <path>
```

### 6. Evaluation

```bash
python -m tinker_cookbook.recipes.nemotron_cascade.run_evals \
    --checkpoint <path>
```

## RL Environments

| Environment | Reward Type | Dataset | max_tokens |
|---|---|---|---|
| `if_rl` | Programmatic (IFEval) | NVIDIA IF-RL split | 49K |
| `mcqa` | Programmatic (exact match) | NVIDIA multi-domain | 8K |
| `structured_output` | Programmatic (jsonschema) | NVIDIA multi-domain | 49K |
| `workbench` | Programmatic (tool match + partial credit) | NVIDIA multi-domain | 49K |
| `code_rl` | Execution (MBPP) | MBPP sanitized | 118K |
| `longctx_rl` | LLM judge (Qwen3.5) | NarrativeQA | 49K |
| `rlhf` | GenRM (Kimi K2.5) | HelpSteer3 | 16K |
| `swe_rl` | LLM judge / Execution | Cascade-RL-SWE / R2E-Gym | 98K |
| `swe_agentic` | Execution (Docker) | R2E-Gym-Subset | 262K |

## Key Differences from Paper

| Setting | Paper | This Recipe |
|---|---|---|
| Fine-tuning | Full | LoRA rank 64 |
| LR (RL) | 3e-6 | 3e-5 (10x for LoRA) |
| Model | Nemotron-3-Nano-30B | Nemotron-3-Super-120B |
| Code RL data | 3.5K competitive prog (unreleased) | 257 MBPP (public) |
| RLHF reward model | Qwen3-235B | Kimi K2.5 |
| MOPD stage | Token-level distillation | Not implemented |

## Files

| File | Purpose |
|---|---|
| `train_sft.py` | SFT training CLI (chz) |
| `train_rl.py` | RL training CLI (chz) |
| `run_super_sft.py` | Full SFT launcher for Super 120B |
| `run_cascade.py` | Cascade orchestrator |
| `run_evals.py` | Benchmark evaluation |
| `sft_datasets.py` | SFT dataset builders |
| `utils.py` | Shared utilities (think-block stripping) |
| `*_env.py` | RL environment implementations |
