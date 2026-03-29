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
- **Config**: batch=2048, lr=3e-4 cosine, rank=64, beta2=0.98, 33K steps
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
- **Paper**: ~180 steps, batch=128x16, lr=3e-6 (we use 3e-5 for LoRA)

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

## RL Environment Reference

### Overview

| Environment | Reward Type | Dataset | max_tokens | External Deps |
|---|---|---|---|---|
| `if_rl` | Programmatic (IFEval) | NVIDIA IF-RL split | 49K | None |
| `mcqa` | Programmatic (exact match) | NVIDIA multi-domain | 8K | None |
| `structured_output` | Programmatic (jsonschema) | NVIDIA multi-domain | 49K | `jsonschema` |
| `workbench` | Programmatic (tool match) | NVIDIA multi-domain | 49K | None |
| `code_rl` | Execution (Modal sandbox) | MBPP sanitized | 118K | Modal |
| `longctx_rl` | LLM judge (Qwen3.5) | NarrativeQA | 49K | None |
| `rlhf` | GenRM (Kimi K2.5) | HelpSteer3 | 16K | None |
| `swe_rl` | LLM judge / Execution | Cascade-RL-SWE / R2E-Gym | 98K | Modal (execution mode) |
| `swe_agentic` | Execution (Docker) | R2E-Gym-Subset | 262K | Modal |

### IF-RL (Instruction Following)

- **Dataset**: `nvidia/Nemotron-Cascade-2-RL-data`, IF-RL split (auto-downloaded from HuggingFace)
- **Reward**: Fraction of satisfied IFEval instructions (48 instruction types). Programmatic verification, no external model calls.
- **Dependencies**: `langdetect` (optional, for `language:response_language` checks; falls back to True if missing)
- **Config**: `group_size=16, groups_per_batch=32` (paper uses batch=128). `temperature=1.0, top_p=1.0, kl_coeff=0`.
- **Super base reward**: 0.77-1.0 (very strong baseline)
- **Known issues**: Two verifier soft spots (`constrained_response`, `counting_composition`) always return True, affecting ~2-5% of instruction types.

### MCQA (Multiple Choice QA)

- **Dataset**: `nvidia/Nemotron-Cascade-2-RL-data`, multi-domain split (auto-downloaded)
- **Reward**: Exact match after answer extraction. Supports `\boxed{}`, `Option Selected: X`, `<final_answer>X</final_answer>`, `((X))`, bold/italic formats, and standalone letter. Partial credit (0.5) for correct-but-overlong responses.
- **Dependencies**: None
- **Config**: `max_tokens=8192` recommended (base model generates very long `<think>` chains at 49K). Optional `system_prompt` to encourage concise reasoning (set to `None` for paper-matched).
- **Super base reward**: 1.0 (too easy at small scale; mixed groups emerge at larger batch)
- **Known issues**: At small batch sizes, all groups may be all-correct (no GRPO signal).

### Structured Output

- **Dataset**: `nvidia/Nemotron-Cascade-2-RL-data`, multi-domain split (auto-downloaded)
- **Reward**: JSON extraction + `jsonschema.validate()` against the provided JSON Schema. Validates field types, required fields, nested constraints, enum/pattern.
- **Dependencies**: `pip install jsonschema`
- **Config**: `max_tokens=49152`
- **Super base reward**: 0.67-0.89 (good GRPO variance)
- **Known issues**: None significant.

### Workbench (Tool-Calling)

- **Dataset**: `nvidia/Nemotron-Cascade-2-RL-data`, workbench split (4,686 examples across calendar, email, analytics, project management, CRM)
- **Reward**: Partial credit based on tool name match and argument match. `reward = 0.5 * name_match_rate + 0.5 * exact_match_rate`. Ground-truth IDs are seeded into mock tool backends so lookup calls return expected values.
- **Dependencies**: None
- **Config**: `max_turns=3` (info-gathering + action call), `max_tokens=49152`
- **Super base reward**: 0.44 (name_match ~87%, exact_match ~37%)
- **Known issues**: Strict argument matching (no fuzzy date/name matching). Chained multi-call tasks may struggle.

### Code RL

- **Dataset**: MBPP sanitized from HuggingFace (257 problems). Paper uses unreleased AtCoder/Codeforces (3.5K problems).
- **Reward**: Code execution in Modal sandbox. Each test assertion runs independently; supports strict binary or partial credit (`partial_credit=True` on `CodeRLDatasetBuilder`).
- **Dependencies**: Modal account + authentication (`modal token new`)
- **Config**: `max_tokens=118000` (paper-matched). `context_window=262144`.
- **Super base reward**: 0.75-1.0 (MBPP is relatively easy)
- **Known issues**: `test_imports` not included (affects 5% of MBPP). No public competitive programming dataset matching the paper's.

### Long-Context QA

- **Dataset**: NarrativeQA test split (auto-downloaded from HuggingFace)
- **Reward**: LLM judge (Qwen3.5-397B-A17B) scores response quality 0-10, normalized to [0,1]. Judge runs on the Tinker sampling API (no local GPU needed).
- **Dependencies**: None (judge model accessed via Tinker API)
- **Config**: `max_tokens=49152`, `judge_max_tokens=512` (must be high enough for thinking model reasoning)
- **Super base reward**: 0.60-0.65
- **Known issues**: Judge context truncated to 12K chars (may miss information in long documents). Uses document summaries when available (shorter than full text).

### RLHF

- **Dataset**: HelpSteer3 from HuggingFace (auto-downloaded). Uses `context` field (conversation format).
- **Reward**: Pairwise GenRM comparisons via Kimi K2.5 (paper uses Qwen3-235B). The GenRM judge runs on the Tinker sampling API. This is the only env with `kl_coeff=0.03`.
- **Dependencies**: None (GenRM accessed via Tinker API)
- **Config**: `max_tokens=16384`, `genrm_max_tokens=512` (may need increase to 2048+ for reliable verdicts)
- **Super base reward**: 0.50 (random baseline for pairwise comparison)
- **Known issues**: GenRM verdicts often unparseable when thinking consumes all tokens before the `VERDICT:` line.

### SWE Agentless

- **Dataset**: Two options:
  - `nvidia/Nemotron-Cascade-RL-SWE` (~110K instances, recommended) -- includes `relevant_file_contents` with codebase context. This is the paper's actual dataset.
  - `R2E-Gym/R2E-Gym-Subset` (4,578 instances) -- no codebase context, lower reward.
- **Reward**: Two modes:
  - `llm_judge` (default): Qwen3.5-397B judges patch quality. No sandbox needed.
  - `execution`: Modal sandbox clones repo, applies patch, runs tests. Uses R2E-Gym Docker images (`namanjain12/*` on Docker Hub) with pre-installed dependencies.
- **Dependencies**: Modal account (execution mode only). HuggingFace access for dataset download.
- **Config**: `max_tokens=98304`, `context_window=262144`. Set `reward_mode` in `SWERLDatasetBuilder`.
- **Super base reward**: 0.27-0.46 (LLM judge, Cascade data), 0.12-0.20 (LLM judge, R2E-Gym)
- **Known issues**: Execution mode has sandbox timeout issues on some repos. Failed `git apply` now exits immediately (was silently continuing).

### SWE Agentic

- **Dataset**: `R2E-Gym/R2E-Gym-Subset` (4,578 instances, 10 repos) from HuggingFace. Set `use_r2e_gym=True` (default).
- **Reward**: Execution-based. Modal sandbox runs model's code changes against FAIL_TO_PASS test suite. Binary (1 if tests pass, 0 otherwise).
- **Dependencies**: Modal account + authentication (`modal token new`). Docker Hub access for R2E-Gym images (`namanjain12/*` namespace). Images are ~300-500MB each, cached after first pull.
- **Config**: `max_tokens=262144`, `context_window=262144`, `max_turns=200`, `sandbox_timeout=600`. Paper uses `group_size=64, groups_per_batch=16, temperature=0.8`.
- **Setup**:
  1. Install Modal: `pip install modal && modal token new`
  2. The env auto-pulls R2E-Gym Docker images from Docker Hub via `modal.Image.from_registry()`
  3. Images include repo at `/testbed` with all dependencies pre-installed
- **Super base reward**: TBD (untested at scale on Super)
- **Known issues**:
  - Docker Hub rate limits: 100 anonymous pulls/6h. Authenticate or pre-pull for large group sizes.
  - ~19 min/step at small scale. Very expensive at paper's group_size=64.
  - Model likely needs SFT on coding tasks before getting non-zero reward.
  - `--timeout` flag removed from pytest commands (R2E-Gym images lack `pytest-timeout`); timeouts handled at sandbox level.

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
