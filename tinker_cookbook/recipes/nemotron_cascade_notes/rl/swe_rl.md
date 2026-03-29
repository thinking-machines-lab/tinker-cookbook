# SWE-RL (Agentless) Environment

## Status: Working with LLM judge (reward 0.12-0.46)

## Configuration
- max_tokens: 98K (paper-matched)
- Reward modes:
  - `llm_judge` (default): Qwen3.5-397B judges patch quality 0-10, normalized to [0,1]
  - `execution`: Modal sandbox applies patch, runs FAIL_TO_PASS tests (binary)
- Datasets:
  - `nvidia/Nemotron-Cascade-RL-SWE` (~110K instances with codebase context) -- recommended
  - `R2E-Gym/R2E-Gym-Subset` (4,578 instances, no codebase context)

## Super Base Model Results (LLM judge)
- Cascade SWE dataset: reward 0.27-0.46, frac_mixed=1.0
- R2E-Gym-Subset: reward 0.12-0.20, frac_mixed=1.0
- Cascade data gives 25% higher reward due to `relevant_file_contents` context

## Key Finding: Use Cascade SWE Dataset
`nvidia/Nemotron-Cascade-RL-SWE` is the paper's actual dataset. It includes `relevant_file_contents` (golden + retrieved localizations), which is essential context the model needs. The R2E-Gym-Subset alone lacks this.

## Execution Mode
- R2E-Gym Docker images via Modal (`modal.Image.from_registry(docker_image)`)
- Two bugs fixed: removed `--timeout=60` (broke pytest), changed failed `git apply` to exit immediately
- Still has sandbox timeout issues on some repos (e.g., pandas)

## Known Limitations
- Execution mode can overflow context (24K prompt + 49K max_tokens > context window at smaller models)
- Patch extraction is fragile (handles fenced blocks but not all formats)
- `git apply` is strict about context lines; model-hallucinated patches often fail
