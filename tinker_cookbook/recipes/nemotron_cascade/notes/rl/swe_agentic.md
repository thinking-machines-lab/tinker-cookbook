# SWE Agentic Environment

## Status: R2E-Gym Docker integration wired up, untested at scale

## Configuration
- max_tokens: 262K (paper-matched)
- max_turns: 200
- group_size: 64 (paper), temperature: 0.8
- Dataset: `R2E-Gym/R2E-Gym-Subset` (4,578 instances, 10 repos)
- Reward: binary (FAIL_TO_PASS tests pass = 1, else = 0)
- Tools: read_file, write_file, run_command in Modal sandbox

## R2E-Gym Integration (verified)
- `SWEAgenticDatasetBuilder` loads R2E-Gym-Subset when `use_r2e_gym=True` (default)
- `docker_image` from dataset -> `modal.Image.from_registry()` with workdir `/testbed`
- Test files from `execution_result_content` JSON -> written to `r2e_tests/` in sandbox
- Pre-built images have all dependencies installed (solves main pain point)

## Why It's Expensive
- Each rollout creates a Modal sandbox, with up to 200 turns of tool interaction
- 64 sandboxes per group at paper's group_size
- ~19 minutes per training step at small scale

## Known Limitations
- Docker Hub rate limits (100 anonymous pulls/6h) with group_size=64
- Untested at paper scale (batch=16, group=64)
- Context overflow at 200 turns possible (context_overflow_reward=0.0)
- Super base model likely needs SFT on coding tasks before getting non-zero reward
