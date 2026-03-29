# R2E-Gym Integration Analysis

## What R2E-Gym Is

R2E-Gym (UC Berkeley / ANU, COLM 2025) is the largest procedurally curated gym environment
for training SWE agents. It provides:

- **8,100+ problems across 13 repos** with executable environments, unit tests, and
  natural-language task descriptions.
- **Pre-built Docker images** (one per commit) hosted on Docker Hub under the
  `namanjain12/` namespace, with all repo-specific dependencies pre-installed.
- A **gym-style Python interface** (`RepoEnv`) backed by a `DockerRuntime` that manages
  containers, command execution, file I/O, and reward calculation.
- **SFT trajectory datasets** on HuggingFace for bootstrapping agents before RL.

Paper: <https://arxiv.org/abs/2504.07164>
Repo: <https://github.com/R2E-Gym/R2E-Gym> (Apache-2.0, ~256 stars)

## How It Differs from SWE-bench

| Aspect | SWE-bench | R2E-Gym |
|---|---|---|
| Data source | Human-written PRs with issues | Synthetic from commits (SWE-GEN) |
| Scale | ~2,294 (full), 500 (verified) | 8,100+ |
| Docker images | `swebench/` namespace, per-instance | `namanjain12/` namespace, per-commit |
| Reward | External harness (`swebench.harness`) | Built-in `_calculate_reward()` via `DockerRuntime` |
| Intended use | Evaluation benchmark | Training gym (SFT + RL) |
| Gym interface | None (separate harness) | `gym.Env` with `step()`, `reset()`, `get_task_instruction()` |

Key: R2E-Gym was designed for **training** (large, diverse, gym-compatible), while SWE-bench
is primarily an **evaluation** benchmark. R2E-Gym's Docker images bake in the correct
Python version and all project dependencies, which is the main pain point in our current env.

## Connection to Nemotron-Cascade-2

The Nemotron-Cascade-2 paper's SWE Agentic RL stage uses data from both SWE-Gym and
"R2E-Subset". NVIDIA's `nvidia/Nemotron-Cascade-2-RL-data` (SWE-RL split) confirms this:
the `dataset_name` field in instances references `R2E-Gym/R2E-Gym-Subset`. So R2E-Gym is
already part of the paper's training data -- we just need to use its Docker environments
properly.

## Available Data on HuggingFace

**Datasets:**
- `R2E-Gym/R2E-Gym-Lite` -- curated lite subset (streaming-friendly)
- `R2E-Gym/R2E-Gym-Subset` -- the subset used in Nemotron-Cascade-2
- `R2E-Gym/R2E-Gym-Full` (alias `R2E-Gym/R2E-Gym-V1`) -- all 8.1K instances
- `R2E-Gym/SWE-Bench-Verified` -- SWE-bench Verified wrapped with R2E-Gym Docker images
- `R2E-Gym/SWE-Bench-Lite` -- SWE-bench Lite wrapped similarly
- `R2E-Gym/R2EGym-SFT-Trajectories` -- Claude-3.5-Sonnet SFT trajectories for editing agent
- `R2E-Gym/R2EGym-TestingAgent-SFT-Trajectories` -- SFT data for testing agent
- `R2E-Gym/R2EGym-Verifier-Trajectories` -- SFT data for verifier

**Models (fine-tuned Qwen2.5-Coder):**
- `R2E-Gym/R2EGym-32B-Agent`, `R2E-Gym/R2EGym-14B-Agent`, `R2E-Gym/R2EGym-7B-Agent`
- `R2E-Gym/R2EGym-Verifier`, `R2E-Gym/R2E-TestgenAgent`
- `agentica-org/DeepSWE-Preview` -- latest SOTA model (RL-trained via rLLM)

**No pip package.** Must clone the repo and `pip install -e .` or just use the datasets
and Docker images directly.

## Dataset Schema (R2E-Gym-Lite)

Each row contains:
- `repo_name`: e.g. `"aiohttp"`
- `docker_image`: e.g. `"namanjain12/aiohttp_final:f0d74880..."` -- pull-ready Docker Hub image
- `commit_hash`: the target commit
- `parsed_commit_content`: JSON with file diffs (ground truth patch)
- `problem_statement`: natural-language issue description
- `expected_output_json`: expected test results for grading
- `modified_files`, `relevant_files`, `modified_entity_summaries`: metadata
- `prompt`: system prompt for the agent
- `execution_result_content`: JSON with test file codes and execution logs

## How We Could Integrate

### Option A: Use R2E-Gym Docker images in our Modal sandbox (recommended)

Replace the current approach in `swe_agentic_env.py`:

```python
# CURRENT: generic Debian image, clone repo, pip install -e . (fails)
image = modal.Image.debian_slim().apt_install("git").pip_install("pytest", ...)

# PROPOSED: pull R2E-Gym's pre-built Docker image
image = modal.Image.from_registry(row["docker_image"])
```

Modal supports `Image.from_registry()` for arbitrary Docker Hub images. This would:
1. Eliminate the dependency installation problem (P0 from swe_agentic.md)
2. Eliminate the shallow-clone / checkout failures
3. Provide a `/testbed` working directory with the repo already set up
4. Include the correct Python version and test framework

Changes needed:
- `SWEAgenticEnvGroupBuilder.__init__` takes `docker_image` instead of `repo`/`base_commit`
- `make_envs()` uses `modal.Image.from_registry(docker_image)` instead of building from scratch
- `SWEAgenticReward` uses R2E-Gym's test execution approach (or keeps our pytest runner --
  the deps will be available either way)
- Workdir changes from `/workspace/repo` to `/testbed`

### Option B: Use R2E-Gym's DockerRuntime directly (more work, less Modal)

R2E-Gym's `DockerRuntime` manages Docker containers natively (or via Kubernetes). We could:
1. Run Docker containers on the training host (or a Docker-enabled VM)
2. Wrap `DockerRuntime` in our `SandboxInterface`
3. Skip Modal entirely for SWE tasks

This is more efficient (no Modal overhead) but requires Docker access on the training
machine, which may conflict with cloud/cluster setups.

### Option C: Use R2E-Gym's HF dataset + our own Docker orchestration

Load `R2E-Gym/R2E-Gym-Subset` from HuggingFace, extract the `docker_image` field, and
`docker pull` each image. Then use any container runtime (Modal, Docker, Kubernetes).
This is the most flexible approach and doesn't require installing R2E-Gym's codebase.

## Feasibility Assessment

**High feasibility for Option A.** The main integration points are:

1. **Data loading**: Already done -- `nvidia/Nemotron-Cascade-2-RL-data` SWE-RL split
   references R2E-Gym instances and includes all needed fields. Alternatively, load
   `R2E-Gym/R2E-Gym-Subset` directly for the `docker_image` field (which the NVIDIA
   dataset may not include -- needs verification).

2. **Docker image availability**: Images are on Docker Hub under `namanjain12/`. Each is
   300-500MB. Modal can pull these. For 64 concurrent sandboxes (paper's group_size),
   this means ~20-30GB of image pulls per unique repo, but images cache after first pull.

3. **Reward calculation**: R2E-Gym uses `swebench.harness` for grading on SWE-bench
   instances and its own test runner for synthetic instances. We can reuse our existing
   `SWEAgenticReward` class since the deps will actually be available.

4. **Risks**:
   - Docker Hub rate limits (anonymous: 100 pulls/6h, authenticated: 200 pulls/6h).
     With group_size=64, we'd hit this fast. Mitigation: pre-pull images, use a registry
     mirror, or authenticate.
   - Image staleness: `namanjain12/` images are research artifacts, not maintained infra.
     They could disappear. Mitigation: mirror to our own registry.
   - Modal `from_registry()` may add latency on first pull. Subsequent runs use cache.

5. **Effort estimate**: ~1 day to modify `swe_agentic_env.py` to use R2E-Gym Docker
   images via Option A. The dataset builder needs a join between NVIDIA's RL data
   (which has instance_id, problem_statement, etc.) and R2E-Gym's dataset (which has
   docker_image). Or we load R2E-Gym's dataset directly and skip NVIDIA's.

## Next Steps

1. Verify whether `nvidia/Nemotron-Cascade-2-RL-data` includes `docker_image` or if we
   need to join with `R2E-Gym/R2E-Gym-Subset`.
2. Test `modal.Image.from_registry("namanjain12/aiohttp_final:...")` to confirm Modal
   can pull R2E-Gym images and run commands in them.
3. Confirm the working directory is `/testbed` and that `pytest` works out of the box.
4. Update `SWEAgenticEnvGroupBuilder` to use the R2E-Gym Docker images.
5. Start with a small group_size (4-8) to validate reward > 0 before scaling up.
