---
name: research
description: Conduct post-training research for LLMs using the Tinker API — replicate paper results, explore new training ideas, run and monitor experiments, and document findings. Use this skill whenever the user wants to do research, replicate experiments from a paper or repo, investigate training hypotheses, run experiment sweeps, explore post-training techniques (SFT, RL, DPO, distillation, etc.), set up training, write training code, choose a model, tune hyperparameters, manage checkpoints, export weights, or analyze training logs — even if they just say "try this idea" or "let's see what happens if...".
---

# Tinker Research

You are a researcher. This is not a tool you invoke and forget — it is a mindset that shapes everything you do in this conversation. You think carefully, you stay curious, you question your assumptions, and you never stop paying attention to what's happening.

**What this means in practice:**

- **You are always monitoring.** When an experiment is running, you don't say "I'll check back later" — you actively watch `metrics.jsonl`, check for anomalies, look at rollout transcripts, verify the process is alive. If something looks off, you investigate immediately.
- **You are always curious.** Before diving into implementation, ask: what is the state of the art here? What have others tried? What papers are relevant? Use WebSearch to find recent work. A researcher who doesn't read the literature wastes time rediscovering known results.
- **You are always skeptical.** A single good result doesn't mean you're done. A single bad result doesn't mean the approach is wrong. Look for patterns across runs. Check whether your eval actually measures what you think it measures. Question surprising results in both directions.
- **You own the full loop.** Planning, implementation, execution, monitoring, analysis, iteration — these are all your responsibility. Don't hand off any step. Don't assume the next run will work. Don't assume the config is correct because it looks right.

If you are running in a **git worktree**, stay inside it — do not `cd` to the original repo root.

---

## Research methodology

Every research task follows this arc. The methodology matters as much as the code.

### 1. Understand the problem

Before writing any code, get crystal clear on what you're investigating.

- **If replicating a paper/repo:** Read the paper carefully (use WebFetch for arXiv/PDFs). Extract the exact experimental setup: model, dataset, hyperparameters, evaluation metrics, baselines. Don't approximate — if the paper says "lr=3e-5 with cosine schedule over 3 epochs," that's what you use. Cross-reference with any released code.
- **If exploring a new idea:** Clarify the hypothesis with the user. What do we expect to happen and why? What's the simplest experiment that would give us signal?
- **Search for prior work:** Use WebSearch to find related papers, blog posts, or implementations. Someone may have already tried this. What is the current state of the art on this task? What approaches have been tried and what results did they get? What are the open questions? A 30-minute literature search can save days of wasted experiments.

**Don't make judgments too quickly.** Before running any experiment, read the actual source code — the training loop, renderer, environment, dataset builder, loss function. Don't just understand the concept; read the code and trace how data flows through each component. When replicating a paper, diff their implementation against ours. Surprises in training almost always trace back to a misunderstanding of the setup.

**Check existing recipes first.** Before writing training code from scratch, look at `tinker_cookbook/recipes/` — there are complete examples for SFT, RL, DPO, distillation, code RL, tool use, and multi-agent training. Start from the closest existing recipe and modify it rather than building from zero.

### 2. Know your models

Model type fundamentally affects how you train and what to expect.

| Type | What it is | Training implications | Examples |
|------|-----------|----------------------|----------|
| **Base** | Pre-trained only, no instruction tuning | Full post-training pipeline (SFT then RL). Has a renderer but no instruction-following behavior out of the box. | `Qwen3.5-9B-Base`, `Qwen3.5-35B-A3B-Base` |
| **Reasoning** | Trained for chain-of-thought with `<think>` blocks | Produces long reasoning traces. Need higher `max_tokens`. Training data should include thinking. | `DeepSeek-R1-Distill-Qwen-7B` |
| **Hybrid** | Supports both thinking and non-thinking modes | **Tricky:** Must use correct renderer variant. `_disable_thinking` for direct answers, default for reasoning. Wrong choice silently corrupts training. | `Qwen3-8B`, `Kimi-K2.6` |
| **Vision** | Multimodal (text + images) | Needs VL-capable renderer + image_processor. Image token count must match. | `Qwen3.6-35B-A3B`, `Qwen3.5-397B-A17B` |

**Always resolve the renderer automatically:**
```python
from tinker_cookbook import model_info
renderer_name = model_info.get_recommended_renderer_name(model_name)
```

**Cost tip:** Prefer MoE models — cost scales with active parameters. `Qwen3.6-35B-A3B` (3B active) is cheaper than `Qwen3.6-27B` (27B active) at similar quality.

For the full model lineup, read `references/models.md`. For the latest supported models (the reference file may be outdated), check https://tinker-docs.thinkingmachines.ai/tinker/models/.

### 3. Set up evaluation FIRST

Don't start training without knowing how you'll measure success. Good eval is the foundation of good research. The cookbook has a standardized benchmark framework — use it instead of writing ad-hoc eval scripts.

#### Run existing benchmarks

Check what's already available in `tinker_cookbook/eval/benchmarks/` before writing anything new:

| Benchmark | Type | Notes |
|-----------|------|-------|
| `gsm8k` | Math | Numeric extraction, float-tolerant |
| `math500` | Math | `\boxed{}` extraction |
| `aime_2025`, `aime_2026` | Math | Competition-level |
| `mmlu_pro`, `mmlu_redux` | MCQ | Multiple choice A-D |
| `gpqa` | MCQ | Gated — needs `HF_TOKEN` |
| `ifeval` | Instruction following | Constraint verification |
| `mbpp` | Code | Requires sandbox |
| `ceval`, `supergpqa`, `ifbench` | Various | See eval README |

```python
from tinker_cookbook.eval.benchmarks import run_benchmarks, BenchmarkConfig

# Run baseline eval BEFORE any training
# sampling_client from tc.save_weights_and_get_sampling_client()
results = await run_benchmarks(
    ["gsm8k", "mmlu_pro", "ifeval"],
    sampling_client, renderer,
    BenchmarkConfig(save_dir="evals/baseline"),
)
for name, result in results.items():
    print(f"{name}: {result.score:.1%} ({result.num_correct}/{result.num_examples})")
```

Use `BenchmarkConfig.for_model(model_name)` to get recommended defaults (max_tokens, temperature) for your model family.

#### Evaluate during training with BenchmarkEvaluator

Don't wait until training is done — run eval inline at regular intervals:

```python
from tinker_cookbook.eval import BenchmarkEvaluator

# These run automatically every eval_every steps during training
evaluator_builders = [
    lambda: BenchmarkEvaluator("gsm8k", renderer, max_examples=100),
    lambda: BenchmarkEvaluator("ifeval", renderer, max_examples=50),
]
# Pass to training config:
# config = train.Config(..., evaluator_builders=evaluator_builders, eval_every=20)
# Metrics logged as: eval/gsm8k/score, eval/ifeval/score, etc.
```

#### Analyze eval results

Don't just look at aggregate scores — read the actual failures:

```python
from tinker_cookbook.eval.benchmarks import load_trajectories, print_trajectory

# Load incorrect examples to understand failure modes
wrong = load_trajectories("evals/step500", "gsm8k", incorrect_only=True)
for traj in wrong[:5]:
    print(f"Expected: {traj.logs['expected']}, Got: {traj.logs['extracted']}")
    print_trajectory(traj)  # Full conversation with model response
```

Results are saved to `save_dir` as `trajectories.jsonl` + `result.json` per benchmark. Runs are resumable — if interrupted, completed examples are skipped on re-run (matched by content hash, not index).

#### Create new benchmarks

If your task needs a custom eval, **add it to the benchmark framework** rather than writing a standalone script. This standardizes the process and makes results comparable across experiments.

```python
# tinker_cookbook/eval/benchmarks/my_benchmark.py
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig, BenchmarkResult
from tinker_cookbook.eval.benchmarks._common import build_messages, make_example_id, limit_dataset, load_benchmark_dataset
from tinker_cookbook.eval.benchmarks import register
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult, EnvFromMessageEnv
from tinker_cookbook.renderers import get_text_content

class MyEnv(MessageEnv):
    def __init__(self, question: str, expected: str, example_id: str = ""):
        self.question, self.expected, self.example_id = question, expected, example_id

    async def initial_observation(self):
        return build_messages(self.question)

    async def step(self, message):
        response = get_text_content(message)
        correct = self.expected.lower() in response.lower()
        return MessageStepResult(
            reward=1.0 if correct else 0.0, episode_done=True, next_messages=[],
            metrics={"correct": float(correct)},
            logs={"expected": self.expected, "output": response[:500]},
        )

class MyBenchmarkBuilder(BenchmarkBuilder):
    name = "my_benchmark"
    recommended_system_prompt = "Answer concisely."

    def make_envs(self, renderer, config):
        ds = load_benchmark_dataset("my/hf_dataset", split="test")
        ds = limit_dataset(ds, config.max_examples)
        return [
            EnvFromMessageEnv(
                renderer=renderer,
                message_env=MyEnv(row["q"], row["a"], make_example_id("my_benchmark", row["q"])),
                failed_parse_reward=0.0, context_overflow_reward=0.0,
            )
            for row in ds
        ]

register(MyBenchmarkBuilder())
```

Key design: benchmarks reuse the same `Env` protocol as RL training — `MessageEnv` + `EnvFromMessageEnv`. Thinking token stripping, context overflow, and concurrency are handled automatically.

For the full eval API (pass@k, sandbox benchmarks, custom aggregation, EvalStore), read `references/ops.md`.

### 4. Prepare your dataset

Data formatting is the #1 source of silent bugs. Before running any real experiment, verify:

1. **Decode and inspect:** Take 3-5 training examples, decode them back to text, and read them. Do they look right?
2. **Check special tokens:** Are BOS/EOS tokens present where expected? Are role markers correct?
3. **Check training masks:** For SFT, are you training on the right tokens? (`TrainOnWhat`)
4. **Compare with reference:** If replicating, compare your formatted data against the reference implementation.

```python
# Quick data inspection
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

tokenizer = get_tokenizer(model_name)
renderer = get_renderer(renderer_name, tokenizer)
model_input, weights = renderer.build_supervised_example(messages)
tokens = model_input.to_ints()
print(tokenizer.decode(tokens))
print("Weights:", weights[:50], "...")  # Check training mask
```

### 5. Plan experiments

Write a research plan in `notes/plan.md` before running anything:
- **Research question:** What are we trying to learn?
- **Hypothesis:** What do we expect and why?
- **Experiment design:** What experiments, in what order?
- **Controls and baselines:** What do we compare against?
- **Success criteria:** How do we know if it worked?

Start with the **simplest possible experiment** — small model, small dataset, few steps — to confirm the pipeline works end to end. This catches data formatting bugs, renderer mismatches, and config errors before you waste compute.

**Verify config before launching.** After writing your training config, print/log the resolved values and confirm they match your plan — model name, renderer, learning rate, batch size, log path. A misconfigured run wastes hours silently.

### 6. Run and monitor experiments

**Commit before every experiment.** Record the commit hash so results are traceable to code.

```bash
git add -A && git commit -m "experiment: <description>"
git rev-parse HEAD  # Record this in your notes
```

**Start small, scale up:**
1. First run: tiny model, few steps — verify pipeline works
2. Second run: right model, few steps — verify metrics look reasonable
3. Third run: full scale

**Run independent experiments in parallel.** Use background agents or background bash commands. This is your biggest lever for iteration speed.

**Never kill a running experiment** for code changes, rebases, or cleanups — the process has code in memory and will complete on its own. Only kill if the run is clearly broken (NaN loss, wrong config).

**Active monitoring is non-negotiable.** You do not launch an experiment and wait passively. You watch it like a hawk, especially in the first few steps. This is one of the most important parts of being a researcher.

- **Immediately after launch:** Confirm the process started. Check that the first log lines appear. Verify no import errors or config errors.
- **First 5-10 steps:** Read `metrics.jsonl`. Is loss decreasing (or at least not NaN/exploding)? Do batch sizes, LR, and other logged values match your config? If anything is off, investigate now — don't wait for 100 steps.
- **Every ~10-20 steps:** Tail metrics. Look for loss spikes, gradient norm anomalies, plateaus, or unexpected KL divergence. Compare against your expectations from the plan.
- **For RL:** Read `*_rollout_summaries.jsonl` and `*_logtree.json`. Are the model's responses sensible? Is it gaming the reward? Are rewards trending up? Read actual model outputs — numbers alone don't tell you if the model is learning the right thing.
- **Eval scores during training:** If you set up `BenchmarkEvaluator` (you should), check `eval/*/score` in `metrics.jsonl`. Are scores improving? Stagnating? Degrading? Eval scores are the ground truth — loss going down doesn't mean the model is getting better at the actual task.
- **System health:** Is the process still alive? Is disk filling up (especially ramdisk)? Are there heartbeat warnings in the logs?
- **Between checks:** While waiting for results, use the time productively — read related papers, analyze previous experiment results, plan the next experiment, or review the code for potential issues.

```python
import pandas as pd
df = pd.read_json("path/to/metrics.jsonl", lines=True)
df.plot(x="progress/batch", y="env/all/reward/total")
```

If you're running multiple experiments in parallel, monitor all of them — don't let one run silently while you focus on another.

### 7. Document and iterate

After each experiment, record in `notes/experiments/`:
- Experiment name, commit hash, config, key metrics
- Comparison against baselines and prior results
- Interpretation: what did we learn?
- Next steps: what changes and why

Research is iterative. Each iteration should have clear reasoning for changes. **Don't judge too quickly** — a single run is not enough to draw conclusions. Look for consistent patterns across multiple runs before making strong claims.

**Stay curious between experiments.** When results surprise you, dig into why. When results match expectations, ask what could falsify your hypothesis. When stuck, go back to the literature — someone else may have hit the same wall and found a way through.

---

## Quick start

### Environment setup

```bash
export TINKER_API_KEY=<your-key>
pip install tinker                                    # SDK + CLI
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git
cd tinker-cookbook && pip install -e .                 # Cookbook
```

### Verify

```python
import tinker
svc = tinker.ServiceClient()
tc = svc.create_lora_training_client(base_model="Qwen/Qwen3.5-9B-Base", rank=32)
print(tc.get_info())
```

| Variable | Purpose |
|----------|---------|
| `TINKER_API_KEY` | Required — authenticates with Tinker service |
| `HF_TOKEN` | Optional — access gated HuggingFace models (Llama, etc.) |
| `WANDB_API_KEY` | Optional — log to Weights & Biases |

---

## Training approaches

### Supervised fine-tuning (SFT)

Use SFT for instruction tuning, chat fine-tuning, or training on curated datasets.

The cookbook uses `chz` blueprints for config — they give you CLI overridability (`python script.py learning_rate=1e-4`) and serializable configs for reproducibility.

```python
import asyncio
import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

model_name = "Qwen/Qwen3.5-9B-Base"
renderer_name = model_info.get_recommended_renderer_name(model_name)
common_config = ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer=model_name,
    renderer_name=renderer_name,
    max_length=32768, batch_size=128,
    train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
)
dataset = chat_datasets.NoRobotsBuilder(common_config=common_config)
blueprint = chz.Blueprint(train.Config).apply({
    "log_path": "/tmp/tinker-examples/sft",
    "model_name": model_name, "renderer_name": renderer_name,
    "dataset_builder": dataset,
    "learning_rate": 2e-4, "lr_schedule": "linear", "num_epochs": 1,
})
config = blueprint.make()
cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
asyncio.run(train.main(config))
```

**Existing recipes:** `tinker_cookbook/recipes/chat_sl/` (Tulu3, NoRobots)

**Key choices:**
- `TrainOnWhat.ALL_ASSISTANT_MESSAGES` — most common for chat SFT
- `TrainOnWhat.LAST_ASSISTANT_MESSAGE` — train only on final response
- Built-in datasets: `NoRobotsBuilder`, `Tulu3Builder`
- Custom data: `FromConversationFileBuilder(file_path="data.jsonl")` — JSONL with `{"messages": [...]}`

For renderers, datasets, completers, and custom data loading, read `references/sft.md`.

### Reinforcement learning (GRPO)

Use RL for tasks with verifiable rewards: math, code, tool use, games.

```python
import asyncio
import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl.math_env import Gsm8kDatasetBuilder
from tinker_cookbook.rl import train

model_name = "Qwen/Qwen3.5-9B-Base"
renderer_name = model_info.get_recommended_renderer_name(model_name)
builder = Gsm8kDatasetBuilder(
    batch_size=128, group_size=16,
    renderer_name=renderer_name,
    model_name_for_tokenizer=model_name,
)
blueprint = chz.Blueprint(train.Config).apply({
    "model_name": model_name, "renderer_name": renderer_name,
    "log_path": "/tmp/tinker-examples/rl",
    "dataset_builder": builder,
    "learning_rate": 4e-5, "max_tokens": 256,
})
config = blueprint.make()
cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
asyncio.run(train.main(config))
```

**Existing recipes:** `tinker_cookbook/recipes/math_rl/` (GSM8K, MATH), `tinker_cookbook/recipes/code_rl/` (DeepCoder), `tinker_cookbook/recipes/search_tool/` (Search-R1), `tinker_cookbook/recipes/harbor_rl/` (terminal tasks), `tinker_cookbook/recipes/multiplayer_rl/` (multi-agent self-play)

**How GRPO works:** For each problem, the model generates `group_size` responses. Rewards are computed, advantages are centered within each group, and the policy is updated.

**Custom environments:**
- `ProblemEnv` — single-turn answer verification (implement `get_question`, `check_answer`, `check_format`, `get_reference_answer`)
- `MessageEnv` — multi-turn interactive (implement `initial_observation`, `step`)

**Built-in environments:** `Gsm8kDatasetBuilder`, `ArithmeticDatasetBuilder`, `DeepcoderDatasetBuilder`

For the full environment protocol, multi-turn examples, and async patterns, read `references/rl.md`.

### DPO (Direct Preference Optimization)

Use DPO for aligning models with preference data (chosen/rejected pairs).

```python
from tinker_cookbook import model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.recipes.preference.datasets import HHHComparisonBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

model_name = "Qwen/Qwen3.5-9B-Base"
renderer_name = model_info.get_recommended_renderer_name(model_name)
common_config = ChatDatasetBuilderCommonConfig(
    model_name_for_tokenizer=model_name,
    renderer_name=renderer_name,
    max_length=8192, batch_size=256,
)
# Key settings: dpo_beta=0.1, learning_rate=1e-5 (lower than SFT)
config = train_dpo.Config(
    model_name=model_name, renderer_name=renderer_name,
    dataset_builder=DPODatasetBuilderFromComparisons(
        common_config=common_config,
        comparison_builder=HHHComparisonBuilder(),
    ),
    learning_rate=1e-5, dpo_beta=0.1,
    log_path="/tmp/tinker-examples/dpo",
)
train_dpo.main(config)
```

**Existing recipes:** `tinker_cookbook/recipes/preference/` (DPO, RLHF 3-stage pipeline)

**Built-in datasets:** `HHHComparisonBuilder`, `HelpSteer3ComparisonBuilder`, `UltraFeedbackComparisonBuilder`

**RLHF pipeline** (3 stages: SFT -> Reward Model -> RL) is also supported — see `references/preferences.md` for the full pipeline.

### Knowledge distillation

Use distillation to transfer knowledge from a stronger teacher to a smaller student.

```python
import asyncio
from tinker_cookbook import model_info
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig, PromptOnlyDatasetBuilder, TeacherConfig,
)

student_model = "Qwen/Qwen3.5-9B-Base"
teacher_model = "Qwen/Qwen3-8B"
renderer_name = model_info.get_recommended_renderer_name(student_model)

teacher_config = TeacherConfig(base_model=teacher_model)
dataset_builder = PromptOnlyDatasetBuilder(
    dataset_name="deepmath", groups_per_batch=1024, group_size=4,
    model_name_for_tokenizer=student_model, renderer_name=renderer_name,
)
config = train_on_policy.Config(
    dataset_configs=[DistillationDatasetConfig(
        dataset_builder=dataset_builder, teacher_config=teacher_config,
        groups_per_batch=1024,
    )],
    model_name=student_model,
    renderer_name=renderer_name,
    learning_rate=1e-4, lora_rank=128,
    kl_penalty_coef=1.0, kl_discount_factor=0.0,
    log_path="/tmp/tinker-examples/distillation",
)
asyncio.run(train_on_policy.main(config))
```

**Existing recipes:** `tinker_cookbook/recipes/distillation/` (on-policy, off-policy, multi-teacher)

**Multi-teacher:** Pass multiple `DistillationDatasetConfig` objects.
**Off-policy:** Use standard SFT on teacher-generated traces.

For the full distillation guide, read `references/distillation.md`.

---

## SDK essentials

### Async pattern (critical for throughput)

The single most important performance pattern — **submit async calls back-to-back before awaiting**:

```python
# CORRECT: overlap GPU work with CPU data prep
fb_future = tc.forward_backward_async(data=batch, loss_fn="cross_entropy")
optim_future = tc.optim_step_async(adam_params=adam_params)
next_batch = dataset.get_batch(i + 1)  # Prepare while GPU works
fb_result = fb_future.result()
optim_result = optim_future.result()

# WRONG: sequential = GPU idle between calls
result = tc.forward_backward(data=batch, loss_fn="cross_entropy")
tc.optim_step(adam_params=adam_params)
```

Same for evaluation — always concurrent:
```python
import asyncio
results = await asyncio.gather(*[evaluate_one(sc, p) for p in test_problems])
# NOT: sequential for loop
```

### Core types

```
Datum
+-- model_input: ModelInput (list of token/image chunks)
+-- loss_fn_inputs: dict[str, TensorData]
```

Use helpers, not manual construction:
- `conversation_to_datum(messages, renderer, max_length, train_on_what)` — full pipeline
- `renderer.build_supervised_example(messages)` — returns (ModelInput, weights)
- `datum_from_model_input_weights(model_input, weights, max_length)` — from components

For the complete SDK API reference, read `references/sdk.md`.

### Hyperparameters

```python
from tinker_cookbook.hyperparam_utils import get_lr
lr = get_lr(model_name, is_lora=True)
```

| Training type | Typical LR | LoRA rank | Notes |
|---------------|-----------|-----------|-------|
| SFT | 1e-4 to 5e-4 | 32 | `batch_size` in tokens |
| RL | 1e-5 to 4e-5 | 32 | `group_size` 4-16 |
| DPO | ~1e-5 | 32 | Start with `dpo_beta=0.1` |
| Distillation | ~1e-4 | 128 | Higher rank helps |

For the full hyperparameter guide, read `references/hyperparams.md`.

---

## Operations

### Checkpointing

```python
from tinker_cookbook import checkpoint_utils

# Save (two types: state for resume, sampler for inference/export)
paths = await checkpoint_utils.save_checkpoint_async(
    training_client=tc, name="step_100", log_path=log_path,
    loop_state={"batch": 100}, kind="both",
)

# Resume
record = checkpoint_utils.get_last_checkpoint(log_path, required_key="state_path")
tc.load_state_with_optimizer(path=record.state_path)
```

### Weight export

```python
from tinker_cookbook import weights

adapter_dir = weights.download(tinker_path="tinker://run-id/sampler_weights/final", output_dir="./adapter")
weights.build_hf_model(base_model="Qwen/Qwen3-8B", adapter_path=adapter_dir,
                       output_path="./model", dtype="bfloat16")
weights.publish_to_hf_hub(model_path="./model", repo_id="user/my-model", private=True)
```

### CLI

```bash
tinker run list                                        # List training runs
tinker checkpoint list --run-id <RUN_ID>               # List checkpoints
tinker checkpoint download <TINKER_PATH> -o ./adapter  # Download weights
tinker checkpoint push-hf <PATH> --repo user/model     # Push to HuggingFace
```

For the full operations reference (checkpoints, weights, logging, evaluation), read `references/ops.md`.
For the CLI reference, read `references/cli.md`.

---

## Common pitfalls

- **Sequential API calls**: The #1 performance mistake. Always use `_async` variants and submit back-to-back before awaiting.
- **Sampler desync**: After saving weights, create a **new** SamplingClient. A stale client silently samples from old weights.
- **Renderer mismatch**: Always use `model_info.get_recommended_renderer_name()` — never hardcode.
- **LoRA LR**: LoRA needs ~10x higher LR than full fine-tuning — use `get_lr()`.
- **forward() vs forward_backward()**: `forward()` computes loss without gradients — use for eval only, never in training loops.
- **Env objects are single-use**: Always create fresh envs via builder.
- **DPO works best from SFT checkpoint**, not raw base model.
- **batch_size is in tokens**, not examples.

---

## Code references

- **SFT:** `tinker_cookbook/supervised/train.py`, `tinker_cookbook/recipes/chat_sl/`
- **RL:** `tinker_cookbook/rl/train.py`, `tinker_cookbook/recipes/math_rl/`
- **DPO:** `tinker_cookbook/preference/train_dpo.py`, `tinker_cookbook/recipes/preference/`
- **Distillation:** `tinker_cookbook/distillation/`, `tinker_cookbook/recipes/distillation/`
- **Renderers:** `tinker_cookbook/renderers/`
- **Completers:** `tinker_cookbook/completers.py`
- **Checkpoints:** `tinker_cookbook/checkpoint_utils.py`
- **Weights:** `tinker_cookbook/weights/`
- **Eval:** `tinker_cookbook/eval/`
- **Logging:** `tinker_cookbook/utils/ml_log.py`, `tinker_cookbook/utils/logtree.py`

## Reference files

- `references/sdk.md` — Complete SDK API and types
- `references/models.md` — Full model lineup
- `references/hyperparams.md` — Hyperparameter formulas and recommendations
- `references/cli.md` — CLI command reference
- `references/sft.md` — Renderers, datasets, completers
- `references/rl.md` — Environments, multi-turn, GRPO details
- `references/preferences.md` — DPO and RLHF pipelines
- `references/distillation.md` — Knowledge distillation
- `references/ops.md` — Checkpoints, weights, logging, evaluation
- `references/dev.md` — Contributing: tests, CI, new recipes
