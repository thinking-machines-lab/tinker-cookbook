# True-Thinking Score (TTS)

Replicates and validates the **True-Thinking Score** metric from
["Can Aha Moments Be Fake? Identifying True and Decorative Thinking Steps in Chain-of-Thought"](https://arxiv.org/abs/2510.24941)
(Zhao et al., 2025).

TTS measures the **causal contribution** of each reasoning step in
chain-of-thought (CoT) to the model's final prediction. The paper's key
finding is that most CoT steps are *decorative* — they look like reasoning
but barely influence the answer. Only ~2% of steps are truly causal.

This recipe implements TTS computation using the Tinker API and validates
the finding across 3 models (Qwen3.5-4B, Qwen3.5-27B, DeepSeek-V3.1)
on MATH-500 problems.

---

### What the paper reports

The authors test three distilled reasoning models (DeepSeek-R1-Distill-Qwen-7B,
DeepSeek-R1-Distill-Llama-8B, Nemotron-1.5B) on AMC, AIME, MATH, and
CommonsenseQA. Key findings on AIME with DeepSeek-R1-Distill-Qwen-7B:

- Mean TTS ~0.03 — most steps contribute almost nothing
- Only **2.3%** of steps have TTS >= 0.7 (truly causal)
- Only **6.4%** of steps have TTS >= 0.3
- **12%** of self-verification steps in Qwen-7B (21% in Nemotron) have
  TTS < 0.005 — "aha moments" that are purely decorative

The paper also extracts "TrueThinking" steering vectors from internal
activations (Section 6), achieving ~55% prediction flip rates on AMC/AIME.
This requires residual-stream access which Tinker does not expose, so we
focus on TTS computation only.

### How TTS works

For each reasoning step $s_i$ in a chain-of-thought, we run **4 forward
passes** that measure the model's confidence in the correct answer under
different perturbation conditions:

| | Intact step ($x$=1) | Perturbed step ($x$=0) |
|---|---|---|
| **Intact context** ($c$=1) | $S_1(1)$: original steps 1..i-1 + original step i | $S_0(1)$: original steps 1..i-1 + perturbed step i |
| **Perturbed context** ($c$=0) | $S_1(0)$: perturbed steps 1..i-1 + original step i | $S_0(0)$: perturbed steps 1..i-1 + perturbed step i |

Each cell measures $S_x(c) = P(y^* \mid \text{context}=c,\;\text{step}=x)$
— the probability of the correct answer given that CoT prefix — via
`compute_logprobs_async`.

TTS is then the average of the two row-wise diffs:

$$\text{TTS}(s) = \tfrac{1}{2}\bigl(|S_1(1) - S_0(1)| + |S_1(0) - S_0(0)|\bigr)$$

- **Row 1** $|S_1(1) - S_0(1)|$: hold context intact, toggle the step.
  Measures **necessity** — does the model rely on this step?
- **Row 2** $|S_1(0) - S_0(0)|$: hold context perturbed, toggle the step.
  Measures **sufficiency** — can this step alone drive the correct answer?

Testing under both contexts matters because a single diff can miss
"OR-type" steps: two steps that independently lead to the answer. With
intact context, each looks unimportant (the other still works). With
perturbed context, each is revealed as sufficient on its own.

A **decorative step** has TTS $\approx$ 0: toggling it makes no difference.
A **true-thinking step** has high TTS: the model's prediction meaningfully
changes when you perturb it.

### What we implement in Tinker

We replicate TTS computation using Tinker's `compute_logprobs_async` API:

1. **Generate CoT**: Sample from a thinking model (greedy, temperature=0)
   using the renderer's chat template. The model produces `<think>...</think>`
   blocks with extended reasoning.

2. **Segment steps**: Split the thinking text using discourse markers
   (numbered lists, transition words like "So", "Wait", "Therefore", etc.).

3. **Perturb steps**: For numeric steps, add small integer offsets from
   {-3,-2,-1,1,2,3} to numbers (matching Appendix A). For example, a
   real step from Qwen3.5-4B's CoT on an inclusion-exclusion problem:
   ```
   Original:  (33 + 19 + 14) - (6 + 4 + 2) + 0
   Perturbed: (30 + 16 + 17) - (5 + 2 + 0) + -2
   ```
   For non-numeric steps, drop them entirely.

4. **Early-exit confidence**: For each of the four conditions, build
   a sequence `[prompt + <think> CoT_prefix </think> \boxed{answer}]`
   using the renderer's chat template, then measure
   $P(\text{answer tokens} \mid \text{prefix})$ via `compute_logprobs_async`.

5. **Compute TTS** from the four confidence measurements.

**Approximations vs. the paper:**

- **Models:** The paper uses DeepSeek-R1-Distill (7B, 8B) and Nemotron-1.5B.
  We use Qwen3.5-4B, Qwen3.5-27B, and DeepSeek-V3.1 (671B-A37B).
  All produce `<think>...</think>` delimited CoT. Our DeepSeek-V3.1 is a
  much larger non-distilled model than the paper's distilled 7B variant.
- **Dataset:** The paper tests on AMC, AIME, MATH, and CommonsenseQA. We
  test on MATH-500 (a held-out subset of MATH).
- **Early-exit cue:** The paper appends `"The final result is"` **inside**
  the reasoning block, probing "what would you predict mid-thought?" We
  close the `</think>` block and use `\boxed{}` format — probing "if you
  stopped thinking here, what would your final answer be?" Both measure how
  the model's answer-prediction **changes** when a step is perturbed (i.e.
  TTS is relative), so the choice of cue mainly shifts the baseline
  probability, not the TTS scores.
- **Confidence measurement:** The paper uses "model's confidence Pr(y*)"
  via early-exit prompting but does not fully specify the computation.
  We compute `exp(sum(logprobs))` over the answer tokens, giving the joint
  probability P(answer_tokens | prefix). Since TTS measures *relative
  changes* in confidence, the exact metric should not significantly affect
  the TTS scores.
- **Step segmentation:** The paper treats **sentences** as steps (Appendix A).
  We use discourse markers (numbered lists, transition words). Both are
  heuristic and produce comparable step counts.
- **Perturbation (matches Appendix A):** We add integer offsets from
  {-3,-2,-1,1,2,3} to numbers and **drop non-numeric steps entirely**,
  matching the paper. Context perturbation only changes numbers.
- **No steering vectors:** The paper's Section 6 extracts "TrueThinking"
  steering directions from internal activations to control step reliance,
  achieving ~55% prediction flip rates vs <30% for random vectors. Tinker
  does not expose residual-stream activations, so this part is not replicated.

---

### Setup

No special data download is needed — MATH-500 and GSM8K are loaded
automatically from HuggingFace. You only need a Tinker API key:

```bash
export TINKER_API_KEY=<your-key>
```

### Running the recipe

**50 MATH-500 problems (~14 minutes with concurrency=64):**

```bash
python -m tinker_cookbook.recipes.true_thinking_score.analyze \
    dataset=math n_problems=50
```

**Quick smoke test (5 problems, ~3 minutes):**

```bash
python -m tinker_cookbook.recipes.true_thinking_score.analyze \
    n_problems=5
```

**DeepSeek-V3.1 (requires thinking renderer override):**

```bash
python -m tinker_cookbook.recipes.true_thinking_score.analyze \
    model_name=deepseek-ai/DeepSeek-V3.1 renderer_name=deepseekv3_thinking \
    n_problems=50
```

**GSM8K, larger model:**

```bash
python -m tinker_cookbook.recipes.true_thinking_score.analyze \
    dataset=gsm8k model_name=Qwen/Qwen3.6-27B n_problems=50
```

Results are saved to `/tmp/tinker-examples/tts/<run-name>/`:
- `tts_per_problem.jsonl` — per-problem details (steps, TTS scores)
- `tts_summary.json` — aggregate statistics

**Using TTS programmatically:**

```python
import asyncio
import tinker
from tinker_cookbook.recipes.true_thinking_score.tts import generate_cot_and_compute_tts

async def main():
    service_client = tinker.ServiceClient()
    result = await generate_cot_and_compute_tts(
        service_client=service_client,
        model_name="Qwen/Qwen3.5-4B",
        question="How many positive integers less than 100 are divisible by 3, 5, or 7?",
        answer_str="54",
        max_tokens=4096,
    )
    print(result.summary())
    for step in result.step_scores:
        tag = " [DECORATIVE]" if step.tts <= 0.005 else ""
        tag = " [TRUE-THINKING]" if step.tts >= 0.7 else tag
        print(f"  Step {step.step_index}: TTS={step.tts:.4f}{tag}")

asyncio.run(main())
```

**Unit tests (no API key needed):**

```bash
pytest tinker_cookbook/recipes/true_thinking_score/tts_test.py -v
```

---

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `Qwen/Qwen3.5-4B` | Thinking model to analyze |
| `renderer_name` | `None` (auto) | Override renderer (e.g. `deepseekv3_thinking` for DeepSeek) |
| `dataset` | `math` | Dataset: `math` (MATH-500) or `gsm8k` |
| `n_problems` | `50` | Number of problems to analyze |
| `concurrency` | `64` | Max parallel problems (steps within a problem are sequential) |
| `max_tokens` | `4096` | Max tokens for CoT generation |
| `seed` | `42` | Random seed for perturbation reproducibility |

---

### Results

**50 MATH-500 problems per model, concurrency=64:**

| Metric | Paper (R1-Distill-7B) | Qwen3.5-4B | Qwen3.5-27B | DeepSeek-V3.1 (671B) |
|---|---|---|---|---|
| Steps/problem | — | 31.7 | 31.9 | **11.3** |
| Mean TTS | ~0.03 | 0.054 | 0.070 | **0.144** |
| TTS >= 0.7 | 2.3% | 2.0% | 1.8% | **5.0%** |
| TTS >= 0.3 | 6.4% | 5.7% | 8.0% | **19.1%** |
| Decorative (<=0.005) | — | 59.3% | 51.1% | **35.1%** |
| SV steps | — | 115 | 110 | 113 |
| SV decorative | 12-21% | 56.5% | 49.1% | **36.3%** |
| Accuracy | — | 58% | 62% | **70%** |

> **Note:** These historical results include `Qwen3.5-27B`, which is deprecated. Rerun that column with `Qwen/Qwen3.6-27B` before treating the comparison as current.

### Findings

1. **The paper's core claim is validated across 3 models:** The ~2% high-TTS
   finding is consistent across Qwen3.5-4B (2.0%) and Qwen3.5-27B (1.8%)
   — closely matching the paper's 2.3%. DeepSeek-V3.1 is higher at 5.0%,
   likely because its concise reasoning style (11 steps vs 32) packs more
   causal content per step.

2. **Scaling reduces decorative reasoning:** DeepSeek-V3.1 (671B) has far
   fewer decorative steps (35.1%) than Qwen models (51-59%). Among Qwen
   models, the larger 27B also has fewer decorative steps (51.1%) than 4B
   (59.3%).

3. **Larger models are more concise:** DeepSeek-V3.1 solves problems in
   11.3 steps on average vs 32 for Qwen models. Each step carries more
   causal weight — the model "wastes" fewer steps exploring dead ends.

4. **Self-verification is often fake:** 36-57% of "Wait, let me re-check"
   steps are decorative across all models. DeepSeek-V3.1 is the most honest
   at 36%. This is higher than the paper's 12-21%, likely because of **step
   granularity**: the paper segments by sentences, bundling self-verification
   cues ("Wait") with the recomputation that follows into one step. Our
   discourse-marker segmentation splits these apart, creating short SV
   fragments (e.g. just `"Wait, let me re-check."`) that are individually
   decorative even if the surrounding recomputation is not.

5. **TTS rises near the answer:** The final steps before the answer
   consistently have the highest TTS, suggesting the model "commits" to an
   answer path late in the chain. Early reasoning steps explore without
   making progress.

6. **Wrong answers have lower TTS:** Problems where the model gets the wrong
   answer tend to have near-zero TTS across all steps — the model never
   locks onto a viable solution path.

---

### Files

| File | Description |
|---|---|
| `tts.py` | Core TTS computation: step segmentation, number perturbation, early-exit confidence, TTS scoring |
| `analyze.py` | CLI entry point for running TTS analysis on MATH-500 or GSM8K |
| `run_small_experiment.py` | Quick validation script (3 hardcoded problems) |
| `tts_test.py` | 18 unit tests for segmentation, perturbation, and self-verification detection |

### References

- [Can Aha Moments Be Fake?](https://arxiv.org/abs/2510.24941) — Zhao et al., 2025
- [Tinker API docs](https://tinker-docs.thinkingmachines.ai/) — `compute_logprobs_async` reference
