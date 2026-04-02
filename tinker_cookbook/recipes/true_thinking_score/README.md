# True-Thinking Score (TTS)

Replicates and validates the **True-Thinking Score** metric from
["Can Aha Moments Be Fake? Identifying True and Decorative Thinking Steps in Chain-of-Thought"](https://arxiv.org/abs/2510.24941)
(Zhao et al., 2025).

TTS measures the **causal contribution** of each reasoning step in
chain-of-thought (CoT) to the model's final prediction. The paper's key
finding is that most CoT steps are *decorative* — they look like reasoning
but barely influence the answer. Only ~2% of steps are truly causal.

This recipe implements TTS computation using the Tinker API and validates
the finding on Qwen3.5-4B with competition-math problems.

---

### What the paper reports

The authors test three reasoning models (DeepSeek-R1-Distill-Qwen-7B,
DeepSeek-R1-Distill-Llama-8B, Nemotron-1.5B) on AMC, AIME, MATH, and
CommonsenseQA:

| Metric | AIME (Qwen-7B) |
|---|---|
| Mean TTS | ~0.03 |
| Steps with TTS >= 0.7 | 2.3% |
| Steps with TTS >= 0.3 | 6.4% |
| Self-verification steps with TTS < 0.005 | 12% |

The distribution is heavily long-tailed: the vast majority of reasoning
steps have near-zero causal impact.

### How TTS works

For each reasoning step $s$ in a chain-of-thought, TTS measures what
happens when you perturb it (by introducing small numerical offsets)
under two contexts:

$$\text{TTS}(s) = \tfrac{1}{2}\bigl(|S_1(1) - S_0(1)| + |S_1(0) - S_0(0)|\bigr)$$

where $S_x(c) = P(y^* \mid \text{context}=c,\;\text{step}=x)$ and:
- $c=1$: preceding steps are **intact**, $c=0$: preceding steps are **perturbed**
- $x=1$: step $s$ is **intact**, $x=0$: step $s$ is **perturbed**

This captures both **necessity** (does perturbing the step hurt performance
when the context is intact?) and **sufficiency** (can the step drive the
correct answer even when the context is corrupted?).

### What we implement in Tinker

We replicate TTS computation using Tinker's `compute_logprobs_async` API:

1. **Generate CoT**: Sample from a thinking model (greedy, temperature=0)
   using the renderer's chat template. The model produces `<think>...</think>`
   blocks with extended reasoning.

2. **Segment steps**: Split the thinking text using discourse markers
   (numbered lists, transition words like "So", "Wait", "Therefore", etc.).

3. **Perturb numbers**: Apply 10-30% relative offsets to numerical values
   in the reasoning text, preserving grammatical structure.

4. **Early-exit confidence**: For each of the four conditions, build
   a sequence `[prompt + <think> CoT_prefix </think> \boxed{answer}]`
   using the renderer's chat template, then measure
   $P(\text{answer tokens} \mid \text{prefix})$ via `compute_logprobs_async`.

5. **Compute TTS** from the four confidence measurements.

**Approximations vs. the paper:**

- **Models:** The paper uses DeepSeek-R1-Distill models; we use Qwen3.5
  thinking models. Both produce `<think>...</think>` delimited CoT.
- **Early-exit cue:** The paper appends `"The final result is"` **inside**
  the reasoning block, probing "what would you predict mid-thought?" We
  close the `</think>` block and use `\boxed{}` format — probing "if you
  stopped thinking here, what would your final answer be?" Both measure how
  the model's answer-prediction **changes** when a step is perturbed (i.e.
  TTS is relative), so the choice of cue mainly shifts the baseline
  probability, not the TTS scores.
- **Confidence measurement:** We compute `exp(sum(logprobs))` over the
  answer tokens after the cue, giving P(y* | prefix). This is a
  continuous, fine-grained equivalent of the paper's model confidence.
- **Step segmentation:** Both the paper and our implementation use heuristic
  text-based segmentation. The exact boundary detection differs but produces
  comparable step counts.
- **No steering vectors:** The paper's Section 6 extracts steering directions
  from internal activations to control step reliance. Tinker does not expose
  residual-stream activations, so this part is not replicated.

---

### Setup

No special data download is needed. The experiment uses hardcoded math
problems. You only need a Tinker API key:

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

**GSM8K, larger model:**

```bash
python -m tinker_cookbook.recipes.true_thinking_score.analyze \
    dataset=gsm8k model_name=Qwen/Qwen3.5-27B n_problems=50
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
| `dataset` | `math` | Dataset: `math` (MATH-500) or `gsm8k` |
| `n_problems` | `50` | Number of problems to analyze |
| `concurrency` | `64` | Max parallel problems (steps within a problem are sequential) |
| `max_tokens` | `4096` | Max tokens for CoT generation |
| `seed` | `42` | Random seed for perturbation reproducibility |

---

### Results

**Qwen3.5-4B on 50 MATH-500 problems (1,521 total steps):**

| Metric | Paper (AIME, Qwen-7B) | Ours (MATH-500, Qwen3.5-4B) |
|---|---|---|
| Mean TTS | ~0.03 | 0.057 |
| Steps with TTS >= 0.7 | 2.3% | **2.2%** |
| Steps with TTS >= 0.3 | 6.4% | **6.7%** |
| Decorative steps (TTS <= 0.005) | — | 61.8% |
| Self-verification steps | — | 132 |
| Self-verification steps decorative | 12-21% | **62.1%** |

**Additional statistics:**
- Model accuracy: 60% (30/50 correct)
- Mean steps per problem: 30.4
- Total compute: 50 problems in ~14 minutes (concurrency=64)

### Findings

1. **The paper's core claim is validated at scale:** On 50 MATH-500 problems
   with 1,521 total steps, **only 2.2% have TTS >= 0.7** (paper: 2.3%) and
   **6.7% have TTS >= 0.3** (paper: 6.4%). The TTS distribution closely
   matches the paper despite using a different model family and dataset.

2. **Most reasoning is decorative:** 61.8% of steps have TTS <= 0.005 —
   perturbing these steps has essentially zero effect on the model's answer
   prediction. The model generates elaborate-looking reasoning that it
   doesn't actually use.

3. **Self-verification is often fake:** 62.1% of "Wait, let me re-check"
   steps are decorative (TTS <= 0.005). The model frequently performs
   self-verification rituals that don't causally influence its answer.
   This is higher than the paper's 12-21%, possibly because Qwen3.5-4B
   generates more verbose self-checks.

4. **TTS rises near the answer:** The final steps before the answer
   consistently have the highest TTS, suggesting the model "commits" to an
   answer path late in the chain. Early reasoning steps explore without
   making progress.

5. **Wrong answers have lower TTS:** Problems where the model gets the wrong
   answer tend to have near-zero TTS across all steps — the model never
   locks onto a viable solution path.

---

### Files

| File | Description |
|---|---|
| `tts.py` | Core TTS computation: step segmentation, number perturbation, early-exit confidence, TTS scoring |
| `analyze.py` | CLI entry point for running TTS analysis on MATH-500 or GSM8K |
| `run_small_experiment.py` | Quick validation script (3 hardcoded problems) |
| `tts_test.py` | 16 unit tests for segmentation, perturbation, and self-verification detection |
| `RESEARCH_NOTES.md` | Development notes and experiment log |

### References

- [Can Aha Moments Be Fake?](https://arxiv.org/abs/2510.24941) — Zhao et al., 2025
- [Tinker API docs](https://tinker-docs.thinkingmachines.ai/) — `compute_logprobs_async` reference
