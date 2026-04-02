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
- **Early-exit measurement:** The paper appends "The final result is" and
  measures confidence. We compute the joint probability of the answer tokens
  via logprob summation — a continuous, fine-grained equivalent.
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

**Quick validation (3 problems, ~4 minutes):**

```bash
python -m tinker_cookbook.recipes.true_thinking_score.run_small_experiment
```

This runs 3 AMC-level math problems through Qwen3.5-4B, generates CoT,
computes TTS for each step, and reports aggregate statistics. Results are
saved to `/tmp/tinker-tts-experiment/tts_results.json`.

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
| `max_tokens` | `4096` | Max tokens for CoT generation |
| `temperature` | `0.0` | Greedy decoding (matches paper) |
| `seed` | `42` | Random seed for perturbation reproducibility |
| `min_step_chars` | `20` | Minimum characters for a step (shorter segments are merged) |

---

### Results

**Qwen3.5-4B on 3 AMC-level problems (54 total steps):**

| Metric | Paper (AIME, Qwen-7B) | Ours (AMC, Qwen3.5-4B) |
|---|---|---|
| Mean TTS | ~0.03 | 0.061 |
| Steps with TTS >= 0.7 | 2.3% | 1.9% |
| Decorative steps (TTS <= 0.005) | — | 59% |
| Self-verification steps found | — | 5 |
| Self-verification steps decorative | 12% | 40% (2/5) |

**Per-problem breakdown:**

| Problem | Steps | Mean TTS | High TTS | Decorative | Model correct |
|---|---|---|---|---|---|
| Inclusion-exclusion (divisible by 3,5,7) | 10 | 0.147 | 10% | 20% | Yes |
| $n^2 \equiv 1 \pmod{24}$ sum | 32 | 0.002 | 0% | 88% | No |
| Probability (balls) | 12 | 0.146 | 0% | 17% | No |

### Findings

1. **The paper's core claim is validated:** Most reasoning steps are decorative.
   On the hardest problem (32 steps of number theory), 88% of steps had
   TTS <= 0.005 — the model's lengthy exploration barely affected its output.

2. **Difficulty matters:** Easy problems produce fewer, higher-TTS steps.
   The model's confidence is concentrated in a small number of key calculations.
   Hard problems produce many decorative steps as the model "searches" without
   making progress.

3. **Self-verification can be fake:** "Wait, let me re-check" steps sometimes
   have near-zero TTS — the model appears to verify its work but the
   verification doesn't causally influence the answer.

4. **TTS rises at the end:** The final steps before the answer tend to have
   the highest TTS (steps 8-9 in problem 1: TTS=0.76, 0.62). This suggests
   the model "commits" to an answer path late in the reasoning chain.

---

### Files

| File | Description |
|---|---|
| `tts.py` | Core TTS computation: step segmentation, number perturbation, early-exit confidence, TTS scoring |
| `run_small_experiment.py` | End-to-end validation script with AMC-level problems |
| `tts_test.py` | Unit tests for segmentation, perturbation, and self-verification detection |
| `RESEARCH_NOTES.md` | Internal development notes and experiment log |

### References

- [Can Aha Moments Be Fake?](https://arxiv.org/abs/2510.24941) — Zhao et al., 2025
- [Tinker API docs](https://tinker-docs.thinkingmachines.ai/) — `compute_logprobs_async` reference
