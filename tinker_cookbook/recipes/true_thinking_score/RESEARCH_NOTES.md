# True-Thinking Score (TTS) Recipe - Research Notes

## Paper: "Can Aha Moments Be Fake?"
- **Authors:** Jiachen Zhao, Yiyou Sun, Weiyan Shi, Dawn Song
- **arXiv:** 2510.24941v2

## Key Idea

Chain-of-thought (CoT) reasoning steps in LLMs can be either:
- **True-thinking steps**: Causally influence the model's final prediction
- **Decorative-thinking steps**: Appear to show reasoning but minimally affect outputs

The paper proposes the **True-Thinking Score (TTS)** to measure the causal contribution of each step.

## TTS Formula

```
TTS(s) = 1/2 * (|S_1(1) - S_0(1)| + |S_1(0) - S_0(0)|)
```

Where:
- `S_x(c) = Pr(Y=1 | C=c, do(X=x))` = model confidence in correct answer
- `C=1`: intact context (preceding steps unchanged)
- `C=0`: perturbed context (preceding steps have small numerical offsets)
- `X=1`: intact step s
- `X=0`: perturbed step s
- `Y=1`: model predicts correct answer y*

The formula captures both:
- **Necessity** (ATE_nec): does removing step s harm performance with intact context?
- **Sufficiency** (ATE_suf): can step s drive correct answers with corrupted context?

## Operational Procedure

1. Generate CoT reasoning for a math problem
2. Segment CoT into steps (by model's natural structure)
3. For each step s:
   a. Test with intact context + intact step: S_1(1)
   b. Test with intact context + perturbed step: S_0(1)
   c. Test with perturbed context + intact step: S_1(0)
   d. Test with perturbed context + perturbed step: S_0(0)
4. Early-exit prediction: append "The final result is" and get P(y*)
5. Compute TTS from the four measurements

## Perturbation Method
- "Small random numerical offsets to quantities appearing in reasoning text"
- Preserves semantic/grammatical coherence
- Only modifies numbers to break logical connections

## Key Findings
- On AIME (Qwen-7B): only 2.3% of steps have TTS >= 0.7, mean TTS ~ 0.03
- 12-21% of self-verification "aha moments" have near-zero TTS
- True and decorative steps are interleaved throughout CoT

## Models Used in Paper
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Llama-8B  
- Nemotron-1.5B (reasoning variant)

## Steering (Section 6)
- Extract "TrueThinking direction" vector: v = mean(activations of high-TTS steps) - mean(activations of low-TTS steps)
- Add to residual stream at specific layer during inference
- Flip rates: ~55% for Qwen on AMC/AIME

## Datasets
- AMC (American Mathematics Competitions)
- AIME 2020-2024
- MATH dataset
- CommonsenseQA

## What We Can Replicate with Tinker API

### Feasible:
1. **TTS computation** - Generate CoT, perturb, measure prediction changes via logprobs
2. **TTS distribution analysis** - Replicate the finding that most steps are decorative
3. **Self-verification analysis** - Identify "aha moments" and measure their TTS
4. **TTS-aware training** (novel extension) - Use TTS as signal for RL training

### Needs Investigation:
- **Steering vectors** - Requires access to internal activations (residual stream)
  - Tinker may support this via forward pass with activation hooks?
  - Need to check API docs

### Probably Not Feasible:
- Layer-wise activation extraction for steering direction computation
  (unless Tinker has activation access API)

## Experiment Plan

### Phase 1: TTS Computation (small scale)
- Pick a small model (e.g., Qwen3-0.6B or similar reasoning model)
- Use 5-10 math problems from GSM8K or MATH
- Generate CoT, segment, compute TTS for each step
- Validate: do we see the same distribution (mostly decorative)?

### Phase 2: Scale Up Analysis
- Run on full MATH/AMC datasets
- Compare across model sizes
- Analyze self-verification steps specifically

### Phase 3: TTS-Aware Training (novel contribution)
- Idea 1: RL reward that penalizes decorative reasoning
- Idea 2: Filter SFT data to only include high-TTS reasoning chains
- Idea 3: Train verifier that predicts TTS scores

## Experiment Results (2026-04-02)

### Small-scale validation (3 AMC-level problems, Qwen3.5-4B)
- Total steps analyzed: 54
- **Mean TTS: 0.061** (paper: ~0.03 on AIME)
- **High TTS (>=0.7): 1.9%** (paper: ~2.3%)
- **Decorative (<=0.005): 59%**
- TTS distribution is heavy-tailed, matching the paper

### Key findings
1. Easy problems produce fewer steps with high TTS (model is very confident)
2. Harder problems (n^2 ≡ 1 mod 24) produce 32 steps, nearly all decorative
3. Self-verification "Wait" steps found but often with near-zero TTS
4. The pattern is robust across problem types

### Resolved questions
- [x] Tinker API exposes logprobs via compute_logprobs_async ✓
- [x] Conditional probabilities: build_generation_prompt + append tokens + compute_logprobs ✓
- [x] Qwen3.5-4B is a good thinking model (produces <think> blocks) ✓
- [x] Step segmentation: discourse cues + numbered lists + markdown headers ✓
- [x] Perturbation: 10-30% relative offsets on numbers ✓

### Open questions
- [ ] How does TTS distribution change with model size?
- [ ] Can we use TTS as a reward signal for RL training?
- [ ] Does training on high-TTS-only data improve reasoning faithfulness?
