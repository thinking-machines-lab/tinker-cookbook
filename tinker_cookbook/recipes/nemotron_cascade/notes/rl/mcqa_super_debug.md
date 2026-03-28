# MCQA Environment Debug — Nemotron Super Base Model

## Problem

MCQA RL environment gets reward=0.0 with frac_mixed=0.0 on the Nemotron Super
base model (`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:peft:262144`).
No training signal at all.

## Root Causes (Two Independent Issues)

### Issue 1: Model generates extremely long thinking chains

The base Nemotron Super model generates very long `<think>` reasoning blocks
for MCQA questions. At max_tokens=4096 or even 8192, the model often fills
the entire token budget without ever closing the `</think>` tag or producing
a final answer.

When `stop_reason="length"`, the original code gave a blanket reward=0,
resulting in 100% overlong responses and zero training signal.

**Evidence:**
- At max_tokens=4096: 100% overlong, 0% correct on all tested seeds
- At max_tokens=8192, seed=0 (pH cycling question): still overlong (>8K tokens of thinking)
- At max_tokens=8192, seed=42 (chemistry question): model finishes within budget
- The paper uses max_response=49K tokens, suggesting this was expected behavior

### Issue 2: Answer extraction missed common formats

The MCQA data uses diverse answer-format instructions. The model follows the
prompt's format instructions, but the extraction code only handled a few patterns.

Missing formats discovered:
- `Option Selected: X` — model follows prompt instruction, not extracted
- `Correct Option: X` — same issue
- `<final_answer>X</final_answer>` — XML-style, not extracted
- `((X))` — double parentheses, not extracted
- `*X*` — italic format, not extracted

With seed=42 at 8K tokens, the model answered correctly on all 4 rollouts
("Option Selected: A" for answer "A") but got reward=0 because the extraction
returned the last-10-chars fallback (`"elected: A"`) which didn't match.

## Fixes Applied

### 1. Expanded answer extraction (`_extract_answer_from_text`)

Added patterns for all formats found in the data:
- `\boxed{X}` (existing, now uses last match)
- `<final_answer>X</final_answer>` (new)
- `((X))` (new)
- `Option Selected: X` / `Correct Option: X` (new)
- `The answer is X` / `Answer: X` (existing)
- `**X**` / `*X*` (existing + new italic)
- Last standalone letter on own line (existing)

### 2. Think-content fallback for truncated responses

When `stop_reason="length"`, the env now:
1. Decodes the raw tokens (since `parse_response` skips block parsing for truncated responses)
2. Searches the thinking content (`include_think=True`) for answer patterns
3. Awards partial credit (0.5) for correct-but-overlong to provide signal

New functions: `_get_think_content()`, `extract_answer(include_think=True)`

### 3. Concise system prompt

Added default system prompt: "You are a helpful assistant. Think briefly,
then give your final answer. Do not over-explain. Keep your reasoning short
and focused."

Configurable via `MCQARLDatasetBuilder.system_prompt` (set to `None` to disable).

## Validation Results

### Before fix (seed=42, max_tokens=8192)
- reward: 0.0, correct: 0.0, overlong: 0.0
- Model answered "A" correctly but extraction failed on "Option Selected: A"

### After fix (seed=42, max_tokens=8192)
- reward: 1.0, correct: 1.0, overlong: 0.0
- Extraction correctly identifies "Option Selected: A" -> "A"

### After fix (seed=42, max_tokens=8192, groups_per_batch=3)
- reward: 1.0, correct: 1.0, overlong: 0.0
- All 3 groups (12 rollouts) answered correctly

## Remaining Considerations

1. **Token budget**: Some questions still trigger >8K token thinking chains.
   The paper's 49K budget handles these. For smaller budgets, the partial-credit
   mechanism (0.5 reward for overlong-but-correct) helps, but some very hard
   questions may still produce 0 signal.

2. **frac_mixed**: With the fix, easy questions get all-correct groups
   (frac_all_good=1.0) which also don't produce GRPO gradients. At scale with
   larger batches and diverse questions, mixed groups should emerge naturally.

3. **System prompt**: The concise system prompt is NOT in the original paper.
   Set `system_prompt=None` in the builder to disable it. With the paper's 49K
   token budget, it may not be needed.
