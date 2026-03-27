# MCQA Environment Analysis

## Status: SLIGHT REGRESSION at lr=1e-5 and 3e-5

## Core Problem: Noisy Answer Extraction + Overly Generous Matching

### Issue 1: `extract_answer` has ambiguous fallbacks
The answer extraction pipeline has 5 cascading fallbacks:
1. `\boxed{...}` — good, unambiguous
2. "the answer is (X)" — good for single letter
3. `**X**` bold single letter — reasonable
4. Last standalone capital letter on its own line — OK
5. **Last capital letter in the final 200 chars** — VERY NOISY

Fallback 5 means almost any response will extract *some* letter. For a 4-option MCQA, random extraction has 25% chance of matching the answer by luck.

### Issue 2: `check_answer` is too generous
```python
if expected_norm in extracted_norm or extracted_norm in expected_norm:
    return True
```
This containment check means:
- Expected "A" matches extracted "ABC" or "ABCDEF" — false positives
- Expected "B" matches extracted "B" (correct) but also "BIG" — false positive
- For numeric answers, "1" matches "12", "21", etc.

This creates **false positive rewards** that pollute the GRPO signal.

### Issue 3: No thinking-chain filtering
The model uses `<think>` tags. If the answer extraction searches the full response (including reasoning), it may extract intermediate answer mentions rather than the final answer.

## Actionable Improvements

### P0: Fix answer extraction to reduce false positives
1. **Strip `<think>` block before extraction**: Only search the text after `</think>` (or the whole response if no think tags).
2. **Remove the containment check in `check_answer`**: Replace with exact match only. The current `expected_norm in extracted_norm` is a major source of noise.
3. **Tighten fallback 5**: Instead of "any capital letter in last 200 chars", require it to be near answer-like context (e.g., after "answer", "choice", "option").

### P1: Add answer normalization
- Strip whitespace, periods, parentheses from both sides
- Handle "A)" vs "A" vs "(A)" uniformly
- For numeric answers, parse to float and compare with tolerance

### P2: Consider partial credit
Currently binary (0 or 1). For multi-part questions, partial credit could help.

### P3: Verify data quality
- Check what fraction of `expected_answer` values are single letters (A/B/C/D) vs longer strings
- If some are longer (e.g., "The mitochondria..."), the single-letter extraction pipeline will always fail on those → those problems contribute 0 reward and add noise

## Expected Impact
Fixing the false-positive answer matching should turn the noisy regression into positive signal, since the model will no longer be "rewarded" for wrong answers that happen to contain the right letter.
