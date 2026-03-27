# Long-Context RL Environment Analysis

## Status: LOW REWARD (0.1) — LLM judge may not be scoring effectively

## Architecture
- Dataset: NarrativeQA (test split) — long documents with QA pairs
- Judge: Qwen3.5-397B-A17B, scores 0-10 normalized to [0,1]
- Reward computed in `compute_group_rewards` (like RLHF, deferred from step)
- Env stores `_model_answer` and `_stop_reason` as instance attributes for group reward

## Judge Prompt Analysis

### System prompt is clear but generic
The judge scores on a 0-10 scale considering "correct, complete, and well-supported by the context." This is reasonable.

### User template truncates context to 12K chars
```python
_MAX_JUDGE_CONTEXT_CHARS = 12_000
```
NarrativeQA documents can be very long. If the answer requires information beyond the first 12K chars, the judge can't verify it and may score 0. The student sees the full document but the judge sees only a truncated version — asymmetric information.

### Judge max_tokens = 32
```python
judge_max_tokens: int = 32
```
The judge is asked to output "ONLY the integer score, nothing else." But Qwen3.5 is a thinking model — it may want to use `<think>` tags before answering. 32 tokens may not be enough for the model to output its reasoning + score. If the response gets truncated before outputting a number, `_parse_judge_score` returns 0.0.

### Score parsing is fragile
```python
def _parse_judge_score(response_text: str) -> float:
    match = re.search(r'\b(\d{1,2})\b', response_text)
```
This finds the first 1-2 digit number. If the judge outputs thinking like "The answer addresses 3 of 5 points..." the parser would extract "3" instead of the final score. With 32 max tokens, the model likely outputs just a number, but if it outputs reasoning first, this is wrong.

## Data Quality Concerns

### NarrativeQA summaries as context
The `_parse_longcontext_example` function prefers `document.summary` over `document.text`:
```python
context = summary or text or ""
```
NarrativeQA summaries are typically short (~500 words). This is NOT a "long-context" task if summaries are used. The env name suggests long context but the actual context may be short.

### Reference answers in NarrativeQA
NarrativeQA reference answers are short (1-2 sentences). The judge compares against these but the model may give longer, more detailed answers that are correct but don't match the reference style.

## Actionable Improvements

### P0: Increase judge max_tokens to 256+
The thinking model needs space for `<think>` reasoning before outputting the score. 32 tokens is almost certainly causing truncation. Set `judge_max_tokens=256` or `512`.

### P1: Parse score from END of response, not beginning
Change `_parse_judge_score` to search from the end:
```python
# Find the LAST integer in the response (after thinking)
matches = re.findall(r'\b(\d{1,2})\b', response_text)
if matches:
    score = int(matches[-1])
```

### P2: Use document text instead of summary
For a long-context RL env, force `context = text or summary` instead of `summary or text`. If text is empty, the example isn't useful for long-context training.

### P3: Increase judge context window
12K chars is too little for documents that may be 50K+ chars. Increase `_MAX_JUDGE_CONTEXT_CHARS` to 32K or higher, since Qwen3.5-397B supports long contexts.

### P4: Add reference answer to judge prompt
The current judge prompt includes question + context + model answer, but NOT the reference answer. Adding the reference answer would make judging much more accurate:
```
Reference answer: {reference_answer}
```

## Expected Impact
P0 alone (increasing max_tokens) could increase reward from 0.1 to 0.3-0.5 by allowing the judge to actually output scores instead of truncating. P1+P4 would further improve judge accuracy.
