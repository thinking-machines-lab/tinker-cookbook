# SFT Data Download Tracking

## Source
nvidia/Nemotron-Cascade-2-SFT-Data on HuggingFace (8 subsets, ~24.8M examples total)

## Download Location
~/data/nemotron-cascade-2/sft_{subset}_full.jsonl

## Status (2026-03-27, verified complete)

All 8 subsets fully downloaded and verified via `wc -l` line counts.

| Subset | Line Count | File Size | Status | Notes |
|--------|-----------|-----------|--------|-------|
| math | 5,226,364 | 229 GB | DONE | Very long reasoning (avg 13.5K tokens) |
| science | 2,717,163 | 43 GB | DONE | Verified via wc -l |
| chat | 13,972,873 | 198 GB | DONE | Verified via wc -l |
| instruction_following | 820,263 | 3.3 GB | DONE | |
| safety | 3,570 | 14 MB | DONE | |
| conversational_agent | 822,213 | 17 GB | DONE | Verified via wc -l |
| swe | 439,610 | 34 GB | DONE | Verified via wc -l |
| terminal_agent | 485,667 | 29 GB | DONE | Actual count is 485,667 (not 822,213 as originally expected; the original script had a copy-paste error from conversational_agent). Verified by streaming the HF dataset. |

**Total: ~24,487,723 examples, ~553 GB on disk**

## Verification Commands
```bash
# Verify all files
wc -l ~/data/nemotron-cascade-2/sft_*_full.jsonl
```

## Issues Encountered
1. Sequential downloads too slow -> switched to parallel (6 processes)
2. Parallel processes got killed when main process ran pkill
3. Some processes corrupted files by overwriting (using 'w' instead of 'a')
4. Chat subset is massive (~198GB for 14M examples)
5. terminal_agent expected count was wrong in original download script (822,213 was copy-pasted from conversational_agent; actual dataset has 485,667 examples)

## Data Characteristics
- All subsets use standard {"messages": [...]} format
- Math: 100% use <think> tags, avg 13.5K tokens/conversation
- Science: 100% <think>, avg 3.8K tokens
- IF: 63% <think>, 31% multi-turn, avg 947 tokens
- Safety: 100% <think>, all single-turn, avg 959 tokens
- System messages: mostly empty or "You are a helpful and harmless assistant"
