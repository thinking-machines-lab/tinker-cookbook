# SFT Data Download Tracking

## Source
nvidia/Nemotron-Cascade-2-SFT-Data on HuggingFace (8 subsets, ~24.8M examples total)

## Download Location
~/data/nemotron-cascade-2/sft_{subset}_full.jsonl

## Status (2026-03-27)

| Subset | Expected | File Size | Status | Notes |
|--------|----------|-----------|--------|-------|
| math | 5,226,364 | 229 GB | DONE | Very long reasoning (avg 13.5K tokens) |
| science | 2,717,163 | 43 GB | NEEDS CHECK | Download died, may be partial |
| chat | 13,972,873 | 198 GB | INCOMPLETE | Largest subset, download died at 198GB |
| instruction_following | 820,263 | 3.3 GB | DONE | |
| safety | 3,570 | 14 MB | DONE | |
| conversational_agent | 822,213 | 17 GB | NEEDS CHECK | Download died |
| swe | 439,610 | 34 GB | NEEDS CHECK | Download died |
| terminal_agent | 822,213 | 29 GB | NEEDS CHECK | Download died |

## Download Script
```bash
# Resume download for a subset (appends to existing file)
python3 -c "
import json
from datasets import load_dataset
ds = load_dataset('nvidia/Nemotron-Cascade-2-SFT-Data', name='SUBSET', split='train', streaming=True)
# Count existing lines
import os
existing = 0
fpath = os.path.expanduser('~/data/nemotron-cascade-2/sft_SUBSET_full.jsonl')
if os.path.exists(fpath):
    existing = sum(1 for _ in open(fpath))
# Skip existing and append
count = 0
with open(fpath, 'a') as f:
    for row in ds:
        count += 1
        if count <= existing:
            continue
        json.dump({'messages': row['messages']}, f)
        f.write('\n')
        if count % 500000 == 0:
            print(f'SUBSET: {count}', flush=True)
print(f'SUBSET: DONE ({count})')
"
```

## Issues Encountered
1. Sequential downloads too slow → switched to parallel (6 processes)
2. Parallel processes got killed when main process ran pkill
3. Some processes corrupted files by overwriting (using 'w' instead of 'a')
4. Chat subset is massive (~198GB+ for 14M examples)

## Data Characteristics
- All subsets use standard {"messages": [...]} format
- Math: 100% use <think> tags, avg 13.5K tokens/conversation
- Science: 100% <think>, avg 3.8K tokens
- IF: 63% <think>, 31% multi-turn, avg 947 tokens
- Safety: 100% <think>, all single-turn, avg 959 tokens
- System messages: mostly empty or "You are a helpful and harmless assistant"
