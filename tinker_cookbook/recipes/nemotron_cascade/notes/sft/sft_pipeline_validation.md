# SFT Pipeline Validation (2026-03-27)

## Smoke Test: Nano + InterleavedChatDatasetBuilder

**Config**: Nano-30B, batch_size=64, lr=3e-4, cosine schedule, lora_rank=32, 10 steps
**Data**: safety + instruction_following subsets (interleaved via HF interleave_datasets)

| Step | Train NLL | Test NLL | LR |
|------|-----------|----------|-----|
| 0 | 0.615 | 0.670 | 3.00e-4 |
| 1 | 0.588 | — | 2.93e-4 |
| 2 | 0.544 | — | 2.71e-4 |
| 3 | **0.495** | — | 2.38e-4 |
| 4 | 0.597 | — | 1.96e-4 |
| 5 | 0.612 | **0.606** | 1.50e-4 |
| 6 | 0.651 | — | 1.04e-4 |
| 7 | 0.535 | — | 6.2e-5 |
| 8 | 0.563 | — | 2.9e-5 |
| 9 | 0.591 | — | 7.0e-6 |

**Pipeline validated:**
- InterleavedChatDatasetBuilder works with HF datasets
- Cosine LR schedule correct (0.0003 → 0.000007)
- Checkpoints save correctly (step 5 + final)
- Resume mechanism exists (get_last_checkpoint + start_batch)
- ~50K loss tokens/step at batch_size=64

## Full SFT Configuration (for Super 120B)

| Setting | Paper | Our Plan |
|---|---|---|
| Model | Nano-30B (full FT) | Super-120B:peft:262144 (LoRA) |
| Batch size | 64 packed × 256K = 16.4M tokens/step | batch_size=2048 raw examples (~15M tokens/step) |
| LR | 5e-5 → 5e-6 cosine with 200 warmup | 3e-4 cosine (no warmup/min_lr in Config yet) |
| β₂ | 0.98 | 0.98 |
| Steps | 33,000 | 33,000 |
| Data | All 8 SFT subsets packed into 256K | All 8 subsets via InterleavedChatDatasetBuilder |

## Gaps to Address Before Full Run
1. No warmup support in Config (paper uses 200 steps)
2. No min_lr support (paper uses 5e-6 floor)
3. Super 120B backend capacity limited — max 2-3 concurrent sessions
4. Need to verify batch_size=2048 works on Super (untested due to capacity)
