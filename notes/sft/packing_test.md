# SFT Packing Test Results

Date: 2026-03-27

## Context Limit Finding

The Nemotron-3-Nano model has a **65,536 token context window** (not 131K or 256K).
- Tested by sending real tokens via sampling client
- 50K tokens (48K after encoding): OK
- 100K tokens: FAIL with "Prompt length plus max_tokens exceeds the model's context window: 100000 + 1 > 65536"
- **Implication**: max_packed_length should be 49152 (our standard cap), not 131072

## Unit Tests

All packing unit tests pass (`sft_packing_test.py`):
- Single example packing
- Multi-example bin-packing
- Oversized example truncation
- Empty example skipping
- Weight boundary preservation across packed examples
- Packing ratio efficiency

## Packing Smoke Test (50K sample, 20 steps)

**Config**: model=Nemotron-3-Nano, lr=3e-4, batch_size=16, lora_rank=64, max_length=49152, packing=True, max_packed_length=49152

**Packing Statistics**:
- 48,972 raw examples packed into 7,600 sequences
- Average **6.4 examples per packed sequence**
- ~550K loss tokens per step (vs ~2-3K without packing at same batch size)
- ~666K total tokens per step

**NLL Trajectory**:
```
step  0: train=0.6914  test=0.6941
step  1: train=0.6626
step  2: train=0.6541
step  3: train=0.6852
step  4: train=0.5963
step  5: train=0.6258
step  6: train=0.6817
step  7: train=0.6714
step  8: train=0.6442
step  9: train=0.6140
step 10: train=0.5942  test=0.6342
step 11: train=0.6164
step 12: train=0.6359
step 13: train=0.6169
step 14: train=0.5983
step 15: train=0.6245
step 16: train=0.6295
step 17: train=0.5021
step 18: train=0.6254
step 19: train=0.6226
```

**Min train NLL**: 0.502 (step 17)
**Test NLL**: 0.694 -> 0.634 after 10 steps

### Comparison with Unpacked Sweep

The unpacked sweep (same 50K sample) achieved min NLL ~0.40. However:
- Unpacked uses 49K max_length per individual sequence (mostly padding for short examples)
- Packed uses 49K sequences densely filled with ~6.4 examples each
- Packed processes **~180x more loss tokens per step** (550K vs ~3K)
- The higher NLL with packing is expected: each packed batch contains much more diverse data, making the loss harder to overfit
- Test NLL of 0.634 after just 10 steps is promising -- the model is learning on real diverse data, not memorizing a few long sequences

**Verdict**: Packing works correctly and is dramatically more efficient.

## Full SFT v2 Launch

Launched with nohup (running in background):
```
python run_full_sft_v2.py \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --lr 3e-4 --batch-size 64 --lora-rank 128 \
    --max-length 49152 --packing --max-packed-length 49152 \
    --save-every 1000 --eval-every 500 \
    --log-path /tmp/tinker-examples/nemotron_cascade_sft_v2_full \
    --data-dir ~/data/nemotron-cascade-2
```

Note: Used max-length=49152 instead of 131072 because the model's actual context window is 65,536 tokens.

Log file: `/tmp/tinker-examples/nemotron_cascade_sft_v2_full_nohup.log`
Metrics: `/tmp/tinker-examples/nemotron_cascade_sft_v2_full/metrics.jsonl`

## IF-RL batch=128 Status

Run is in progress. After 1 iteration (of 20):
- **Fraction correct: 71.4%**
- KL from reference: 0.003 (very small, just started)
- Avg completion tokens: 2,494 per episode
- Total episodes per batch: 1,568 (128 groups x ~12 valid per group after filtering constant reward)
- 100% mixed groups (no all-good or all-bad), good for learning signal
- Entropy: 1.12
- Time per iteration: ~2,495 seconds (~42 minutes) -- dominated by sampling (2,413s)
- Note: some rollouts took up to 2,413s (40 min), suggesting long-context completions or queuing

Metrics: `/tmp/tinker-examples/nano_ifrl_b128/metrics.jsonl`
WandB: https://wandb.ai/thinking-machines-lab-inc/nemotron-cascade-2-replication/runs/9yh0cogx
