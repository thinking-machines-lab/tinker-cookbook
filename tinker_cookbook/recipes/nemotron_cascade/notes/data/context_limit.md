# Nemotron-3-Nano Context Limit on Tinker

## Tested 2026-03-27

The **actual inference context limit is 65,536 tokens (64K)**.

Creating sampling clients with `:peft:131072` or `:peft:262144` succeeds,
but actual sampling fails above 65K:

```
Error code: 400 - Prompt length plus max_tokens exceeds the model's
context window: 100000 prompt tokens + 1 max_tokens > 65536.
```

## Test Results

| Input Tokens | Result |
|-------------|--------|
| 1,000 | OK |
| 10,000 | OK |
| 50,000 | OK |
| 100,000 | FAIL (>65536) |
| 131,072 | FAIL |

## Implications

- **SFT packing**: Use max_packed_length=49152 (49K, safe margin)
- **RL max_tokens**: 49K is fine for most stages
- **Code RL**: Paper uses 118K — we're limited to 49K
- **SWE**: Paper uses 98K-256K — we're limited to 49K

This is a Tinker platform limit, not a model architecture limit
(Nemotron-3-Nano supports 128K+ in other deployments).
