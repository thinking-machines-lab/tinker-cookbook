# Serialization Regression Test

A standalone script to test whether the user's pydantic version causes slow serialization of Tinker SDK payloads. This is the most common "invisible" performance regression — the user's training script works but each step takes minutes instead of seconds.

## When to use

- Step 3 timing shows high submit time (time to submit `forward_backward` call)
- Step 4 profiling shows time in `pydantic.model_dump()` or `__repr__`
- User has a non-standard pydantic version (beta, pre-release, or very new)

## The test script

```python
"""
Tinker payload serialization benchmark.

Tests whether pydantic model_dump() is fast on typical training payloads.
Expected: < 0.1s for BS=4, SEQ_LEN=8192.
If > 1s, the pydantic version likely has a serialization regression.

Usage:
    python serialization_test.py
"""
import sys
import time

import pydantic
import torch
from tinker._compat import model_dump
from tinker._version import __version__ as tinker_version
from tinker.types import (
    Datum,
    EncodedTextChunk,
    ForwardBackwardInput,
    ForwardBackwardRequest,
    ModelInput,
)

# ── Print versions ────────────────────────────────────────────────
print(f"Python:   {sys.version.split()[0]}")
print(f"pydantic: {pydantic.__version__}")
print(f"tinker:   {tinker_version}")
print(f"torch:    {torch.__version__}")
try:
    import numpy; print(f"numpy:    {numpy.__version__}")
except ImportError:
    print("numpy:    (not installed)")
print()

# ── Build a realistic payload ─────────────────────────────────────
BS = 4
SEQ_LEN = 8192
VOCAB = 32000

data = []
for _ in range(BS):
    tokens = torch.randint(0, VOCAB, (SEQ_LEN,)).tolist()
    model_input = ModelInput(chunks=[EncodedTextChunk(tokens=tokens)])
    weights = torch.ones(SEQ_LEN)
    target_tokens = torch.randint(0, VOCAB, (SEQ_LEN,)).tolist()
    data.append(
        Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "weights": weights,
            },
        )
    )

request = ForwardBackwardRequest(
    forward_backward_input=ForwardBackwardInput(
        data=data,
        loss_fn="cross_entropy",
        loss_fn_config=None,
    ),
    model_id="tinker://dummy/example",
    seq_id=1,
)

# ── Benchmark serialization ──────────────────────────────────────
t0 = time.perf_counter()
body = model_dump(request, exclude_unset=True, mode="json")
dt = time.perf_counter() - t0

print(f"Payload: batch_size={BS}, seq_len={SEQ_LEN}, total_tokens={BS * SEQ_LEN:,}")
print(f"model_dump() took {dt:.3f}s")
print()

if dt < 0.1:
    print("PASS: Serialization is fast.")
elif dt < 1.0:
    print("WARNING: Serialization is slower than expected. May cause minor slowdowns.")
else:
    print("FAIL: Serialization is very slow. This will cause major training slowdowns.")
    print(f"  Your pydantic version ({pydantic.__version__}) likely has a regression.")
    print("  Fix: pip install 'pydantic<2.13'  (or use the latest stable release)")
```

## Expected output (healthy)

```
Python:   3.12.9
pydantic: 2.12.5
tinker:   0.16.1
torch:    2.8.0
numpy:    2.2.5

Payload: batch_size=4, seq_len=8192, total_tokens=32,768
model_dump() took 0.007s

PASS: Serialization is fast.
```

## Example output (regression)

```
Python:   3.12.12
pydantic: 2.13.0b2
torch:    2.10.0
tinker:   0.16.1

Payload: batch_size=4, seq_len=8192, total_tokens=32,768
model_dump() took 147.231s

FAIL: Serialization is very slow. This will cause major training slowdowns.
  Your pydantic version (2.13.0b2) likely has a regression.
  Fix: pip install 'pydantic<2.13'  (or use the latest stable release)
```
