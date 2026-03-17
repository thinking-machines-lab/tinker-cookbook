---
name: tinker-types
description: Reference for Tinker SDK types — Datum, ModelInput, TensorData, SamplingParams, and helper functions for constructing them. Use when the user needs to build training data, construct model inputs, or understand the type hierarchy.
---

# Tinker SDK Types

Quick reference for the core types used throughout the Tinker SDK and cookbook.

## Reference

Read `docs/api-reference/types.md` for the complete type reference.

## Type hierarchy

```
Datum
├── model_input: ModelInput (list of chunks)
│   ├── EncodedTextChunk (token IDs)
│   └── ImageChunk (vision inputs)
└── loss_fn_inputs: dict[str, TensorData]
    └── TensorData (numpy/torch wrapper)
```

## ModelInput

A sequence of chunks representing the model's input (tokens + optional images).

```python
from tinker.types import ModelInput

# From token list
mi = ModelInput.from_ints([1, 2, 3, 4, 5])

# Get tokens back
tokens = mi.to_ints()

# Length
length = mi.length()

# Append
mi2 = mi.append(another_chunk)

# Empty
mi_empty = ModelInput.empty()
```

## TensorData

Wrapper for numpy arrays or torch tensors with shape info.

```python
from tinker.types import TensorData
import numpy as np

# From numpy
td = TensorData.from_numpy(np.array([1.0, 0.0, 1.0, 0.0]))

# From torch
import torch
td = TensorData.from_torch(torch.tensor([1.0, 0.0, 1.0, 0.0]))

# Convert back
arr = td.to_numpy()
tensor = td.to_torch()
```

## Datum

A single training sample: model input + loss function inputs.

```python
from tinker.types import Datum, ModelInput, TensorData

datum = Datum(
    model_input=ModelInput.from_ints(tokens),
    loss_fn_inputs={
        "weights": TensorData.from_numpy(weights_array),
    },
)
```

## SamplingParams

Controls text generation behavior.

```python
from tinker.types import SamplingParams

params = SamplingParams(
    max_tokens=256,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    stop=["<|eot_id|>"],  # Stop sequences (strings or token IDs)
    seed=42,
)
```

## AdamParams

Optimizer configuration.

```python
from tinker.types import AdamParams

adam = AdamParams(
    learning_rate=2e-4,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0,
    grad_clip_norm=1.0,
)
```

## Helper functions (use these instead of manual construction)

The cookbook provides helpers that handle the boilerplate:

```python
# SL: conversation → datum (full pipeline)
from tinker_cookbook.supervised.data import conversation_to_datum
datum = conversation_to_datum(messages, renderer, max_length, train_on_what)

# SL: model_input + weights → datum
from tinker_cookbook.supervised.common import datum_from_model_input_weights
datum = datum_from_model_input_weights(model_input, weights, max_length)

# Renderer: messages → (model_input, weights)
model_input, weights = renderer.build_supervised_example(messages)
```

See `tinker_cookbook/supervised/data.py` and `tinker_cookbook/supervised/common.py` for implementations.

## Common pitfalls
- Use helper functions (`conversation_to_datum`, `datum_from_model_input_weights`) instead of manual dict construction
- `TensorData` wraps arrays — don't pass raw numpy/torch directly to `loss_fn_inputs`
- `ModelInput.from_ints()` expects a flat list of integers, not nested lists
- Call `datum.convert_tensors()` if you used torch.Tensor or numpy arrays directly in `loss_fn_inputs`
