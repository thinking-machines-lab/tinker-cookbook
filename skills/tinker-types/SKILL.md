---
name: tinker-types
description: Reference for Tinker SDK types — Datum, ModelInput, TensorData, SamplingParams, response types, error types, and helper functions. Use when the user needs to build training data, construct model inputs, understand response objects, or handle errors.
---

# Tinker SDK Types

Quick reference for the core types used throughout the Tinker SDK and cookbook.

## Reference

Read `docs/api-reference/types.md` for the complete type reference.

## Core data types

### Type hierarchy
```
Datum
├── model_input: ModelInput (list of chunks)
│   ├── EncodedTextChunk (token IDs)
│   └── ImageChunk (vision inputs)
└── loss_fn_inputs: dict[str, TensorData]
    └── TensorData (numpy/torch wrapper)
```

### ModelInput
```python
from tinker import ModelInput

mi = ModelInput.from_ints([1, 2, 3, 4, 5])  # From token list
tokens = mi.to_ints()                        # Back to list
length = mi.length                           # Token count (property)
mi2 = mi.append(chunk)                       # Append a chunk
mi3 = mi.append_int(42)                      # Append a single token
mi_empty = ModelInput.empty()                # Empty input
```

### TensorData
```python
from tinker import TensorData

td = TensorData.from_numpy(np.array([1.0, 0.0, 1.0]))  # From numpy
td = TensorData.from_torch(torch.tensor([1.0, 0.0]))    # From torch
arr = td.to_numpy()                                       # Back to numpy
tensor = td.to_torch()                                    # Back to torch
lst = td.tolist()                                         # Back to list
# Fields: data (flat list), dtype ("int64"|"float32"), shape (optional)
```

### Datum
```python
from tinker import Datum, ModelInput, TensorData

datum = Datum(
    model_input=ModelInput.from_ints(tokens),
    loss_fn_inputs={"weights": TensorData.from_numpy(weights_array)},
)
```

## Configuration types

### SamplingParams
```python
from tinker import SamplingParams

params = SamplingParams(
    max_tokens=256,        # Max generation length
    temperature=1.0,       # Sampling temperature
    top_k=50,              # Top-K sampling (-1 = no limit)
    top_p=0.95,            # Nucleus sampling
    stop=["<|eot_id|>"],   # Stop sequences (strings or token IDs)
    seed=42,               # Reproducible seed
)
```

### AdamParams
```python
from tinker import AdamParams

adam = AdamParams(
    learning_rate=2e-4,
    beta1=0.9,             # Gradient moving average
    beta2=0.95,            # Gradient squared moving average
    eps=1e-12,             # Numerical stability
    weight_decay=0.0,      # Decoupled weight decay
    grad_clip_norm=1.0,    # Global gradient norm clipping (0.0 = disabled)
)
```

### LoraConfig
```python
from tinker import LoraConfig

config = LoraConfig(
    rank=32,               # LoRA rank
    seed=None,             # Initialization seed
    train_mlp=True,        # Train MLP layers
    train_attn=True,       # Train attention layers
    train_unembed=True,    # Train unembedding layer
)
```

## Response types

### ForwardBackwardOutput
Returned by `forward_backward()` and `forward()`:
```python
result = tc.forward_backward(data=batch, loss_fn="cross_entropy")
result.metrics              # dict[str, float] — training metrics (includes loss)
result.loss_fn_outputs      # list[LossFnOutput] — per-sample outputs
result.loss_fn_output_type  # str — loss output class name
```

### SampleResponse / SampledSequence
Returned by `sample()`:
```python
response = sc.sample(prompt=mi, num_samples=4, sampling_params=params)
response.sequences                # list[SampledSequence]
response.prompt_logprobs          # Optional[list[Optional[float]]] — per-prompt-token logprobs
response.topk_prompt_logprobs     # Optional[list[Optional[list[tuple[int, float]]]]] — top-K

for seq in response.sequences:
    seq.tokens       # list[int] — generated token IDs
    seq.logprobs     # Optional[list[float]] — per-token logprobs
    seq.stop_reason  # StopReason: "length" | "stop"
```

### Other response types
- `OptimStepResponse` — confirms parameter update
- `SaveWeightsResponse` — `path: str` (tinker:// path to saved weights)
- `LoadWeightsResponse` — confirms loaded weights
- `GetInfoResponse` — `model_data: ModelData` (model_name, lora_rank, tokenizer_id)
- `GetServerCapabilitiesResponse` — `supported_models: list[SupportedModel]`
- `WeightsInfoResponse` — `base_model`, `lora_rank`, `is_lora`, `train_mlp`, `train_attn`, `train_unembed`

## Checkpoint and run types

```python
from tinker import TrainingRun, Checkpoint, CheckpointType, ParsedCheckpointTinkerPath

# TrainingRun — metadata about a training run
run.training_run_id    # str
run.base_model         # str
run.is_lora            # bool
run.lora_rank          # Optional[int]
run.last_checkpoint    # Optional[Checkpoint]
run.user_metadata      # Optional[dict[str, str]]

# Checkpoint — metadata about a saved checkpoint
ckpt.checkpoint_id     # str
ckpt.checkpoint_type   # CheckpointType: "training" | "sampler"
ckpt.tinker_path       # str (tinker:// path)
ckpt.size_bytes        # Optional[int]
ckpt.public            # bool
ckpt.expires_at        # Optional[datetime]

# Parse a tinker:// path
parsed = ParsedCheckpointTinkerPath.from_tinker_path("tinker://run-id/weights/ckpt-id")
parsed.training_run_id  # str
parsed.checkpoint_type  # CheckpointType
parsed.checkpoint_id    # str
```

## Error types

All exceptions inherit from `tinker.TinkerError`:
- **`APIError`** → **`APIStatusError`**: `BadRequestError` (400), `AuthenticationError` (401), `PermissionDeniedError` (403), `NotFoundError` (404), `ConflictError` (409), `UnprocessableEntityError` (422), `RateLimitError` (429), `InternalServerError` (500+)
- **`APIConnectionError`**, **`APITimeoutError`**, **`APIResponseValidationError`**
- **`RequestFailedError`** — async request failure with error category

## Cookbook helper functions

Use these instead of manual Datum construction:
- `tinker_cookbook.supervised.data.conversation_to_datum(messages, renderer, max_length, train_on_what)` — full SL pipeline
- `tinker_cookbook.supervised.common.datum_from_model_input_weights(model_input, weights, max_length)` — from ModelInput + weights
- `renderer.build_supervised_example(messages)` — returns `(ModelInput, weights)`

## Common pitfalls
- Use helper functions instead of manual dict construction for Datum
- `TensorData` wraps arrays — don't pass raw numpy/torch directly to `loss_fn_inputs`
- `ModelInput.from_ints()` expects a flat list of integers, not nested lists
- `ModelInput.length` is a property, not a method
- Handle `tinker.RateLimitError` in production code with exponential backoff
