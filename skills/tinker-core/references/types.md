# Tinker SDK Types

Complete reference for all SDK types.

## Reference

Source: `tinker_cookbook/` Python package.

## Core data types

### ModelInput

```python
from tinker import ModelInput

mi = ModelInput.from_ints([1, 2, 3, 4, 5])
tokens = mi.to_ints()
length = mi.length           # Property, not method
mi2 = mi.append(chunk)
mi3 = mi.append_int(42)
mi_empty = ModelInput.empty()
```

### TensorData

```python
from tinker import TensorData

td = TensorData.from_numpy(np.array([1.0, 0.0, 1.0]))
td = TensorData.from_torch(torch.tensor([1.0, 0.0]))
arr = td.to_numpy()
tensor = td.to_torch()
lst = td.tolist()
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
    max_tokens=256, temperature=1.0, top_k=50, top_p=0.95,
    stop=["<|eot_id|>"], seed=42,
)
```

### AdamParams

```python
from tinker import AdamParams

adam = AdamParams(
    learning_rate=2e-4, beta1=0.9, beta2=0.95, eps=1e-12,
    weight_decay=0.0, grad_clip_norm=1.0,
)
```

### LoraConfig

```python
from tinker import LoraConfig

config = LoraConfig(rank=32, seed=None, train_mlp=True, train_attn=True, train_unembed=True)
```

## Response types

### ForwardBackwardOutput

```python
result = tc.forward_backward(data=batch, loss_fn="cross_entropy")
result.metrics              # dict[str, float]
result.loss_fn_outputs      # list[LossFnOutput]
result.loss_fn_output_type  # str
```

### SampleResponse / SampledSequence

```python
response = sc.sample(prompt=mi, num_samples=4, sampling_params=params)
response.sequences                # list[SampledSequence]
response.prompt_logprobs          # Optional[list[Optional[float]]]
response.topk_prompt_logprobs     # Optional[list[Optional[list[tuple[int, float]]]]]

for seq in response.sequences:
    seq.tokens       # list[int]
    seq.logprobs     # Optional[list[float]]
    seq.stop_reason  # StopReason: "length" | "stop"
```

### Other response types

- `OptimStepResponse` — confirms parameter update
- `SaveWeightsResponse` — `path: str` (tinker:// path)
- `LoadWeightsResponse` — confirms loaded weights
- `GetInfoResponse` — `model_data: ModelData` (model_name, lora_rank, tokenizer_id)
- `GetServerCapabilitiesResponse` — `supported_models: list[SupportedModel]`
- `WeightsInfoResponse` — `base_model`, `lora_rank`, `is_lora`, `train_mlp`, `train_attn`, `train_unembed`

## Checkpoint and run types

```python
from tinker import TrainingRun, Checkpoint, CheckpointType, ParsedCheckpointTinkerPath

# TrainingRun
run.training_run_id, run.base_model, run.is_lora, run.lora_rank
run.last_checkpoint, run.user_metadata

# Checkpoint
ckpt.checkpoint_id, ckpt.checkpoint_type, ckpt.tinker_path
ckpt.size_bytes, ckpt.public, ckpt.expires_at

# Parse a tinker:// path
parsed = ParsedCheckpointTinkerPath.from_tinker_path("tinker://run-id/weights/ckpt-id")
parsed.training_run_id, parsed.checkpoint_type, parsed.checkpoint_id
```

## Error types

All exceptions inherit from `tinker.TinkerError`:
- **`APIStatusError`**: `BadRequestError` (400), `AuthenticationError` (401), `PermissionDeniedError` (403), `NotFoundError` (404), `ConflictError` (409), `UnprocessableEntityError` (422), `RateLimitError` (429), `InternalServerError` (500+)
- **`APIConnectionError`**, **`APITimeoutError`**, **`APIResponseValidationError`**
- **`RequestFailedError`** — async request failure with error category

## Cookbook helper functions

Use these instead of manual Datum construction:
- `tinker_cookbook.supervised.data.conversation_to_datum(messages, renderer, max_length, train_on_what)` — full SL pipeline
- `tinker_cookbook.supervised.common.datum_from_model_input_weights(model_input, weights, max_length)` — from ModelInput + weights
- `renderer.build_supervised_example(messages)` — returns `(ModelInput, weights)`
