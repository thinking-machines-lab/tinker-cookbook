## `LoadWeightsResponse` Objects

```python
class LoadWeightsResponse(BaseModel)
```

#### `path`

A tinker URI for model weights at a specific step

## `WeightsInfoResponse` Objects

```python
class WeightsInfoResponse(BaseModel)
```

Minimal information for loading public checkpoints.

## `LoadWeightsRequest` Objects

```python
class LoadWeightsRequest(StrictBase)
```

#### `path`

A tinker URI for model weights at a specific step

#### `optimizer`

Whether to load optimizer state along with model weights

## `CreateModelRequest` Objects

```python
class CreateModelRequest(StrictBase)
```

#### `base_model`

The name of the base model to fine-tune (e.g., 'Qwen/Qwen3-8B').

#### `user_metadata`

Optional metadata about this model/training run, set by the end-user.

#### `lora_config`

LoRA configuration

## `UnhandledExceptionEvent` Objects

```python
class UnhandledExceptionEvent(BaseModel)
```

#### `event`

Telemetry event type

#### `severity`

Log severity level

#### `traceback`

Optional Python traceback string

## `Datum` Objects

```python
class Datum(StrictBase)
```

#### `loss_fn_inputs`

Dictionary mapping field names to tensor data

#### `convert_tensors`

```python
def convert_tensors(cls, data: Any) -> Any
```

Convert torch.Tensor and numpy arrays to TensorData in loss_fn_inputs during construction.

## `Checkpoint` Objects

```python
class Checkpoint(BaseModel)
```

#### `checkpoint_id`

The checkpoint ID

#### `checkpoint_type`

The type of checkpoint (training or sampler)

#### `time`

The time when the checkpoint was created

#### `tinker_path`

The tinker path to the checkpoint

#### `size_bytes`

The size of the checkpoint in bytes

#### `public`

Whether the checkpoint is publicly accessible

#### `expires_at`

When this checkpoint expires (None = never expires)

## `ParsedCheckpointTinkerPath` Objects

```python
class ParsedCheckpointTinkerPath(BaseModel)
```

#### `tinker_path`

The tinker path to the checkpoint

#### `training_run_id`

The training run ID

#### `checkpoint_type`

The type of checkpoint (training or sampler)

#### `checkpoint_id`

The checkpoint ID

#### `from_tinker_path`

```python
def from_tinker_path(cls, tinker_path: str) -> "ParsedCheckpointTinkerPath"
```

Parse a tinker path to an instance of ParsedCheckpointTinkerPath

## `SamplingParams` Objects

```python
class SamplingParams(BaseModel)
```

#### `max_tokens`

Maximum number of tokens to generate

#### `seed`

Random seed for reproducible generation

#### `stop`

Stop sequences for generation

#### `temperature`

Sampling temperature

#### `top_k`

Top-k sampling parameter (-1 for no limit)

#### `top_p`

Nucleus sampling probability

## `SaveWeightsForSamplerRequest` Objects

```python
class SaveWeightsForSamplerRequest(StrictBase)
```

#### `path`

A file/directory name for the weights

#### `ttl_seconds`

TTL in seconds for this checkpoint (None = never expires)

## `ModelInput` Objects

```python
class ModelInput(StrictBase)
```

#### `chunks`

Sequence of input chunks (formerly TokenSequence)

#### `from_ints`

```python
def from_ints(cls, tokens: List[int]) -> "ModelInput"
```

Create a ModelInput from a list of ints (tokens).

#### `to_ints`

```python
def to_ints() -> List[int]
```

Convert the ModelInput to a list of ints (tokens)
Throws exception if there are any non-token chunks

#### `length`

```python
def length() -> int
```

Return the total context length used by this ModelInput.

#### `empty`

```python
def empty(cls) -> "ModelInput"
```

Create an empty ModelInput.

#### `append`

```python
def append(chunk: ModelInputChunk) -> "ModelInput"
```

Add a new chunk, return a new ModelInput.

#### `append_int`

```python
def append_int(token: int) -> "ModelInput"
```

Add a new token, return a new ModelInput.

## `SessionEndEvent` Objects

```python
class SessionEndEvent(BaseModel)
```

#### `duration`

ISO 8601 duration string

#### `event`

Telemetry event type

#### `severity`

Log severity level

## `CreateSamplingSessionResponse` Objects

```python
class CreateSamplingSessionResponse(BaseModel)
```

#### `sampling_session_id`

The generated sampling session ID

## `CheckpointsListResponse` Objects

```python
class CheckpointsListResponse(BaseModel)
```

#### `checkpoints`

List of available model checkpoints for the model

#### `cursor`

Pagination cursor information (None for unpaginated responses)

## `SampleResponse` Objects

```python
class SampleResponse(BaseModel)
```

#### `prompt_logprobs`

If prompt_logprobs was set to true in the request, logprobs are computed for
every token in the prompt. The `prompt_logprobs` response contains a float32
value for every token in the prompt.

#### `topk_prompt_logprobs`

If topk_prompt_logprobs was set to a positive integer k in the request,
the top-k logprobs are computed for every token in the prompt. The
`topk_prompt_logprobs` response contains, for every token in the prompt,
a list of up to k (token_id, logprob) tuples.

## `FutureRetrieveRequest` Objects

```python
class FutureRetrieveRequest(StrictBase)
```

#### `request_id`

The ID of the request to retrieve

#### `allow_metadata_only`

When True, the server may return only response metadata (status and size)
instead of the full payload if the response exceeds the server's inline size limit.

## `ForwardBackwardOutput` Objects

```python
class ForwardBackwardOutput(BaseModel)
```

#### `loss_fn_output_type`

The class name of the loss function output records (e.g., 'TorchLossReturn', 'ArrayRecord').

#### `loss_fn_outputs`

Dictionary mapping field names to tensor data

#### `metrics`

Training metrics as key-value pairs.

The following metrics are recorded only during MoE (Mixture of Experts) training.
Note: Don't fixate on the exact values of these metrics at the start of training.
Different models on different data will have different initial values. How these
metrics evolve over training is what matters.

In the definitions below, *perfect balance* means ``total_tokens / num_experts``
— the number of tokens each expert would receive if routing were perfectly uniform.

- ``e_frac_with_tokens:mean``: Fraction of experts that received at least one token,
  averaged across layers. A value of 1.0 means every expert got work; 0.5 means half
  were idle. Decreasing over time is concerning (routing collapse).

- ``e_frac_oversubscribed:mean``: Fraction of experts receiving more tokens than
  perfect balance, averaged across layers. Increasing over time is concerning.

- ``e_max_violation:mean``: How much the most overloaded expert exceeds perfect
  balance, as a fraction of perfect balance, averaged across layers. Computed as
  ``(max_tokens - perfect_balance) / perfect_balance``. A value of 2.0 means the
  busiest expert got 3x the fair share. Increasing over time is concerning.

- ``e_max_violation:max``: Same as ``e_max_violation:mean`` but takes the max
  across layers instead of the mean. Shows the worst-case load imbalance in any
  single layer.

- ``e_min_violation:mean``: How much the least loaded expert is below perfect
  balance, as a fraction of perfect balance, averaged across layers. Computed as
  ``(min_tokens - perfect_balance) / perfect_balance``. A value of -0.5 means the
  least-used expert got half the fair share; -1.0 means it got nothing. Typically
  negative. Decreasing over time (more negative) is concerning.

## `ModelData` Objects

```python
class ModelData(BaseModel)
```

Metadata about a model's architecture and configuration.

#### `arch`

The model architecture identifier.

#### `model_name`

The human-readable model name.

#### `tokenizer_id`

The identifier of the tokenizer used by this model.

## `GetInfoResponse` Objects

```python
class GetInfoResponse(BaseModel)
```

Response containing information about a training client's model.

#### `type`

Response type identifier.

#### `model_data`

Detailed metadata about the model.

#### `model_id`

Unique identifier for the model.

#### `is_lora`

Whether this is a LoRA fine-tuned model.

#### `lora_rank`

The rank of the LoRA adaptation, if applicable.

#### `model_name`

The name of the model.

## `SaveWeightsResponse` Objects

```python
class SaveWeightsResponse(BaseModel)
```

#### `path`

A tinker URI for model weights at a specific step

## `LoraConfig` Objects

```python
class LoraConfig(StrictBase)
```

#### `rank`

LoRA rank (dimension of low-rank matrices)

#### `seed`

Seed used for initialization of LoRA weights.

Useful if you need deterministic or reproducible initialization of weights.

#### `train_unembed`

Whether to add lora to the unembedding layer

#### `train_mlp`

Whether to add loras to the MLP layers (including MoE layers)

#### `train_attn`

Whether to add loras to the attention layers

## `SaveWeightsForSamplerResponseInternal` Objects

```python
class SaveWeightsForSamplerResponseInternal(BaseModel)
```

#### `path`

A tinker URI for model weights for sampling at a specific step

#### `sampling_session_id`

The generated sampling session ID

## `SaveWeightsForSamplerResponse` Objects

```python
class SaveWeightsForSamplerResponse(BaseModel)
```

#### `path`

A tinker URI for model weights for sampling at a specific step

## `CreateSamplingSessionRequest` Objects

```python
class CreateSamplingSessionRequest(StrictBase)
```

#### `session_id`

The session ID to create the sampling session within

#### `sampling_session_seq_id`

Sequence ID for the sampling session within the session

#### `base_model`

Optional base model name to sample from.

Is inferred from model_path, if provided. If sampling against a base model, this
is required.

#### `model_path`

Optional tinker:// path to your model weights or LoRA weights.

If not provided, samples against the base model.

## `OptimStepResponse` Objects

```python
class OptimStepResponse(BaseModel)
```

#### `metrics`

Optimization step metrics as key-value pairs

## `SampleRequest` Objects

```python
class SampleRequest(StrictBase)
```

#### `num_samples`

Number of samples to generate

#### `base_model`

Optional base model name to sample from.

Is inferred from model_path, if provided. If sampling against a base model, this
is required.

#### `model_path`

Optional tinker:// path to your model weights or LoRA weights.

If not provided, samples against the base model.

#### `sampling_session_id`

Optional sampling session ID to use instead of model_path/base_model.

If provided along with seq_id, the model configuration will be loaded from the
sampling session. This is useful for multi-turn conversations.

#### `seq_id`

Sequence ID within the sampling session.

Required when sampling_session_id is provided. Used to generate deterministic
request IDs for the sampling request.

#### `prompt_logprobs`

If set to `true`, computes and returns logprobs on the prompt tokens.

Defaults to false.

#### `topk_prompt_logprobs`

If set to a positive integer, returns the top-k logprobs for each prompt token.

## `TrainingRun` Objects

```python
class TrainingRun(BaseModel)
```

#### `training_run_id`

The unique identifier for the training run

#### `base_model`

The base model name this model is derived from

#### `model_owner`

The owner/creator of this model

#### `is_lora`

Whether this model uses LoRA (Low-Rank Adaptation)

#### `corrupted`

Whether the model is in a corrupted state

#### `lora_rank`

The LoRA rank if this is a LoRA model, null otherwise

#### `last_request_time`

The timestamp of the last request made to this model

#### `last_checkpoint`

The most recent training checkpoint, if available

#### `last_sampler_checkpoint`

The most recent sampler checkpoint, if available

#### `user_metadata`

Optional metadata about this training run, set by the end-user

## `TelemetrySendRequest` Objects

```python
class TelemetrySendRequest(StrictBase)
```

#### `platform`

Host platform name

#### `sdk_version`

SDK version string

## `CheckpointArchiveUrlResponse` Objects

```python
class CheckpointArchiveUrlResponse(BaseModel)
```

#### `url`

Signed URL to download the checkpoint archive

#### `expires`

Unix timestamp when the signed URL expires, if available

## `SupportedModel` Objects

```python
class SupportedModel(BaseModel)
```

Information about a model supported by the server.

#### `model_name`

The name of the supported model.

## `GetServerCapabilitiesResponse` Objects

```python
class GetServerCapabilitiesResponse(BaseModel)
```

Response containing the server's supported models and capabilities.

#### `supported_models`

List of models available on the server.

## `SessionStartEvent` Objects

```python
class SessionStartEvent(BaseModel)
```

#### `event`

Telemetry event type

#### `severity`

Log severity level

## `GenericEvent` Objects

```python
class GenericEvent(BaseModel)
```

#### `event`

Telemetry event type

#### `event_name`

Low-cardinality event name

#### `severity`

Log severity level

#### `event_data`

Arbitrary structured JSON payload

## `TryAgainResponse` Objects

```python
class TryAgainResponse(BaseModel)
```

#### `request_id`

Request ID that is still pending

## `TrainingRunsResponse` Objects

```python
class TrainingRunsResponse(BaseModel)
```

#### `training_runs`

List of training runs

#### `cursor`

Pagination cursor information

## `ForwardBackwardInput` Objects

```python
class ForwardBackwardInput(StrictBase)
```

#### `data`

Array of input data for the forward/backward pass

#### `loss_fn`

Fully qualified function path for the loss function

#### `loss_fn_config`

Optional configuration parameters for the loss function (e.g., PPO clip thresholds, DPO beta)

## `ImageAssetPointerChunk` Objects

```python
class ImageAssetPointerChunk(StrictBase)
```

#### `format`

Image format

#### `location`

Path or URL to the image asset

#### `expected_tokens`

Expected number of tokens this image represents.
This is only advisory: the tinker backend will compute the number of tokens
from the image, and we can fail requests quickly if the tokens does not
match expected_tokens.

## `TelemetryBatch` Objects

```python
class TelemetryBatch(BaseModel)
```

#### `platform`

Host platform name

#### `sdk_version`

SDK version string

## `TensorData` Objects

```python
class TensorData(StrictBase)
```

#### `data`

Flattened tensor data as array of numbers.

#### `shape`

Optional.

The shape of the tensor (see PyTorch tensor.shape). The shape of a
one-dimensional list of length N is `(N,)`. Can usually be inferred if not
provided, and is generally inferred as a 1D tensor.

#### `to_numpy`

```python
def to_numpy() -> npt.NDArray[Any]
```

Convert TensorData to numpy array.

#### `to_torch`

```python
def to_torch() -> "torch.Tensor"
```

Convert TensorData to torch tensor.

## `EncodedTextChunk` Objects

```python
class EncodedTextChunk(StrictBase)
```

#### `tokens`

Array of token IDs

## `AdamParams` Objects

```python
class AdamParams(StrictBase)
```

#### `learning_rate`

Learning rate for the optimizer

#### `beta1`

Coefficient used for computing running averages of gradient

#### `beta2`

Coefficient used for computing running averages of gradient square

#### `eps`

Term added to the denominator to improve numerical stability

#### `weight_decay`

Weight decay for the optimizer. Uses decoupled weight decay.

#### `grad_clip_norm`

Maximum global gradient norm. If the global gradient norm is greater than this value, it will be clipped to this value. 0.0 means no clipping.

## `ImageChunk` Objects

```python
class ImageChunk(StrictBase)
```

#### `data`

Image data as bytes

#### `format`

Image format

#### `expected_tokens`

Expected number of tokens this image represents.
This is only advisory: the tinker backend will compute the number of tokens
from the image, and we can fail requests quickly if the tokens does not
match expected_tokens.

#### `validate_data`

```python
def validate_data(cls, value: Union[bytes, str]) -> bytes
```

Deserialize base64 string to bytes if needed.

#### `serialize_data`

```python
def serialize_data(value: bytes) -> str
```

Serialize bytes to base64 string for JSON.

## `SampledSequence` Objects

```python
class SampledSequence(BaseModel)
```

#### `stop_reason`

Reason why sampling stopped

#### `tokens`

List of generated token IDs

#### `logprobs`

Log probabilities for each token (optional)

## `Cursor` Objects

```python
class Cursor(BaseModel)
```

#### `offset`

The offset used for pagination

#### `limit`

The maximum number of items requested

#### `total_count`

The total number of items available

## `SaveWeightsRequest` Objects

```python
class SaveWeightsRequest(StrictBase)
```

#### `path`

A file/directory name for the weights

#### `ttl_seconds`

TTL in seconds for this checkpoint (None = never expires)

