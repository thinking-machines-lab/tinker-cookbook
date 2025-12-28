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

Gradient clip norm for the optimizer. 0.0 means no clipping.

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

## `OptimStepResponse` Objects

```python
class OptimStepResponse(BaseModel)
```

#### `metrics`

Optimization step metrics as key-value pairs

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

## `WeightsInfoResponse` Objects

```python
class WeightsInfoResponse(BaseModel)
```

Minimal information for loading public checkpoints.

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

## `CheckpointArchiveUrlResponse` Objects

```python
class CheckpointArchiveUrlResponse(BaseModel)
```

#### `url`

Signed URL to download the checkpoint archive

#### `expires`

Unix timestamp when the signed URL expires, if available

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

## `TryAgainResponse` Objects

```python
class TryAgainResponse(BaseModel)
```

#### `request_id`

Request ID that is still pending

## `LoadWeightsRequest` Objects

```python
class LoadWeightsRequest(StrictBase)
```

#### `path`

A tinker URI for model weights at a specific step

#### `optimizer`

Whether to load optimizer state along with model weights

## `TelemetrySendRequest` Objects

```python
class TelemetrySendRequest(StrictBase)
```

#### `platform`

Host platform name

#### `sdk_version`

SDK version string

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

## `CheckpointsListResponse` Objects

```python
class CheckpointsListResponse(BaseModel)
```

#### `checkpoints`

List of available model checkpoints for the model

#### `cursor`

Pagination cursor information (None for unpaginated responses)

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

## `EncodedTextChunk` Objects

```python
class EncodedTextChunk(StrictBase)
```

#### `tokens`

Array of token IDs

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

## `SessionStartEvent` Objects

```python
class SessionStartEvent(BaseModel)
```

#### `event`

Telemetry event type

#### `severity`

Log severity level

## `TrainingRunsResponse` Objects

```python
class TrainingRunsResponse(BaseModel)
```

#### `training_runs`

List of training runs

#### `cursor`

Pagination cursor information

## `SaveWeightsResponse` Objects

```python
class SaveWeightsResponse(BaseModel)
```

#### `path`

A tinker URI for model weights at a specific step

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

## `ForwardBackwardOutput` Objects

```python
class ForwardBackwardOutput(BaseModel)
```

#### `loss_fn_output_type`

The class name of the loss function output records (e.g., 'TorchLossReturn', 'ArrayRecord').

#### `loss_fn_outputs`

Dictionary mapping field names to tensor data

#### `metrics`

Training metrics as key-value pairs

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

## `CreateSamplingSessionResponse` Objects

```python
class CreateSamplingSessionResponse(BaseModel)
```

#### `sampling_session_id`

The generated sampling session ID

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

## `TelemetryBatch` Objects

```python
class TelemetryBatch(BaseModel)
```

#### `platform`

Host platform name

#### `sdk_version`

SDK version string

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

## `SaveWeightsRequest` Objects

```python
class SaveWeightsRequest(StrictBase)
```

#### `path`

A file/directory name for the weights

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

## `FutureRetrieveRequest` Objects

```python
class FutureRetrieveRequest(StrictBase)
```

#### `request_id`

The ID of the request to retrieve

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

## `SaveWeightsForSamplerRequest` Objects

```python
class SaveWeightsForSamplerRequest(StrictBase)
```

#### `path`

A file/directory name for the weights

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

## `LoadWeightsResponse` Objects

```python
class LoadWeightsResponse(BaseModel)
```

#### `path`

A tinker URI for model weights at a specific step
