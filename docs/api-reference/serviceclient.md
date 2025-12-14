ServiceClient for Tinker API.

## `ServiceClient` Objects

```python
class ServiceClient(TelemetryProvider)
```

The ServiceClient is the main entry point for the Tinker API. It provides methods to:
- Query server capabilities and health status
- Generate TrainingClient instances for model training workflows
- Generate SamplingClient instances for text generation and inference
- Generate RestClient instances for REST API operations like listing weights

Args:
    **kwargs: advanced options passed to the underlying HTTP client,
             including API keys, headers, and connection settings.

Example:
```python
# Near instant
client = ServiceClient()

# Takes a moment as we initialize the model and assign resources
training_client = client.create_lora_training_client(base_model="Qwen/Qwen3-8B")

# Near-instant
sampling_client = client.create_sampling_client(base_model="Qwen/Qwen3-8B")

# Near-instant
rest_client = client.create_rest_client()
```

#### `get_server_capabilities`

```python
def get_server_capabilities() -> types.GetServerCapabilitiesResponse
```

Query the server's supported features and capabilities.

Returns:
- `GetServerCapabilitiesResponse` with available models, features, and limits

Example:
```python
capabilities = service_client.get_server_capabilities()
print(f"Supported models: {capabilities.supported_models}")
print(f"Max batch size: {capabilities.max_batch_size}")
```

#### `get_server_capabilities_async`

```python
async def get_server_capabilities_async(
) -> types.GetServerCapabilitiesResponse
```

Async version of get_server_capabilities.

#### `create_lora_training_client`

```python
def create_lora_training_client(
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None) -> TrainingClient
```

Create a TrainingClient for LoRA fine-tuning.

Args:
- `base_model`: Name of the base model to fine-tune (e.g., "Qwen/Qwen2.5-7B")
- `rank`: LoRA rank controlling the size of adaptation matrices (default 32)
- `seed`: Random seed for initialization. None means random seed.
- `train_mlp`: Whether to train MLP layers (default True)
- `train_attn`: Whether to train attention layers (default True)
- `train_unembed`: Whether to train unembedding layers (default True)
- `user_metadata`: Optional metadata to attach to the training run

Returns:
- `TrainingClient` configured for LoRA training

Example:
```python
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen2.5-7B",
    rank=16,
    train_mlp=True,
    train_attn=True
)
# Now use training_client.forward_backward() to train
```

#### `create_lora_training_client_async`

```python
async def create_lora_training_client_async(
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None) -> TrainingClient
```

Async version of create_lora_training_client.

#### `create_training_client_from_state`

```python
def create_training_client_from_state(
        path: str,
        user_metadata: dict[str, str] | None = None) -> TrainingClient
```

Create a TrainingClient from saved model weights.

This loads only the model weights, not optimizer state. To also restore
optimizer state (e.g., Adam momentum), use create_training_client_from_state_with_optimizer.

Args:
- `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")
- `user_metadata`: Optional metadata to attach to the new training run

Returns:
- `TrainingClient` loaded with the specified weights

Example:
```python
# Resume training from a checkpoint (weights only, optimizer resets)
training_client = service_client.create_training_client_from_state(
    "tinker://run-id/weights/checkpoint-001"
)
# Continue training from the loaded state
```

#### `create_training_client_from_state_async`

```python
async def create_training_client_from_state_async(
        path: str,
        user_metadata: dict[str, str] | None = None) -> TrainingClient
```

Async version of create_training_client_from_state.

#### `create_training_client_from_state_with_optimizer`

```python
def create_training_client_from_state_with_optimizer(
        path: str,
        user_metadata: dict[str, str] | None = None) -> TrainingClient
```

Create a TrainingClient from saved model weights and optimizer state.

This is similar to create_training_client_from_state but also restores
optimizer state (e.g., Adam momentum), which is useful for resuming
training exactly where it left off.

Args:
- `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")
- `user_metadata`: Optional metadata to attach to the new training run

Returns:
- `TrainingClient` loaded with the specified weights and optimizer state

Example:
```python
# Resume training from a checkpoint with optimizer state
training_client = service_client.create_training_client_from_state_with_optimizer(
    "tinker://run-id/weights/checkpoint-001"
)
# Continue training with restored optimizer momentum
```

#### `create_training_client_from_state_with_optimizer_async`

```python
async def create_training_client_from_state_with_optimizer_async(
        path: str,
        user_metadata: dict[str, str] | None = None) -> TrainingClient
```

Async version of create_training_client_from_state_with_optimizer.

#### `create_sampling_client`

```python
def create_sampling_client(
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None) -> SamplingClient
```

Create a SamplingClient for text generation.

Args:
- `model_path`: Path to saved model weights (e.g., "tinker://run-id/weights/checkpoint-001")
- `base_model`: Name of base model to use (e.g., "Qwen/Qwen2.5-7B")
- `retry_config`: Optional configuration for retrying failed requests

Returns:
- `SamplingClient` configured for text generation

Raises:
    ValueError: If neither model_path nor base_model is provided

Example:
```python
# Use a base model
sampling_client = service_client.create_sampling_client(
    base_model="Qwen/Qwen2.5-7B"
)

# Or use saved weights
sampling_client = service_client.create_sampling_client(
    model_path="tinker://run-id/weights/checkpoint-001"
)
```

#### `create_sampling_client_async`

```python
async def create_sampling_client_async(
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None) -> SamplingClient
```

Async version of create_sampling_client.

#### `create_rest_client`

```python
def create_rest_client() -> RestClient
```

Create a RestClient for REST API operations.

The RestClient provides access to various REST endpoints for querying
model information, checkpoints, sessions, and managing checkpoint visibility.

Returns:
- `RestClient` for accessing REST API endpoints

Example:
```python
rest_client = service_client.create_rest_client()

# List checkpoints for a training run
checkpoints = rest_client.list_checkpoints("run-id").result()

# Get training run info
training_run = rest_client.get_training_run("run-id").result()

# Publish a checkpoint
rest_client.publish_checkpoint_from_tinker_path(
    "tinker://run-id/weights/checkpoint-001"
).result()
```
