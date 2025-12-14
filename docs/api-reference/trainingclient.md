TrainingClient for Tinker API.

## `TrainingClient` Objects

```python
class TrainingClient(TelemetryProvider, QueueStateObserver)
```

Client for training ML models with forward/backward passes and optimization.

The TrainingClient corresponds to a fine-tuned model that you can train and sample from.
You typically get one by calling `service_client.create_lora_training_client()`.
Key methods:
- forward_backward() - compute gradients for training
- optim_step() - update model parameters with Adam optimizer
- save_weights_and_get_sampling_client() - export trained model for inference

Args:
- `holder`: Internal client managing HTTP connections and async operations
- `model_id`: Unique identifier for the model to train. Required for training operations.

Example:
```python
training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen2.5-7B")
fwdbwd_future = training_client.forward_backward(training_data, "cross_entropy")
optim_future = training_client.optim_step(types.AdamParams(learning_rate=1e-4))
fwdbwd_result = fwdbwd_future.result()  # Wait for gradients
optim_result = optim_future.result()    # Wait for parameter update
sampling_client = training_client.save_weights_and_get_sampling_client("my-model")
```

#### `forward`

```python
def forward(
    data: List[types.Datum],
    loss_fn: types.LossFnType,
    loss_fn_config: Dict[str, float] | None = None
) -> APIFuture[types.ForwardBackwardOutput]
```

Compute forward pass without gradients.

Args:
- `data`: List of training data samples
- `loss_fn`: Loss function type (e.g., "cross_entropy")
- `loss_fn_config`: Optional configuration for the loss function

Returns:
- `APIFuture` containing the forward pass outputs and loss

Example:
```python
data = [types.Datum(
    model_input=types.ModelInput.from_ints(tokenizer.encode("Hello")),
    loss_fn_inputs={"target_tokens": types.ModelInput.from_ints(tokenizer.encode("world"))}
)]
future = training_client.forward(data, "cross_entropy")
result = await future
print(f"Loss: {result.loss}")
```

#### `forward_async`

```python
async def forward_async(
    data: List[types.Datum],
    loss_fn: types.LossFnType,
    loss_fn_config: Dict[str, float] | None = None
) -> APIFuture[types.ForwardBackwardOutput]
```

Async version of forward.

#### `forward_backward`

```python
def forward_backward(
    data: List[types.Datum],
    loss_fn: types.LossFnType,
    loss_fn_config: Dict[str, float] | None = None
) -> APIFuture[types.ForwardBackwardOutput]
```

Compute forward pass and backward pass to calculate gradients.

Args:
- `data`: List of training data samples
- `loss_fn`: Loss function type (e.g., "cross_entropy")
- `loss_fn_config`: Optional configuration for the loss function

Returns:
- `APIFuture` containing the forward/backward outputs, loss, and gradients

Example:
```python
data = [types.Datum(
    model_input=types.ModelInput.from_ints(tokenizer.encode("Hello")),
    loss_fn_inputs={"target_tokens": types.ModelInput.from_ints(tokenizer.encode("world"))}
)]

# Compute gradients
fwdbwd_future = training_client.forward_backward(data, "cross_entropy")

# Update parameters
optim_future = training_client.optim_step(
    types.AdamParams(learning_rate=1e-4)
)

fwdbwd_result = await fwdbwd_future
print(f"Loss: {fwdbwd_result.loss}")
```

#### `forward_backward_async`

```python
async def forward_backward_async(
    data: List[types.Datum],
    loss_fn: types.LossFnType,
    loss_fn_config: Dict[str, float] | None = None
) -> APIFuture[types.ForwardBackwardOutput]
```

Async version of forward_backward.

#### `forward_backward_custom`

```python
def forward_backward_custom(
        data: List[types.Datum],
        loss_fn: CustomLossFnV1) -> APIFuture[types.ForwardBackwardOutput]
```

Compute forward/backward with a custom loss function.

Allows you to define custom loss functions that operate on log probabilities.
The custom function receives logprobs and computes loss and gradients.

Args:
- `data`: List of training data samples
- `loss_fn`: Custom loss function that takes (data, logprobs) and returns (loss, metrics)

Returns:
- `APIFuture` containing the forward/backward outputs with custom loss

Example:
```python
def custom_loss(data, logprobs_list):
    # Custom loss computation
    loss = torch.mean(torch.stack([torch.mean(lp) for lp in logprobs_list]))
    metrics = {"custom_metric": loss.item()}
    return loss, metrics

future = training_client.forward_backward_custom(data, custom_loss)
result = future.result()
print(f"Custom loss: {result.loss}")
print(f"Metrics: {result.metrics}")
```

#### `forward_backward_custom_async`

```python
async def forward_backward_custom_async(
        data: List[types.Datum],
        loss_fn: CustomLossFnV1) -> APIFuture[types.ForwardBackwardOutput]
```

Async version of forward_backward_custom.

#### `optim_step`

```python
def optim_step(
        adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]
```

Update model parameters using Adam optimizer.

Args:
- `adam_params`: Adam optimizer parameters (learning_rate, betas, eps, weight_decay)

Returns:
- `APIFuture` containing optimizer step response

Example:
```python
# First compute gradients
fwdbwd_future = training_client.forward_backward(data, "cross_entropy")

# Then update parameters
optim_future = training_client.optim_step(
    types.AdamParams(
        learning_rate=1e-4,
        weight_decay=0.01
    )
)

# Wait for both to complete
fwdbwd_result = await fwdbwd_future
optim_result = await optim_future
```

#### `optim_step_async`

```python
async def optim_step_async(
        adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]
```

Async version of optim_step.

#### `save_state`

```python
def save_state(name: str) -> APIFuture[types.SaveWeightsResponse]
```

Save model weights to persistent storage.

Args:
- `name`: Name for the saved checkpoint

Returns:
- `APIFuture` containing the save response with checkpoint path

Example:
```python
# Save after training
save_future = training_client.save_state("checkpoint-001")
result = await save_future
print(f"Saved to: {result.path}")
```

#### `save_state_async`

```python
async def save_state_async(name: str) -> APIFuture[types.SaveWeightsResponse]
```

Async version of save_state.

#### `load_state`

```python
def load_state(path: str) -> APIFuture[types.LoadWeightsResponse]
```

Load model weights from a saved checkpoint.

This loads only the model weights, not optimizer state (e.g., Adam momentum).
To also restore optimizer state, use load_state_with_optimizer.

Args:
- `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")

Returns:
- `APIFuture` containing the load response

Example:
```python
# Load checkpoint to continue training (weights only, optimizer resets)
load_future = training_client.load_state("tinker://run-id/weights/checkpoint-001")
await load_future
# Continue training from loaded state
```

#### `load_state_async`

```python
async def load_state_async(path: str) -> APIFuture[types.LoadWeightsResponse]
```

Async version of load_state.

#### `load_state_with_optimizer`

```python
def load_state_with_optimizer(
        path: str) -> APIFuture[types.LoadWeightsResponse]
```

Load model weights and optimizer state from a checkpoint.

Args:
- `path`: Tinker path to saved weights (e.g., "tinker://run-id/weights/checkpoint-001")

Returns:
- `APIFuture` containing the load response

Example:
```python
# Resume training with optimizer state
load_future = training_client.load_state_with_optimizer(
    "tinker://run-id/weights/checkpoint-001"
)
await load_future
# Continue training with restored optimizer momentum
```

#### `load_state_with_optimizer_async`

```python
async def load_state_with_optimizer_async(
        path: str) -> APIFuture[types.LoadWeightsResponse]
```

Async version of load_state_with_optimizer.

#### `save_weights_for_sampler`

```python
def save_weights_for_sampler(
        name: str) -> APIFuture[types.SaveWeightsForSamplerResponse]
```

Save model weights for use with a SamplingClient.

Args:
- `name`: Name for the saved sampler weights

Returns:
- `APIFuture` containing the save response with sampler path

Example:
```python
# Save weights for inference
save_future = training_client.save_weights_for_sampler("sampler-001")
result = await save_future
print(f"Sampler weights saved to: {result.path}")

# Use the path to create a sampling client
sampling_client = service_client.create_sampling_client(
    model_path=result.path
)
```

#### `save_weights_for_sampler_async`

```python
async def save_weights_for_sampler_async(
        name: str) -> APIFuture[types.SaveWeightsForSamplerResponse]
```

Async version of save_weights_for_sampler.

#### `get_info`

```python
def get_info() -> types.GetInfoResponse
```

Get information about the current model.

Returns:
- `GetInfoResponse` with model configuration and metadata

Example:
```python
info = training_client.get_info()
print(f"Model ID: {info.model_data.model_id}")
print(f"Base model: {info.model_data.model_name}")
print(f"LoRA rank: {info.model_data.lora_rank}")
```

#### `get_info_async`

```python
async def get_info_async() -> types.GetInfoResponse
```

Async version of get_info.

#### `get_tokenizer`

```python
def get_tokenizer() -> PreTrainedTokenizer
```

Get the tokenizer for the current model.

Returns:
- `PreTrainedTokenizer` compatible with the model

Example:
```python
tokenizer = training_client.get_tokenizer()
tokens = tokenizer.encode("Hello world")
text = tokenizer.decode(tokens)
```

#### `create_sampling_client`

```python
def create_sampling_client(
        model_path: str,
        retry_config: RetryConfig | None = None) -> SamplingClient
```

Create a SamplingClient from saved weights.

Args:
- `model_path`: Tinker path to saved weights
- `retry_config`: Optional configuration for retrying failed requests

Returns:
- `SamplingClient` configured with the specified weights

Example:
```python
sampling_client = training_client.create_sampling_client(
    "tinker://run-id/weights/checkpoint-001"
)
# Use sampling_client for inference
```

#### `create_sampling_client_async`

```python
async def create_sampling_client_async(
        model_path: str,
        retry_config: RetryConfig | None = None) -> SamplingClient
```

Async version of create_sampling_client.

#### `save_weights_and_get_sampling_client`

```python
def save_weights_and_get_sampling_client(
        name: str | None = None,
        retry_config: RetryConfig | None = None) -> SamplingClient
```

Save current weights and create a SamplingClient for inference.

Args:
- `name`: Optional name for the saved weights (currently ignored for ephemeral saves)
- `retry_config`: Optional configuration for retrying failed requests

Returns:
- `SamplingClient` configured with the current model weights

Example:
```python
# After training, create a sampling client directly
sampling_client = training_client.save_weights_and_get_sampling_client()

# Now use it for inference
prompt = types.ModelInput.from_ints(tokenizer.encode("Hello"))
params = types.SamplingParams(max_tokens=20)
result = sampling_client.sample(prompt, 1, params).result()
```

#### `save_weights_and_get_sampling_client_async`

```python
async def save_weights_and_get_sampling_client_async(
        name: str | None = None,
        retry_config: RetryConfig | None = None) -> SamplingClient
```

Async version of save_weights_and_get_sampling_client.
