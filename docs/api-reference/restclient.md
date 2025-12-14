RestClient for Tinker API REST operations.

## `RestClient` Objects

```python
class RestClient(TelemetryProvider)
```

Client for REST API operations like listing checkpoints and metadata.

The RestClient provides access to various REST endpoints for querying
model information, checkpoints, and other resources. You typically get one
by calling `service_client.create_rest_client()`.

Key methods:
- list_checkpoints() - list available model checkpoints (both training and sampler)
- list_user_checkpoints() - list all checkpoints across all user's training runs
- get_training_run() - get model information and metadata as ModelEntry
- delete_checkpoint() - delete an existing checkpoint for a training run
- get_checkpoint_archive_url() - get signed URL to download checkpoint archive
- publish_checkpoint_from_tinker_path() - publish a checkpoint to make it public
- unpublish_checkpoint_from_tinker_path() - unpublish a checkpoint to make it private

Args:
- `holder`: Internal client managing HTTP connections and async operations

Example:
```python
rest_client = service_client.create_rest_client()
training_run = rest_client.get_training_run("run-id").result()
print(f"Training Run: {training_run.training_run_id}, LoRA: {training_run.is_lora}")
checkpoints = rest_client.list_checkpoints("run-id").result()
print(f"Found {len(checkpoints.checkpoints)} checkpoints")
for checkpoint in checkpoints.checkpoints:
    print(f"  {checkpoint.checkpoint_type}: {checkpoint.checkpoint_id}")
```

#### `get_training_run`

```python
def get_training_run(
        training_run_id: types.ModelID) -> ConcurrentFuture[types.TrainingRun]
```

Get training run info.

Args:
- `training_run_id`: The training run ID to get information for

Returns:
- A `Future` containing the training run information

Example:
```python
future = rest_client.get_training_run("run-id")
response = future.result()
print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
```

#### `get_training_run_async`

```python
async def get_training_run_async(
        training_run_id: types.ModelID) -> types.TrainingRun
```

Async version of get_training_run.

#### `get_training_run_by_tinker_path`

```python
def get_training_run_by_tinker_path(
        tinker_path: str) -> ConcurrentFuture[types.TrainingRun]
```

Get training run info.

Args:
- `tinker_path`: The tinker path to the checkpoint

Returns:
- A `Future` containing the training run information

Example:
```python
future = rest_client.get_training_run_by_tinker_path("tinker://run-id/weights/checkpoint-001")
response = future.result()
print(f"Training Run ID: {response.training_run_id}, Base: {response.base_model}")
```

#### `get_training_run_by_tinker_path_async`

```python
async def get_training_run_by_tinker_path_async(
        tinker_path: str) -> types.TrainingRun
```

Async version of get_training_run_by_tinker_path.

#### `get_weights_info_by_tinker_path`

```python
def get_weights_info_by_tinker_path(
        tinker_path: str) -> APIFuture[types.WeightsInfoResponse]
```

Get checkpoint information from a tinker path.

Args:
- `tinker_path`: The tinker path to the checkpoint

Returns:
- An `APIFuture` containing the checkpoint information. The future is awaitable.

Example:
```python
future = rest_client.get_weights_info_by_tinker_path("tinker://run-id/weights/checkpoint-001")
response = future.result()  # or await future
print(f"Base Model: {response.base_model}, LoRA Rank: {response.lora_rank}")
```

#### `list_training_runs`

```python
def list_training_runs(
        limit: int = 20,
        offset: int = 0) -> ConcurrentFuture[types.TrainingRunsResponse]
```

List training runs with pagination support.

Args:
- `limit`: Maximum number of training runs to return (default 20)
- `offset`: Offset for pagination (default 0)

Returns:
- A `Future` containing the `TrainingRunsResponse` with training runs and cursor info

Example:
```python
future = rest_client.list_training_runs(limit=50)
response = future.result()
print(f"Found {len(response.training_runs)} training runs")
print(f"Total: {response.cursor.total_count}")
# Get next page
next_page = rest_client.list_training_runs(limit=50, offset=50)
```

#### `list_training_runs_async`

```python
async def list_training_runs_async(limit: int = 20,
                                   offset: int = 0
                                   ) -> types.TrainingRunsResponse
```

Async version of list_training_runs.

#### `list_checkpoints`

```python
def list_checkpoints(
    training_run_id: types.ModelID
) -> ConcurrentFuture[types.CheckpointsListResponse]
```

List available checkpoints (both training and sampler).

Args:
- `training_run_id`: The training run ID to list checkpoints for

Returns:
- A `Future` containing the `CheckpointsListResponse` with available checkpoints

Example:
```python
future = rest_client.list_checkpoints("run-id")
response = future.result()
for checkpoint in response.checkpoints:
    if checkpoint.checkpoint_type == "training":
        print(f"Training checkpoint: {checkpoint.checkpoint_id}")
    elif checkpoint.checkpoint_type == "sampler":
        print(f"Sampler checkpoint: {checkpoint.checkpoint_id}")
```

#### `list_checkpoints_async`

```python
async def list_checkpoints_async(
        training_run_id: types.ModelID) -> types.CheckpointsListResponse
```

Async version of list_checkpoints.

#### `get_checkpoint_archive_url`

```python
def get_checkpoint_archive_url(
    training_run_id: types.ModelID, checkpoint_id: str
) -> ConcurrentFuture[types.CheckpointArchiveUrlResponse]
```

Get signed URL to download checkpoint archive.

Args:
- `training_run_id`: The training run ID to download weights for
- `checkpoint_id`: The checkpoint ID to download

Returns:
- A `Future` containing the `CheckpointArchiveUrlResponse` with signed URL and expiration

Example:
```python
future = rest_client.get_checkpoint_archive_url("run-id", "checkpoint-123")
response = future.result()
print(f"Download URL: {response.url}")
print(f"Expires at: {response.expires_at}")
# Use the URL to download the archive with your preferred HTTP client
```

#### `get_checkpoint_archive_url_async`

```python
async def get_checkpoint_archive_url_async(
        training_run_id: types.ModelID,
        checkpoint_id: str) -> types.CheckpointArchiveUrlResponse
```

Async version of get_checkpoint_archive_url.

#### `delete_checkpoint`

```python
def delete_checkpoint(training_run_id: types.ModelID,
                      checkpoint_id: str) -> ConcurrentFuture[None]
```

Delete a checkpoint for a training run.

#### `delete_checkpoint_async`

```python
async def delete_checkpoint_async(training_run_id: types.ModelID,
                                  checkpoint_id: str) -> None
```

Async version of delete_checkpoint.

#### `delete_checkpoint_from_tinker_path`

```python
def delete_checkpoint_from_tinker_path(
        tinker_path: str) -> ConcurrentFuture[None]
```

Delete a checkpoint referenced by a tinker path.

#### `delete_checkpoint_from_tinker_path_async`

```python
async def delete_checkpoint_from_tinker_path_async(tinker_path: str) -> None
```

Async version of delete_checkpoint_from_tinker_path.

#### `get_checkpoint_archive_url_from_tinker_path`

```python
def get_checkpoint_archive_url_from_tinker_path(
        tinker_path: str
) -> ConcurrentFuture[types.CheckpointArchiveUrlResponse]
```

Get signed URL to download checkpoint archive.

Args:
- `tinker_path`: The tinker path to the checkpoint

Returns:
- A `Future` containing the `CheckpointArchiveUrlResponse` with signed URL and expiration

#### `get_checkpoint_archive_url_from_tinker_path_async`

```python
async def get_checkpoint_archive_url_from_tinker_path_async(
        tinker_path: str) -> types.CheckpointArchiveUrlResponse
```

Async version of get_checkpoint_archive_url_from_tinker_path.

#### `publish_checkpoint_from_tinker_path`

```python
def publish_checkpoint_from_tinker_path(
        tinker_path: str) -> ConcurrentFuture[None]
```

Publish a checkpoint referenced by a tinker path to make it publicly accessible.

Only the exact owner of the training run can publish checkpoints.
Published checkpoints can be unpublished using the unpublish_checkpoint_from_tinker_path method.

Args:
- `tinker_path`: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")

Returns:
- A `Future` that completes when the checkpoint is published

Raises:
    HTTPException: 400 if checkpoint identifier is invalid
    HTTPException: 404 if checkpoint not found or user doesn't own the training run
    HTTPException: 409 if checkpoint is already public
    HTTPException: 500 if there's an error publishing the checkpoint

Example:
```python
future = rest_client.publish_checkpoint_from_tinker_path("tinker://run-id/weights/0001")
future.result()  # Wait for completion
print("Checkpoint published successfully")
```

#### `publish_checkpoint_from_tinker_path_async`

```python
async def publish_checkpoint_from_tinker_path_async(tinker_path: str) -> None
```

Async version of publish_checkpoint_from_tinker_path.

#### `unpublish_checkpoint_from_tinker_path`

```python
def unpublish_checkpoint_from_tinker_path(
        tinker_path: str) -> ConcurrentFuture[None]
```

Unpublish a checkpoint referenced by a tinker path to make it private again.

Only the exact owner of the training run can unpublish checkpoints.
This reverses the effect of publishing a checkpoint.

Args:
- `tinker_path`: The tinker path to the checkpoint (e.g., "tinker://run-id/weights/0001")

Returns:
- A `Future` that completes when the checkpoint is unpublished

Raises:
    HTTPException: 400 if checkpoint identifier is invalid
    HTTPException: 404 if checkpoint not found or user doesn't own the training run
    HTTPException: 409 if checkpoint is already private
    HTTPException: 500 if there's an error unpublishing the checkpoint

Example:
```python
future = rest_client.unpublish_checkpoint_from_tinker_path("tinker://run-id/weights/0001")
future.result()  # Wait for completion
print("Checkpoint unpublished successfully")
```

#### `unpublish_checkpoint_from_tinker_path_async`

```python
async def unpublish_checkpoint_from_tinker_path_async(
        tinker_path: str) -> None
```

Async version of unpublish_checkpoint_from_tinker_path.

#### `list_user_checkpoints`

```python
def list_user_checkpoints(
        limit: int = 100,
        offset: int = 0) -> ConcurrentFuture[types.CheckpointsListResponse]
```

List all checkpoints for the current user across all their training runs.

This method retrieves checkpoints from all training runs owned by the authenticated user,
sorted by time (newest first). It supports pagination for efficiently handling large
numbers of checkpoints.

Args:
- `limit`: Maximum number of checkpoints to return (default 100)
- `offset`: Offset for pagination (default 0)

Returns:
- A `Future` containing the `CheckpointsListResponse` with checkpoints and cursor info

Example:
```python
future = rest_client.list_user_checkpoints(limit=50)
response = future.result()
print(f"Found {len(response.checkpoints)} checkpoints")
print(f"Total: {response.cursor.total_count if response.cursor else 'Unknown'}")
for checkpoint in response.checkpoints:
    print(f"  {checkpoint.training_run_id}/{checkpoint.checkpoint_id}")
# Get next page if there are more checkpoints
if response.cursor and response.cursor.offset + response.cursor.limit < response.cursor.total_count:
    next_page = rest_client.list_user_checkpoints(limit=50, offset=50)
```

#### `list_user_checkpoints_async`

```python
async def list_user_checkpoints_async(limit: int = 100,
                                      offset: int = 0
                                      ) -> types.CheckpointsListResponse
```

Async version of list_user_checkpoints.

#### `get_session`

```python
def get_session(session_id: str) -> ConcurrentFuture[types.GetSessionResponse]
```

Get session information including all training runs and samplers.

Args:
- `session_id`: The session ID to get information for

Returns:
- A `Future` containing the `GetSessionResponse` with training_run_ids and sampler_ids

Example:
```python
future = rest_client.get_session("session-id")
response = future.result()
print(f"Training runs: {len(response.training_run_ids)}")
print(f"Samplers: {len(response.sampler_ids)}")
```

#### `get_session_async`

```python
async def get_session_async(session_id: str) -> types.GetSessionResponse
```

Async version of get_session.

#### `list_sessions`

```python
def list_sessions(
        limit: int = 20,
        offset: int = 0) -> ConcurrentFuture[types.ListSessionsResponse]
```

List sessions with pagination support.

Args:
- `limit`: Maximum number of sessions to return (default 20)
- `offset`: Offset for pagination (default 0)

Returns:
- A `Future` containing the `ListSessionsResponse` with list of session IDs

Example:
```python
future = rest_client.list_sessions(limit=50)
response = future.result()
print(f"Found {len(response.sessions)} sessions")
# Get next page
next_page = rest_client.list_sessions(limit=50, offset=50)
```

#### `list_sessions_async`

```python
async def list_sessions_async(limit: int = 20,
                              offset: int = 0) -> types.ListSessionsResponse
```

Async version of list_sessions.

#### `get_sampler`

```python
def get_sampler(sampler_id: str) -> APIFuture[types.GetSamplerResponse]
```

Get sampler information.

Args:
- `sampler_id`: The sampler ID (sampling_session_id) to get information for

Returns:
- An `APIFuture` containing the `GetSamplerResponse` with sampler details

Example:
```python
# Sync usage
future = rest_client.get_sampler("session-id:sample:0")
response = future.result()
print(f"Base model: {response.base_model}")
print(f"Model path: {response.model_path}")

# Async usage
response = await rest_client.get_sampler("session-id:sample:0")
print(f"Base model: {response.base_model}")
```

#### `get_sampler_async`

```python
async def get_sampler_async(sampler_id: str) -> types.GetSamplerResponse
```

Async version of get_sampler.
