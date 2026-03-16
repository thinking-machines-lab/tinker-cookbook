API Future classes for handling async operations with retry logic.

## `APIFuture` Objects

```python
class APIFuture(ABC, Generic[T])
```

Abstract base class for futures that can be awaited or accessed synchronously.

APIFuture provides a unified interface for handling async operations that can
be accessed both synchronously (via result()) and asynchronously (via await or result_async()).
This allows for flexible usage patterns in both sync and async contexts.

The future can be awaited directly in async contexts:
```python
result = await api_future  # Equivalent to await api_future.result_async()
```

Or accessed synchronously:
```python
result = api_future.result()  # Blocks until complete
```

Args:
- `T`: The type of the result value

Example:
```python
# In async context
future = training_client.forward_backward(data, "cross_entropy")
result = await future  # Or await future.result_async()

# In sync context
future = training_client.forward_backward(data, "cross_entropy")
result = future.result()
```

#### `result_async`

```python
async def result_async(timeout: float | None = None) -> T
```

Get the result asynchronously with optional timeout.

Args:
- `timeout`: Maximum time to wait in seconds. None means wait indefinitely.

Returns:
- The result value of type `T`

Raises:
    TimeoutError: If timeout is exceeded

#### `result`

```python
def result(timeout: float | None = None) -> T
```

Get the result synchronously with optional timeout.

Args:
- `timeout`: Maximum time to wait in seconds. None means wait indefinitely.

Returns:
- The result value of type `T`

Raises:
    TimeoutError: If timeout is exceeded

## `AwaitableConcurrentFuture` Objects

```python
class AwaitableConcurrentFuture(APIFuture[T])
```

Implementation of APIFuture that wraps a concurrent.futures.Future.

This class bridges Python's concurrent.futures with asyncio, allowing a
standard Future to be used in async contexts. It's commonly returned by
Tinker API methods to provide both sync and async access patterns.

Args:
- `future`: A concurrent.futures.Future to wrap

Example:
```python
# Internal usage - typically you receive these from API methods
concurrent_future = some_operation()
api_future = AwaitableConcurrentFuture(concurrent_future)

# Can be used synchronously
result = api_future.result()

# Or asynchronously
result = await api_future
```

#### `result`

```python
def result(timeout: float | None = None) -> T
```

Get the result synchronously with optional timeout.

Args:
- `timeout`: Maximum time to wait in seconds. None means wait indefinitely.

Returns:
- The result value of type `T`

Raises:
    TimeoutError: If timeout is exceeded
    Exception: Any exception raised by the underlying operation

Example:
```python
future = rest_client.get_training_run("run-id")
result = future.result(timeout=30)  # Wait up to 30 seconds
```

#### `result_async`

```python
async def result_async(timeout: float | None = None) -> T
```

Async version of result.

#### `future`

```python
def future() -> ConcurrentFuture[T]
```

Get the underlying concurrent.futures.Future.

Returns:
- The wrapped `ConcurrentFuture` object

Example:
```python
api_future = rest_client.get_training_run("run-id")
concurrent_future = api_future.future()
# Can now use standard concurrent.futures methods
if concurrent_future.done():
    result = concurrent_future.result()
```
