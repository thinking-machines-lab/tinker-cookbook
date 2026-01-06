## `TinkerError` Objects

```python
class TinkerError(Exception)
```

Base exception for all Tinker-related errors.

## `APIError` Objects

```python
class APIError(TinkerError)
```

Base class for all API-related errors.

#### `body`

The API response body.

If the API responded with a valid JSON structure then this property will be the
decoded result.

If it isn't a valid JSON structure then this will be the raw response.

If there was no response associated with this error then it will be `None`.

## `APIResponseValidationError` Objects

```python
class APIResponseValidationError(APIError)
```

Raised when API response doesn't match expected schema.

## `APIStatusError` Objects

```python
class APIStatusError(APIError)
```

Raised when an API response has a status code of 4xx or 5xx.

## `APIConnectionError` Objects

```python
class APIConnectionError(APIError)
```

Raised when a connection error occurs while making an API request.

## `APITimeoutError` Objects

```python
class APITimeoutError(APIConnectionError)
```

Raised when an API request times out.

## `BadRequestError` Objects

```python
class BadRequestError(APIStatusError)
```

HTTP 400: The request was invalid or malformed.

## `AuthenticationError` Objects

```python
class AuthenticationError(APIStatusError)
```

HTTP 401: Authentication credentials are missing or invalid.

## `PermissionDeniedError` Objects

```python
class PermissionDeniedError(APIStatusError)
```

HTTP 403: Insufficient permissions to access the resource.

## `NotFoundError` Objects

```python
class NotFoundError(APIStatusError)
```

HTTP 404: The requested resource was not found.

## `ConflictError` Objects

```python
class ConflictError(APIStatusError)
```

HTTP 409: The request conflicts with the current state of the resource.

## `UnprocessableEntityError` Objects

```python
class UnprocessableEntityError(APIStatusError)
```

HTTP 422: The request was well-formed but contains semantic errors.

## `RateLimitError` Objects

```python
class RateLimitError(APIStatusError)
```

HTTP 429: Too many requests, rate limit exceeded.

## `InternalServerError` Objects

```python
class InternalServerError(APIStatusError)
```

HTTP 500+: An error occurred on the server.

## `RequestFailedError` Objects

```python
class RequestFailedError(TinkerError)
```

Raised when an asynchronous request completes in a failed state.
