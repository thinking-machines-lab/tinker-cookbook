# Error Reference

Extended decoder for Tinker SDK and cookbook error messages. Organized by where the error originates.

## Tinker SDK errors (from the service)

### HTTP 400 — Bad Request

These are validation errors. The request was malformed or had invalid parameters.

| Error detail | Cause | Fix |
|-------------|-------|-----|
| `base_model is required` | Missing model name in `create_lora_training_client` | Pass `base_model="org/model-name"` |
| `lora_config.rank must be positive` | LoRA rank <= 0 | Use a positive rank (typically 32) |
| `lora_config.rank must be a power of 2` | Rank like 48 or 100 | Use 16, 32, 64, etc. |
| `At least one of train_unembed, train_mlp, or train_attn must be True` | All training flags disabled | Enable at least one (defaults are all True) |
| `Prompt length plus max_tokens exceeds the model's context window: {N} + {M} > {limit}` | Input too long | Reduce prompt length or max_tokens. Error shows exact numbers. |
| `session_id is required. The version of the Tinker SDK you are using is no longer supported.` | SDK too old | `pip install --upgrade tinker` |

### HTTP 402 — Payment Required

Billing account is blocked. Top up at the Tinker console billing page.

### HTTP 403 — Forbidden

| Error detail | Cause | Fix |
|-------------|-------|-----|
| `You do not have access to this model` | No permission for the model or checkpoint | Check API key; request access to the model/org |
| `Invalid session_id` | Session doesn't exist or belongs to another user | Check session ID; create a new session |
| Generic 403 on checkpoint | Checkpoint is private and user isn't the owner | Request that the checkpoint owner make it public, or get project/org access |

### HTTP 404 — Not Found

Model, session, or checkpoint doesn't exist. Double-check:
- Model name spelling (e.g., `Qwen/Qwen3-8B` not `qwen3-8b`)
- Session ID (use `tc.get_info()` to verify)
- Checkpoint path format (`tinker://session-id/sampler_weights/name`)
- That you used `save_weights_for_sampler` (not `save_state`) for sampling/download

### HTTP 409 — Conflict

Resource already exists. Most commonly:
- **Checkpoint save retry**: The first save actually succeeded (network hiccup made it look like it failed). Check if the checkpoint exists before retrying.
- **Session conflict**: Session already exists with different metadata. Use a different session ID.

### HTTP 429 — Rate Limited

Too many concurrent requests. Default limits:
- ~2000 concurrent sampling requests per user
- Tightens under high service load

Fix: Add backoff/retry logic. Reduce concurrency (smaller `num_samples`, fewer parallel `asyncio.gather` calls).

### HTTP 500 — Internal Server Error

Unhandled server-side error. Gather:
1. Session ID (`tc.get_info()`)
2. What operation failed
3. Timestamp
4. Tinker SDK version

Report to the Tinker team.

### HTTP 504 — Gateway Timeout

`Timeout while publishing request` — the server's internal queue is full or slow. Usually transient. Retry after a few seconds.

## Tinker SDK Python exceptions

| Exception | HTTP code | Retryable? | Notes |
|-----------|----------|-----------|-------|
| `tinker.AuthenticationError` | 401/403 | No | Check API key |
| `tinker.BadRequestError` | 400 | No | Fix the request |
| `tinker.RateLimitError` | 429 | Yes | Back off and retry |
| `tinker.APITimeoutError` | timeout | Yes | Increase timeout or retry |
| `tinker.APIConnectionError` | network | Yes | Check connectivity |
| `tinker.TinkerError` | varies | Maybe | Catch-all base class |

### Request-level error codes

These appear in `ForwardBackwardResult` or sampling responses:

| Code | Meaning | Action |
|------|---------|--------|
| `NonFiniteInTensor` | NaN or Inf in input data | Check your data for invalid values; ensure loss weights don't produce Inf |
| `PromptTooLong` | Prompt exceeds context window | Reduce input length |
| `ResourceAlreadyExists` | Duplicate resource | Handle idempotently |

## Cookbook exceptions

### Configuration errors

| Exception | Common triggers |
|-----------|----------------|
| `ConfigurationError("Unknown model: {name}")` | Model name not in cookbook registry. Check spelling; use `model_info.get_model_attributes()`. |
| `ConfigurationError("Log directory already exists")` | Resume from different path, or delete the old log directory |

### Data errors

| Exception | Common triggers |
|-----------|----------------|
| `DataFormatError("Each line must contain a 'messages' field")` | JSONL file lines missing `messages` key |
| `DataValidationError("Cannot seek backward")` | Streaming dataset used with random access |
| `ValueError("tokens and weights must be the same length")` | `max_length` truncated tokens but not weights. Use `datum_from_model_input_weights()` which handles this. |

### Renderer errors

| Exception | Common triggers |
|-----------|----------------|
| `RendererError("Unknown renderer")` | Invalid renderer name. Use `get_recommended_renderer_name()`. |
| `RendererError("requires an image_processor")` | VL renderer created without image processor |
| `RendererError("Expected text content, got multimodal content")` | Passed image content to a text-only renderer |

### Weight errors

| Exception | Common triggers |
|-----------|----------------|
| `WeightsDownloadError` | Invalid tinker:// path, or checkpoint doesn't exist. Verify with `tinker checkpoint list`. |
| `WeightsMergeError` | Adapter incompatible with base model. Check model name matches exactly. |
| `WeightsAdapterError` | Can't convert to PEFT format. Check for empty expert tensors (known issue with some models). |

### Training errors

| Exception | Common triggers |
|-----------|----------------|
| `CheckpointError` | Checkpoint save/load failed. Check path format and permissions. |
| `AllTrajectoriesFailedError` | Every trajectory in a rollout group failed. Check environment code; look at rollout logs. |

## Upstream library errors

| Error | Library | Cause | Fix |
|-------|---------|-------|-----|
| `Expected N tokens, got M from image` | `transformers` < 5.0 | Bug in `Qwen2VLImageProcessor` | Upgrade: `pip install 'transformers>=5.0'` |
| DeepSeek tokenizer loading fails | `transformers` == 5.3.0 | Incorrect `tokenizer_class` on hub (huggingface/transformers#44801) | Upgrade to `transformers>=5.3.1` |
| `ModuleNotFoundError: pkg_resources` | Python 3.14 | `pkg_resources` removed from stdlib | Downgrade Python or update the offending package |
| `401 Unauthorized` on tokenizer download | HuggingFace | Gated model (Llama) needs auth | Set `HF_TOKEN` environment variable |
| Corrupted tokenizer cache | HuggingFace | Cache corruption | Delete `~/.cache/huggingface/hub/models--{org}--{model}/` and retry |
