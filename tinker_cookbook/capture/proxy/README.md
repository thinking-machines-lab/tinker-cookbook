# capture proxy

Anthropic- and OpenAI-compatible chat API proxy for capturing traffic from
black-box agent harnesses (Claude Code, opencode, remote rollout processors)
that cannot use the Tinker SDK or enter `capture(...)` scopes themselves.

## How it fits the pipeline

The capture pipeline (see `../README.md`) tags SDK calls with an ambient
scope and exports them to a local store. This proxy extends that to tools
you cannot modify: they speak a chat API to the proxy, the proxy renders the
chat messages to tokens with a cookbook renderer, samples through
`SamplingClient.sample_async`, and decodes the sampled tokens back into a
chat response. Because the sampling goes through the instrumented SDK,
**everything funnels through the one in-process capture path**: this module
has no export logic of its own; it only enters a per-request `capture(...)`
scope built from the URL path, so rows are born addressed.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST [/r/<key>/<value>/...]/v1/messages` | Anthropic Messages API subset |
| `POST [/r/<key>/<value>/...]/v1/chat/completions` | OpenAI chat completions subset |
| `GET /healthz` | liveness + configured model |

The optional `/r/...` prefix is the **capture address**: `key/value` path
pairs mapped onto scope keys (`run` -> `run_id`, `attempt` -> `run_attempt`,
`split`, `iter` -> `iteration`, `group` -> `group_idx`, `traj` -> `traj_idx`,
`purpose`; unknown keys are preserved verbatim). Bare paths work too and
capture with an empty scope.

### Supported request subset

Chat with tool calling: `messages` with `user`/`assistant` roles (`system`
role for OpenAI, `system` field for Anthropic; `tool` role for OpenAI
results), string content or `text`/`tool_use`/`tool_result` blocks, `tools`,
`tool_choice` auto/none (`none` is enforced on BOTH sides: the catalog is
not rendered into the prompt, and any tool-call syntax the model still
emits is not returned as tool calls; the response is plain text with
`end_turn`/`stop`, carrying the renderer-parsed text content, with the
extracted tool markup dropped rather than reconstructed),
`max_tokens`/`max_completion_tokens`,
`temperature`, `top_p`, `top_k`, `stop_sequences`/`stop` (composed with the
renderer's stop strings; rejected with a 400 when the configured renderer
uses token-id stop conditions, since the sampling API cannot mix the two),
`seed` (OpenAI), `stream`.

**Tools** ride the renderer's own machinery, so agent harnesses like Claude
Code and opencode work end-to-end: the catalog is rendered into the prompt
via `create_conversation_prefix_with_tools` (the system text rides inside
it), `tool_use`/`tool_result` blocks (Anthropic) and `tool_calls`/`tool`
messages (OpenAI) are converted to the renderer's `Message` forms, and tool
calls parsed from the completion come back as `tool_use` blocks with
`stop_reason: "tool_use"` (OpenAI: `tool_calls` with `finish_reason:
"tool_calls"`). Tool support is renderer-dependent: a renderer without tool
calling returns a clean 400 for requests carrying `tools`. Forced
`tool_choice` is rejected (renderers have no forcing mechanism), and a tool
call whose arguments fail to parse as a JSON object is surfaced verbatim as
`{"_raw_arguments": ...}` rather than silently repaired. The `model` field is recorded (into the
scope as `requested_model`) but never overrides the model configured at
proxy start. When generation ends on a client stop sequence, the Anthropic
response reports `stop_reason: "stop_sequence"` with the matched value in
`stop_sequence` (OpenAI reports `finish_reason: "stop"`), and the stop text
is stripped from the content when the sampler included it. When the sampler
EXCLUDES the stop text, attribution is conservative (the response carries no
matched-stop metadata): a client stop is only reported when the renderer
contributed no stop strings of its own (named when there is a single
candidate, `stop_sequence: null` otherwise); if the renderer's own string
stops were also in play, the terminating stop is ambiguous and the response
reports `end_turn` rather than guessing.

Anything else that changes semantics is rejected with a clean 400 rather
than silently ignored: legacy `functions`/`function_call`, forced
`tool_choice`, images, documents, `response_format`, `logprobs`/
`top_logprobs`, `logit_bias`, `modalities`/`audio`/`prediction`, `n` other
than 1, and nonzero `presence_penalty`/`frequency_penalty`. Harmless fields
(`user`, `metadata`, `store`) are accepted and ignored;
`stream_options.include_usage` is honored (a final usage chunk before
`[DONE]`).

### Thinking and system messages

`thinking` is accepted and ignored (any object shape; Claude Code sends it
enabled by default): the proxy serves plain text and never produces
thinking blocks. Models with inline reasoning tags (Qwen-style `<think>`)
may surface them in the text as a model behavior. `system`-role messages
inside `messages[]` (Claude Code's `?beta=true` shape) are accepted and
folded into the system prompt in encounter order; renderers cannot
faithfully express a mid-conversation system turn, so prefix-folding is the
documented v0 behavior.

### Streaming

`"stream": true` returns a valid SSE event sequence on both endpoints
(Anthropic: `message_start`, then per content block a
`content_block_start`, one `content_block_delta` carrying the full payload
(`text_delta` for text, `input_json_delta` for tool_use), and
`content_block_stop`, then `message_delta`, `message_stop`; OpenAI: role
chunk, content chunk, a `tool_calls` delta chunk when present, finish chunk,
`[DONE]`). Each block arrives in a single delta;
token-by-token streaming is future work. This is enough for Claude Code,
which requires streaming.

## Running

```bash
python -m tinker_cookbook.capture.proxy.serve --port 7462 \
    --base-model Qwen/Qwen3-8B [--model-path tinker://...] \
    [--store-data-dir DIR] [--renderer-name qwen3] \
    [--flush-interval-sec 1.0] [--max-queue-size 4096] [--max-batch-size 256]
```

Startup spawns/reuses the local capture store daemon (`ensure_daemon`) and
arms `instrument_tinker()` with a `CaptureExporter` feeding a `StoreSink`.
Shutdown drains the exporter (`wait_pending`, `force_flush`, `shutdown`).

Pointing Claude Code at it:

```bash
export ANTHROPIC_BASE_URL=http://localhost:7462/r/run/X/traj/0
claude
```

`ANTHROPIC_API_KEY` can be anything on a loopback bind; the proxy
authenticates to Tinker with its own credentials.

### Security

The proxy spends Tinker credits, so non-loopback binds require a token:
`--auth-token` (or `TINKER_PROXY_AUTH_TOKEN`). With `--host 0.0.0.0` and no
token, `serve` refuses to start. When a token is set, every request except
`GET /healthz` must present it as `x-api-key: <token>` or
`Authorization: Bearer <token>` (so `ANTHROPIC_API_KEY=<token>` /
`OPENAI_API_KEY=Bearer`-style clients work unchanged); loopback binds work
with no token.

## Caveats and non-goals

- **Renderer fidelity**: capture happens at the token level, but the tokens
  are produced by this proxy's renderer, not by the harness. What is
  captured is faithful to what was sampled, but the chat-to-token mapping is
  renderer-dependent; a different renderer for the same model yields
  different prompts.
- No token-by-token streaming (single-delta SSE only).
- No forced `tool_choice`, parallel-tool-call limits, or schema-validated
  tool arguments; unparsed tool calls (`unparsed_tool_calls`) are dropped
  from responses.
- No Tinker-API passthrough: this is a chat facade, not a general proxy.
