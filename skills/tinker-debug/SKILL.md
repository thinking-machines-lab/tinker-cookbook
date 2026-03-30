---
name: tinker-debug
description: Diagnose training issues with Tinker — slow steps, hanging sessions, output mismatches, error messages, renderer problems, and deployment issues. Use this skill whenever a user reports that training is slow, steps take too long, sessions are hanging, model outputs differ between Tinker and external engines (vLLM, SGLang), they get a confusing error message, training quality is poor (high KL, bad outputs), or they suspect something is wrong. Also trigger when users ask "is this a Tinker issue or my issue?", "is Tinker down?", report unexpected wait times, see output quality regressions, get opaque errors, or want to profile/debug their training or deployment pipeline. This skill walks through systematic triage to determine root cause.
---

# Tinker Debug

Systematic triage for training and deployment issues. Five triage paths:
1. **Performance issues** — slow steps, hanging sessions, throughput problems
2. **Output correctness issues** — mismatches between Tinker sampling and external inference engines
3. **Service availability** — "is Tinker down?" quick diagnostics
4. **Renderer issues** — wrong tokens, training quality degradation, prompt mismatches
5. **Error message decoder** — mapping opaque errors to root causes

Identify which category the user's problem falls into, then follow the appropriate triage.

## How the Tinker SDK works (essential context)

Understanding the SDK's threading model is key to diagnosing most issues. The SDK runs a **background thread** with its own asyncio event loop. All network I/O, heartbeats, and API result polling happen on this thread.

```
┌─────────────────────┐     ┌──────────────────────────────┐
│     Main Thread      │     │      SDK Background Thread    │
│  (user code)         │     │  (asyncio event loop)         │
│                      │     │                               │
│  fb = tc.fwd_bwd_   │────>│  HTTP POST /forward_backward  │
│       async(data)    │     │  → returns request_id         │
│                      │     │                               │
│  # prepare next      │     │  Long-poll /retrieve_future   │
│  # batch here...     │     │  (HTTP 408 = not ready yet)   │
│                      │     │                               │
│  result = fb.result()│<────│  Result arrives → resolve     │
│  # blocks until done │     │                               │
│                      │     │  Heartbeat every 10s          │
└─────────────────────┘     └──────────────────────────────┘
```

When you call `forward_backward_async()`, the SDK:
1. Submits the request coroutine to the background thread
2. Returns a future immediately (main thread continues)
3. Background thread sends HTTP request, starts long-polling for result
4. Calling `.result()` on the future blocks main thread until the background thread resolves it

**Why this matters for debugging:** The background thread shares the Python GIL with the main thread. If user code holds the GIL for extended periods (heavy numpy/torch computation, CPU-bound data processing, slow serialization), the background thread **cannot**:
- Send heartbeats (sessions can expire after missed heartbeats)
- Poll for API results (futures appear to "hang")
- Submit new requests (pipelining breaks)

This means "my training is slow/hanging" is often caused by the user's own code blocking the SDK's background thread via GIL contention — not a network or server issue.

## Triage order

Work through these steps in order. Most issues are caught in steps 1-3 and never need deep profiling.

### Step 1: Environment check

Bad dependency versions are a silent killer. Check these first because they're fast to verify and cause mysterious slowdowns that look like service issues.

```python
import sys, pydantic, tinker
print(f"Python: {sys.version}")
print(f"pydantic: {pydantic.__version__}")
print(f"tinker SDK: {tinker.__version__}")
try:
    import torch; print(f"torch: {torch.__version__}")
except ImportError: pass
try:
    import numpy; print(f"numpy: {numpy.__version__}")
except ImportError: pass
try:
    import transformers; print(f"transformers: {transformers.__version__}")
except ImportError: pass
```

**Known problem versions:**
- `pydantic >= 2.13.0b1` (beta): Serialization regression makes `model_dump()` extremely slow on large payloads (tokens/tensors). Symptom: SDK thread stalls for minutes on `forward_backward` submission. Fix: pin `pydantic<2.13` or use a stable release.
- `transformers == 5.3.0`: Incorrect `tokenizer_class` for DeepSeek V2/V3 models (huggingface/transformers#44801, fixed in 5.3.1). Causes tokenizer loading failures. Upgrade or skip this version.
- `transformers < 5.0`: Bug in `Qwen2VLImageProcessor` that miscounts image tokens for VL models. Fix: upgrade to `>=5.0`.
- Always check if the user is on the **latest stable** tinker SDK. Suggest `pip install --upgrade tinker`.

If the user has a beta or pre-release of any core dependency, that's the likely culprit. Suggest downgrading before deeper investigation.

### Step 2: Async pipelining check

The single most common performance mistake. If the user's code awaits each API call before submitting the next, the GPU sits idle between steps.

**Ask the user to share their training loop code**, then look for these anti-patterns:

```python
# BAD: sequential — GPU idle while client prepares next call
result = tc.forward_backward(data=batch, loss_fn="cross_entropy")  # blocks
tc.optim_step(adam_params=params)  # blocks

# BAD: async but still sequential
result = await tc.forward_backward_async(data=batch, loss_fn="cross_entropy")
await tc.optim_step_async(adam_params=params)

# GOOD: pipelined — submit both before awaiting
fb_future = tc.forward_backward_async(data=batch, loss_fn="cross_entropy")
optim_future = tc.optim_step_async(adam_params=params)
# ... prepare next batch while GPU works ...
fb_result = fb_future.result()
optim_result = optim_future.result()
```

If the user is using the cookbook's `supervised.train` or `rl.train`, pipelining is handled automatically. If they have a custom script, check their loop carefully. If they're struggling with async patterns, suggest switching to the cookbook's training scripts which handle pipelining, checkpointing, and logging out of the box.

**Quick test:** If the user reports "first step is slow but later steps are faster," that's often normal warm-up (model loading, JIT compilation). If *every* step is slow, the issue is likely pipelining or serialization.

### Step 3: Quick timing breakdown

Before reaching for heavy profiling tools, get a rough breakdown of where time goes. The cookbook's built-in tracing does this automatically.

If the user is running the cookbook's train scripts, check the `log_path` output:

```python
# Read timing metrics from a training run
import json
with open("path/to/metrics.jsonl") as f:
    metrics = [json.loads(line) for line in f]

# Look at timing keys for the last few steps
for m in metrics[-3:]:
    timing = {k: v for k, v in m.items() if k.startswith("time/")}
    print(f"step {m.get('progress/batch')}: {timing}")
```

**What to look for:**
- `time/forward_backward` >> `time/optim_step` — Normal, fwd/bwd is heavier
- `time/get_batch` is large — Data loading is the bottleneck, not Tinker
- `time/total` >> sum of individual times — There are gaps between operations (pipelining issue)
- Large `time/forward_backward` on step 0, then normal — Warm-up, not a bug

If the user has a custom script without cookbook tracing, suggest wrapping key sections:

```python
import time

t0 = time.perf_counter()
fb_future = tc.forward_backward_async(data=batch, loss_fn="cross_entropy")
t_submit = time.perf_counter() - t0

t0 = time.perf_counter()
result = fb_future.result()
t_wait = time.perf_counter() - t0

print(f"submit: {t_submit:.2f}s, wait: {t_wait:.2f}s")
```

For a more detailed view, `pyinstrument` with async mode shows exactly where time goes:

```bash
pip install pyinstrument
pyinstrument --async-mode=enabled your_script.py
```

**Interpreting results:**
- **High submit time** → Client-side bottleneck (serialization, data prep). Go to Step 4.
- **High wait time** → Either network or server-side. Go to Step 5.
- **Both reasonable but steps are slow** → Check for gaps between steps (pipelining). Go back to Step 2.

### Step 4: Profile the request lifecycle

The key is to understand where in the request lifecycle time is being spent. Every Tinker API call goes through: **submit → serialize → network send → server compute → long-poll → resolve**. GIL contention can block steps on the SDK background thread silently.

#### Option A: Cookbook tracing (if using cookbook train scripts)

The cookbook automatically produces Gantt charts and Perfetto traces:

```python
# View the Gantt chart — shows each operation as a timeline bar
# Open: <log_path>/iteration_000000/timing_gantt.html

# View Perfetto trace (more detail, shows both threads)
python -m tinker_cookbook.utils.trace <log_path>/trace_events.jsonl -o trace.json
# Open https://ui.perfetto.dev/ and load trace.json
```

In the Gantt chart, look for **gaps between bars** — these are periods where neither the main thread nor the SDK is doing useful work. Common patterns:
- **Long bar for `forward_backward`** with gaps before/after → Pipelining issue
- **Long bar for `get_batch`** → Data loading bottleneck
- **Gaps with no bars at all** → GIL contention (background thread blocked)

#### Option B: pyinstrument (for custom scripts)

```bash
pip install pyinstrument
pyinstrument --async-mode=enabled your_script.py
```

The `--async-mode=enabled` flag is critical — without it, time spent in `await` all gets attributed to `epoll.poll` which tells you nothing.

**What to look for:**
- Time in `pydantic` serialization (`model_dump`, `__repr__`) → Dependency issue (Step 1)
- Time in `AwaitableConcurrentFuture.result_async` → Waiting for server (network or server-side)
- Time in data preparation, tokenization, numpy/torch ops → GIL contention risk

#### Option C: Thread stack watchdog (for hung/slow sessions)

When a session is actively slow or hung, use the async task dump + thread stack watchdog to see what both threads are doing in real-time. Read `references/async-task-dump.md` for ready-to-paste diagnostic code.

**Interpreting the output:**
- SDK thread has pending `_forward_backward_async` tasks → Work submitted, waiting for server
- SDK thread has pending `_result_async` → Normal long-polling behavior
- SDK thread stopped logging entirely → **GIL contention** — the background thread can't wake up
- Main thread stuck in numpy/torch/data code while SDK thread is stalled → GIL is the bottleneck

### Step 5: GIL contention and background thread blocking

This is the most subtle and most common cause of "mysterious" slowdowns. Because the SDK's background thread shares Python's GIL with the main thread, CPU-heavy work in the user's code blocks the SDK from:
- **Sending heartbeats** → Session may expire (warning: "Session heartbeat failed")
- **Polling for results** → Futures appear to hang for minutes
- **Submitting new requests** → Pipelining breaks even when using `_async` variants

**Symptoms of GIL contention:**
- Steps are slow but server-side traces show the GPU finished quickly
- Heartbeat warnings in the logs
- Inconsistent step times (varies with data processing load)
- `pyinstrument` shows most time in `epoll.poll` (even with async mode) — the event loop couldn't run

**Common GIL-heavy operations in training scripts:**
- Large numpy array operations (tokenization, data preprocessing)
- Torch tensor operations on CPU (not GPU — GPU ops release the GIL)
- Heavy JSON/pydantic serialization
- File I/O for large datasets
- `transformers` tokenizer calls on large batches

**Fixes:**
1. **Move heavy work out of the hot loop**: Preprocess data before training starts
2. **Use the cookbook's training scripts**: They pipeline data prep with async API calls
3. **Offload to subprocess**: For sampling-heavy workloads, the SDK supports subprocess isolation via `TINKER_SUBPROCESS_SAMPLING=1`, which gives sampling its own event loop in a separate process (no GIL sharing)
4. **Break up long CPU operations**: Insert `await asyncio.sleep(0)` or small yields between heavy processing to let the background thread run

**Diagnostic: Is GIL the problem?**

```python
import threading, time

def gil_monitor(interval=2.0):
    """Prints when the GIL blocks this thread for too long."""
    expected = interval
    while True:
        t0 = time.monotonic()
        time.sleep(interval)
        actual = time.monotonic() - t0
        jitter = actual - expected
        if jitter > 0.5:  # >500ms jitter = GIL contention
            print(f"[GIL monitor] slept {actual:.2f}s instead of {expected:.1f}s "
                  f"(jitter: {jitter:.2f}s) — likely GIL contention")

threading.Thread(target=gil_monitor, daemon=True).start()
# ... rest of training script ...
```

This runs a monitoring thread that detects when `time.sleep()` takes significantly longer than requested — a sign that the GIL was held by another thread.

### Step 6: Network vs. server-side

If GIL is not the issue and the client submits quickly, check whether the wait is network or server:

```python
import time, tinker
svc = tinker.ServiceClient()
t0 = time.perf_counter()
caps = svc.get_server_capabilities()
print(f"API round-trip: {time.perf_counter() - t0:.2f}s")
# >2s suggests network issues; <1s rules out network
```

If network is fast, share the session ID (`tc.get_info()`) with the Tinker team for server-side investigation.

### Step 7: Escalation

If you've verified that:
1. Dependencies are on stable, recent versions
2. The training loop uses proper async pipelining
3. GIL contention is not the issue
4. Client-side profiling shows most time in SDK `await` calls
5. Network round-trip is fast

Then the issue is likely server-side. Help the user file a report with:
- **Session ID** (from `tc.get_info()`)
- **Step timing** (from metrics.jsonl or manual timing)
- **pyinstrument profile** (with `--async-mode=enabled`)
- **Tinker SDK version** and other dependency versions
- **What machine they're running on** (cloud instance type, region)

---

## Output correctness triage

When the user reports that their fine-tuned model behaves differently in external inference engines (vLLM, SGLang, TGI) compared to Tinker's sampling client, the problem is almost always in the **weight merge/export** step, not in training.

Tinker's sampling client serves the unmerged LoRA adapter directly on top of the base model — it's the ground truth. If the model produces correct outputs through Tinker sampling but wrong outputs after export, the merge is the first suspect.

### Step 1: Use the cookbook merge, not a custom script

The most common cause of merge issues is users writing their own merge scripts. The cookbook's `weights.build_hf_model()` handles model-specific weight layouts automatically — MoE expert fusion, VL model prefixes, split QKV projections, and more.

```python
from tinker_cookbook import weights

# Download adapter
adapter_dir = weights.download(
    tinker_path="tinker://session-id/sampler_weights/step_N",
    output_dir="./adapter",
)

# Merge LoRA into base model (handles all model-specific layouts)
weights.build_hf_model(
    base_model="Qwen/Qwen3-8B",
    adapter_path=adapter_dir,
    output_path="./merged_model",
    dtype="bfloat16",
)
```

If the user has a custom merge script, strongly recommend switching to the cookbook's implementation first. It handles:
- **MoE gate_up_proj fusion** — Correctly detects concatenated vs interleaved layouts per model family
- **VL model prefixes** — Adds `model.language_model.*` prefix for vision-language models
- **Split QKV projections** — Handles Qwen3.5 fused `in_proj_qkv` with unequal Q/K/V dimensions
- **Shard-by-shard processing** — Low memory merge for large models

### Step 2: Verify prompt equivalence

Even with correct weights, different inference engines may tokenize or render prompts differently. Verify the model receives identical input:

```python
# Compare token IDs between Tinker and external engine
# Tinker side:
model_input = renderer.build_supervised_example(messages)[0]
tinker_tokens = [chunk.tokens for chunk in model_input.chunks if hasattr(chunk, 'tokens')]

# External engine side:
external_tokens = tokenizer.apply_chat_template(messages, tokenize=True)

# Compare
assert tinker_tokens == external_tokens, "Token mismatch — check chat template and renderer"
```

For vision models, also verify:
- Image tokens are in the same positions
- Image preprocessing (resize, normalization) matches
- The number of image tokens per image matches

### Step 3: Check for known merge pitfalls

If the user must use a custom merge, the most common issues are:
- **MoE gate_up_proj fusion convention** — Concatenated (Qwen3.5, Qwen3-VL) vs interleaved (GPT-OSS). Wrong convention silently corrupts weights.
- **Precision loss** — LoRA merge math must be done in float32, then cast to bfloat16. Direct bfloat16 matmul introduces errors.
- **Tinker weight naming** — `w1`=gate, `w2`=down, `w3`=up. Swapping `w1`/`w3` is a common bug.
- **VL model prefix** — Vision-language models add `model.language_model.*` prefix that custom scripts often miss.

For full details (weight layout diagrams, validation scripts, tensor comparison code), read `references/merge-debugging.md`.

### Step 4: Try PEFT adapter as workaround

If merge issues persist, skip the merge entirely and serve the unmerged adapter:

```python
weights.build_lora_adapter(
    base_model="Qwen/Qwen3-8B",
    adapter_path=adapter_dir,
    output_path="./peft_adapter",
)
# Then serve with vLLM: --lora-modules my_adapter=./peft_adapter
```

This lets the engine apply the LoRA at inference time, sidestepping merge-related precision and layout bugs.

### Step 5: Escalation for correctness issues

If the cookbook merge + correct prompts still produce wrong outputs, gather: model name/size, session ID, merge method, engine version, example input/output comparison, and token IDs confirming prompt equivalence.

---

## Service availability triage

When the user asks "is Tinker down?" or operations hang/fail unexpectedly, run a quick smoke test before deeper investigation. Many users can't distinguish a service outage from a bug in their code.

### Quick smoke test

Help the user run a three-step diagnostic to isolate where things break:

```python
import time, tinker

svc = tinker.ServiceClient()

# Step 1: API reachable?
t0 = time.perf_counter()
try:
    caps = svc.get_server_capabilities()
    print(f"API reachable ({time.perf_counter() - t0:.2f}s), {len(caps.models)} models available")
except Exception as e:
    print(f"API unreachable: {type(e).__name__}: {e}")

# Step 2: Can we create a training session? (small model for speed)
try:
    tc = svc.create_lora_training_client(base_model="meta-llama/Llama-3.2-1B", rank=32)
    print(f"Training client created: {tc.get_info()}")
except Exception as e:
    print(f"Training client failed: {type(e).__name__}: {e}")
```

**Interpreting:** Step 1 fails → service down or network. Step 1 passes but Step 2 fails → model-specific capacity issue (try smaller model). Both pass but user's script fails → issue is in user's code.

### Common server-side errors

| Error | Meaning |
|-------|---------|
| `APIConnectionError` | Service down or network issue |
| `APITimeoutError` | Service overloaded |
| HTTP 402 | Billing blocked — check Tinker console |
| HTTP 429 | Rate limited — reduce concurrency |
| HTTP 500 | Server bug — gather session ID and report |
| Client creation hangs | Capacity shortage — SDK may not surface error clearly |

---

## Renderer triage

Renderer issues cause **silent training degradation** — the model trains on wrong tokens, producing poor results without any error messages. This is the most common source of subtle quality bugs.

The renderer converts chat-style messages into model-specific token sequences. If the renderer produces different tokens than the model expects, training teaches the wrong associations. The model may still train (loss goes down) but learn garbage.

### When to suspect a renderer issue

- Training loss decreases but model outputs are poor quality
- High KL divergence at step 0 (before any training) — indicates the prompt tokens don't match what the model expects
- Model produces garbled or off-distribution outputs
- Tool calling works in some models but not others
- Thinking/reasoning blocks appear or disappear unexpectedly

### Step 1: Verify you're using the right renderer

```python
from tinker_cookbook import model_info

renderer_name = model_info.get_recommended_renderer_name("Qwen/Qwen3-8B")
print(f"Recommended renderer: {renderer_name}")
```

Never hardcode renderer names. Each model family has specific token formats, and using the wrong renderer silently produces incorrect training data.

**Hybrid models (thinking + non-thinking)** are especially tricky. These models support both reasoning (`<think>` blocks) and direct responses. Using the wrong variant causes token-level mismatches:

| Model family | Thinking renderer | Non-thinking renderer | Notes |
|-------------|-------------------|----------------------|-------|
| Qwen3 | `qwen3` | `qwen3_disable_thinking` | Default is thinking-enabled. `qwen3_instruct` for instruction-only. |
| Qwen3.5 | `qwen3_5` | `qwen3_5_disable_thinking` | Hybrid attention; also has VL variants. |
| DeepSeek V3 | `deepseekv3_thinking` | `deepseekv3` | Default is non-thinking. Thinking adds `<think>` prefill. |
| Kimi K2.5 | `kimi_k25` | `kimi_k25_disable_thinking` | Vision-capable. |
| Nemotron3 | `nemotron3` | `nemotron3_disable_thinking` | |

**Common hybrid model mistakes:**
- Training on data with `<think>` blocks using a `_disable_thinking` renderer → Thinking tokens treated as regular text
- Using a thinking renderer but testing with `temperature=0` and short `max_tokens` → Model spends all tokens thinking, never produces an answer
- Comparing against HF template without passing `thinking=True` → Token mismatch that looks like a renderer bug but is a test setup issue

### Step 2: Compare tokens against HuggingFace

The ground truth is the model's HuggingFace tokenizer with `apply_chat_template`. Compare your renderer's output against it:

```python
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name

model_name = "Qwen/Qwen3-8B"
tokenizer = get_tokenizer(model_name)
renderer_name = get_recommended_renderer_name(model_name)
renderer = get_renderer(renderer_name, tokenizer)

# Test conversation
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
]

# Cookbook tokens
cookbook_mi = renderer.build_generation_prompt(messages)
cookbook_tokens = cookbook_mi.to_ints()

# HuggingFace tokens
hf_tokens = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

if cookbook_tokens == list(hf_tokens):
    print("MATCH: Renderer tokens match HuggingFace")
else:
    print("MISMATCH: Renderer diverges from HuggingFace")
    # Find first divergence point
    for i, (a, b) in enumerate(zip(cookbook_tokens, hf_tokens)):
        if a != b:
            print(f"  First diff at position {i}: cookbook={a} ({tokenizer.decode([a])!r}) vs HF={b} ({tokenizer.decode([b])!r})")
            break
    print(f"  Cookbook length: {len(cookbook_tokens)}, HF length: {len(hf_tokens)}")
```

**If tokens match:** The renderer is correct. The issue is elsewhere (training config, data quality, etc.).

**If tokens don't match:** Check these common causes:
- **Thinking mode**: Some models (Qwen3, DeepSeek) need `thinking=True` passed to `apply_chat_template`. The renderer handles this automatically, but make sure you're comparing apples to apples.
- **System prompt**: Some renderers inject a default system prompt (Kimi K2). If your HF comparison doesn't include one, tokens will diverge.
- **Tool calling format**: Each model family uses a different tool call format. The renderer must match the model's expected format exactly.

### Step 3: Check thinking mode handling

For models with thinking capabilities (Qwen3, DeepSeek V3, Kimi K2.5, Nemotron3):

- Use the `_disable_thinking` renderer variant if you don't want `<think>` blocks
- Historical assistant messages may have thinking stripped by default (depends on renderer)
- If training on thinking data, ensure the renderer preserves `<think>` blocks in the training tokens

### Step 4: Validate tool calling

Tool call formats vary significantly across models. If tool calling quality is poor after training:

1. Check that the renderer's tool format matches what the model was pre-trained on
2. Verify `parse_response()` correctly extracts tool calls from model output
3. Compare tool call rendering against HF's `apply_chat_template(..., tools=tool_specs)`

For the full renderer reference (all models, formats, edge cases), read `references/renderer-debugging.md`.

---

## Error message decoder

Tinker errors can be opaque. This section maps common error messages to root causes and fixes.

### SDK / API errors

| Error message | Root cause | Fix |
|---------------|-----------|-----|
| `max() iterable argument is empty` | Empty token list in a `Datum` — usually an `EncodedTextChunk` with no tokens | Validate your data: ensure every datum has at least one non-empty chunk with tokens |
| `Could not convert loss function inputs to array record` | Extra fields in `loss_fn_inputs` that the loss function doesn't expect (e.g., `mask` field not stripped) | Use the cookbook's data helpers which strip extra fields automatically; or remove `mask` from `loss_fn_inputs` before passing to `forward_backward` |
| `Unknown client error` | Generic catch-all — often means the checkpoint type is wrong | If during sampling: did you pass a `save_state` path instead of `save_weights_for_sampler`? State checkpoints can't be used for sampling. |
| `prompt tokens + max_tokens > context window` | Prompt too long for the model | Reduce prompt length or `max_tokens`. The error message shows the specific limits. |
| `Failed after exhausting retries` | Transient server error that didn't recover | Check service availability (smoke test above). If service is up, retry with a fresh session. |
| `Access blocked` / HTTP 403 | No permission to access the model or checkpoint | Check API key, organization membership, or checkpoint visibility settings |
| HTTP 402 | Billing issue — account blocked | Add credits at the Tinker console billing page |
| HTTP 429 | Rate limited — too many concurrent requests | Reduce concurrency (default limit: ~2000 concurrent sampling requests). Add backoff/retry logic. |
| HTTP 409 on checkpoint save | Checkpoint already exists (retry after transient failure) | The original save succeeded. Check if the checkpoint is already there before retrying. |
| `session_id is required` with hint about SDK version | Tinker SDK too old | Run `pip install --upgrade tinker` |

### Cookbook errors

| Error message | Root cause | Fix |
|---------------|-----------|-----|
| `Unknown model: {name}` | Model name not in cookbook's registry | Check spelling; use `model_info.get_model_attributes()` to list known models |
| `tokens and weights must be the same length` | Mismatch between token sequence and loss weight array | Check your `datum_from_model_input_weights()` call — usually means `max_length` truncated tokens but not weights |
| `Expected X tokens, got Y from image` | VLM image token count mismatch — usually a `transformers` version issue | Upgrade `transformers>=5.0` or install `torchvision`. HuggingFace `Qwen2VLImageProcessor` had a bug in older versions. |
| `RendererError: Unknown renderer` | Invalid renderer name | Use `model_info.get_recommended_renderer_name(model_name)` |
| `qwen3_vl renderer requires an image_processor` | VL renderer needs image processor | Pass `image_processor` to `get_renderer()`. Load it from `transformers.AutoProcessor`. |
| `DataFormatError: Each line must contain a 'messages' field` | JSONL data file has wrong format | Each line must be a JSON object with a `messages` key containing a list of message dicts |
| `StreamingSupervisedDatasetFromHFDataset only supports forward iteration` | Tried to seek backward in streaming dataset | Streaming datasets are forward-only; don't try to restart from an earlier batch |

For the full error reference with additional edge cases, read `references/error-reference.md`.

---

## Decision tree summary

```
Output differs between Tinker and external engine
├─ Using custom merge script? → Switch to cookbook weights.build_hf_model()
├─ Prompts identical? → Compare token IDs and image preprocessing
├─ MoE model? → Check gate_up_proj fusion convention (concat vs interleave)
├─ Large model / numerical issues? → Check merge precision (float32), try PEFT adapter
└─ Cookbook merge + correct prompts + still wrong → Engine-specific issue → Escalate
├─ Using custom merge script? → Switch to cookbook weights.build_hf_model()
├─ Prompts identical? → Compare token IDs and image preprocessing
├─ MoE model? → Check gate_up_proj fusion convention (concat vs interleave)
├─ Large model / numerical issues? → Check merge precision (float32), try PEFT adapter
└─ Cookbook merge + correct prompts + still wrong → Engine-specific issue → Escalate

Training is slow
├─ Check dependency versions (Step 1)
│  └─ Beta/pre-release pydantic? → Downgrade
├─ Check async pipelining (Step 2)
│  └─ Sequential API calls? → Pipeline them
├─ Get timing breakdown (Step 3)
│  ├─ High submit time → Profile request lifecycle (Step 4)
│  │  ├─ Slow serialization → Dependency issue
│  │  ├─ Slow data loading → Optimize data pipeline
│  │  └─ SDK thread stalled → GIL contention (Step 5)
│  └─ High wait time → GIL or network or server
│     ├─ Heartbeat warnings in logs → GIL contention (Step 5)
│     ├─ Fast API round-trip → Server-side → Escalate (Step 7)
│     └─ Slow API round-trip → Network issue
├─ Inconsistent step times → GIL contention (Step 5)
└─ First step slow, rest fast → Normal warm-up

Is Tinker down?
├─ Run smoke test (ServiceClient + create small training client)
│  ├─ API unreachable → Service down or network issue
│  ├─ API works but training client fails → Model-specific capacity issue
│  └─ Everything works → Issue is in user's code
└─ Check error: HTTP 402 = billing, 429 = rate limit, 500 = server bug

Training quality is poor (high KL, bad outputs)
├─ KL high at step 0? → Renderer mismatch (tokens don't match model's expected format)
├─ Compare renderer tokens vs HF apply_chat_template
│  ├─ Tokens match → Issue is elsewhere (LR, data quality, loss function)
│  └─ Tokens differ → Wrong renderer or renderer bug
├─ Tool calling broken? → Check renderer tool format matches model family
└─ Thinking blocks wrong? → Use correct _disable_thinking variant
```

## Common resolutions

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Every step takes 5-10min | Missing async pipelining | Use `_async` variants, submit before await |
| First step slow, rest normal | Model warm-up / JIT | Expected behavior, no fix needed |
| Steps hang indefinitely | Dependency bug or network | Check pydantic version; try different machine |
| Slow with large batch_size | Payload serialization | Check pydantic version; reduce batch_size |
| Works on one machine, not another | Environment difference | Compare `pip freeze` outputs |
| GPU time looks fine but steps slow | Gaps between submissions | Pipeline: submit next step before awaiting current |
| Heartbeat warnings + slow steps | GIL contention (CPU work blocking SDK thread) | Preprocess data outside loop; use `TINKER_SUBPROCESS_SAMPLING=1` |
| Inconsistent step times | GIL contention varies with data batch | Move heavy numpy/torch CPU ops out of hot loop |
| Session expired unexpectedly | Missed heartbeats (SDK thread blocked) | Reduce GIL-holding operations; use subprocess sampling |
| Output correct in Tinker, wrong in vLLM/SGLang | Merge bug (usually fused projection layout) | Use cookbook `weights.build_hf_model()` |
| Model produces invalid/garbled outputs after export | Wrong gate_up_proj convention or precision loss | Check concat vs interleave; merge in float32 |
| Numerical instability after deploy | Merge precision or engine quantization | Try `build_lora_adapter()` instead of merging |
| Outputs subtly different but not completely wrong | bfloat16 merge rounding | Merge in float32, cast back after |
| High KL at step 0 (before any training) | Renderer produces wrong tokens | Compare renderer tokens vs HF `apply_chat_template()` |
| Training loss drops but outputs are poor | Renderer mismatch or wrong `train_on_what` | Verify renderer; check loss weight masking |
| `create_lora_training_client` hangs forever | Capacity shortage for that model | Try smaller model; check service availability |
| `Expected X tokens, got Y from image` | `transformers` version bug in VLM processor | Upgrade to `transformers>=5.0` |
| `max() iterable argument is empty` | Empty token list in datum | Validate all datums have non-empty chunks |
| Operations work but sporadically fail/hang | Service under load or transient issues | Add retry logic; gather session IDs for reports |

## Code references

**Key imports for debugging:**
```python
from tinker_cookbook import model_info, weights
from tinker_cookbook.renderers import get_renderer, get_registered_renderer_names
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import trace, ml_log
import tinker
```

**Reference files in this skill:**
- `references/async-task-dump.md` — Ready-to-paste diagnostic for hung sessions
- `references/serialization-test.md` — Pydantic regression benchmark
- `references/merge-debugging.md` — Weight layout conventions and fusion formats
- `references/renderer-debugging.md` — Renderer validation, token comparison, model-specific quirks
- `references/error-reference.md` — Extended error message decoder with edge cases
