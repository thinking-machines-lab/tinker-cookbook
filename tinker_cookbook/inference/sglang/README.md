# Serving multiple Tinker LoRAs on SGLang

Turn `tinker://` checkpoints into adapters SGLang can load, serve them all from
one process, and check that the served model matches what Tinker trained.

Assumes you already have a container or pod running with SGLang available —
`lmsysorg/sglang:latest` is a good default — and that the adapter directory is
visible inside it.

## Setup

```bash
pip install tinker-cookbook
export TINKER_API_KEY=tml-...
export HF_TOKEN=hf-...            # only for gated base models
```

Install this somewhere separate from the environment running SGLang.
`tinker-cookbook` pins `transformers<=5.5.4` while current SGLang images ship a
newer one, so installing it alongside the server downgrades transformers
underneath it. Conversion and serving only ever exchange a directory of files.

## 1. Convert your checkpoints

```bash
python -m tinker_cookbook.inference.sglang.prepare \
  --base-model Qwen/Qwen3.8-27B \
  --adapter lora1=tinker://<run-id>/sampler_weights/final \
  --adapter lora2=tinker://<run-id>/sampler_weights/final
```

Each adapter is downloaded from the Tinker API, converted to PEFT format under
`./adapters/<base-model>/<name>/`, and reported with its rank and target
modules. The command prints the `--lora-paths` value to paste into step 2; pass
`--lora-root` if the tree appears at a different path inside your container.

Conversion reads the base model's parameter *names and shapes* — never its
values — to remap Tinker's adapter keys, so it needs the base model locally. Set
`HF_HUB_CACHE` to the cache your container uses and that download is paid once
instead of twice, or pass a local directory as `--base-model` to skip the Hub.

## 2. Launch

Take whatever command you already use to serve the base model and add two
flags to it: `--lora-paths` to point at the adapters, and `--max-loras-per-batch`
to bound how many can be active in one batch.

```bash
sglang serve Qwen/Qwen3.8-27B \
  --host 0.0.0.0 --port 30000 \
  --lora-paths lora1=/adapters/Qwen__Qwen3.8-27B/lora1 \
               lora2=/adapters/Qwen__Qwen3.8-27B/lora2 \
  --max-loras-per-batch 4
```

Keep your own `--tp`, quantization, and attention settings alongside these.

MoE models also need a LoRA-capable `--moe-runner-backend`. Several runners
implement only the fused path and cannot apply expert LoRA, including the
`flashinfer_trtllm` that SGLang picks by default on Blackwell. `triton` works;
the Inkling rows below use the `experimental_sgl_*` runners.

## 3. Query

Pick the adapter per request. OpenAI-compatible route:

```bash
curl localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-27B:lora1",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

Drop the `:lora1` suffix to hit the un-adapted base model. On the native
`/generate` route the equivalent is `"lora_path": "lora1"`, or `null` for the
base model — and both accept per-request lists, so one batch can mix adapters.

## 4. Verify

```bash
python -m tinker_cookbook.inference.sglang.compare \
  --base-model Qwen/Qwen3.8-27B --url http://localhost:30000 \
  --adapter lora1=tinker://<run-id>/sampler_weights/final \
  --adapter lora2=tinker://<run-id>/sampler_weights/final
```

Compares Tinker's logprobs against SGLang's on the same tokens and exits nonzero
if they disagree, reporting which adapter failed and why. Three checks: each
adapter is applied at all, each request is answered by the adapter it asked
for, and each adapter's pooled distance from Tinker stays within twice the
noise floor — what the two engines disagree by on the same tokens with no
adapter anywhere, measured in the same run so quantized deployments are held
to their own achievable bar rather than a constant.

The floor needs a base id Tinker can sample. Where the checkpoint's own id is
not sampleable (GLM-5.3), pass `--tinker-base` with one that is, e.g.
`--tinker-base zai-org/GLM-5.3:peft:262144`; otherwise the closeness check
reports itself unavailable and only the first two checks gate the exit code.

## Verified models

Each command below was run end to end on a 4xGB300 node: two adapters trained on
Tinker, converted with `prepare`, served, and checked with `compare`.

### Qwen/Qwen3.8-27B — dense

Base command from the model's
[cookbook page](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B#hw=gb300&variant=default&quant=bf16&nodes=single&spec=none&tier=low-latency&ssmDtype=float32), with the LoRA flags added:

```bash
sglang serve Qwen/Qwen3.8-27B \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 2048 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-full-memory-ratio 4.59 \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-ssm-dtype float32 \
  --lora-paths lora1=/adapters/Qwen__Qwen3.8-27B/lora1 \
               lora2=/adapters/Qwen__Qwen3.8-27B/lora2 \
  --max-loras-per-batch 3 \
  --host 0.0.0.0 --port 30000
```

### thinkingmachines/Inkling-Small — MoE, shared-outer experts

Base command from the model's
[cookbook page](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-Small#hw=gb300&variant=lora&quant=bf16&strategy=balanced&nodes=single), with the LoRA paths filled in:

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
SGLANG_EXPERIMENTAL_LORA_OPTI=1 \
SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1 \
SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN=1 \
sglang serve thinkingmachines/Inkling-Small \
  --trust-remote-code --tp 4 \
  --moe-runner-backend experimental_sgl_trtllm \
  --attention-backend fa4 \
  --enable-torch-symm-mem \
  --mamba-radix-cache-strategy extra_buffer \
  --mem-fraction-static 0.87 \
  --swa-full-tokens-ratio 0.1 \
  --mamba-full-memory-ratio 0.1 \
  --enable-multimodal \
  --reasoning-parser inkling --tool-call-parser inkling \
  --disable-prefill-cuda-graph \
  --lora-backend triton --lora-use-virtual-experts \
  --max-loras-per-batch 3 \
  --lora-paths lora1=/adapters/thinkingmachines__Inkling-Small/lora1 \
               lora2=/adapters/thinkingmachines__Inkling-Small/lora2 \
  --host 0.0.0.0 --port 30000
```

### thinkingmachines/Inkling — MoE, served from the NVFP4 checkpoint

Same shape as above but from the [NVFP4 cell](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling#hw=gb300&variant=lora&quant=nvfp4&strategy=balanced&nodes=single). The adapter is trained
on the bf16 base model and served with w4a16:

```bash
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
SGLANG_EXPERIMENTAL_LORA_OPTI=1 \
SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1 \
sglang serve thinkingmachines/Inkling-NVFP4 \
  --trust-remote-code --tp 4 \
  --quantization modelopt_fp4 \
  --attention-backend fa4 --page-size 128 \
  --fp4-gemm-backend marlin \
  --moe-runner-backend experimental_sgl_marlin \
  --enable-torch-symm-mem \
  --mamba-radix-cache-strategy extra_buffer \
  --mem-fraction-static 0.80 \
  --swa-full-tokens-ratio 0.1 \
  --mamba-full-memory-ratio 0.1 \
  --enable-multimodal \
  --reasoning-parser inkling --tool-call-parser inkling \
  --disable-prefill-cuda-graph \
  --lora-backend triton --lora-use-virtual-experts \
  --max-loras-per-batch 3 \
  --lora-paths lora1=/adapters/thinkingmachines__Inkling/lora1 \
               lora2=/adapters/thinkingmachines__Inkling/lora2 \
  --host 0.0.0.0 --port 30000
```

### zai-org/GLM-5.3 — MoE with sparse attention

Base command from the model's [cookbook page](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3#hw=gb300&variant=default&quant=fp8&strategy=low-latency&nodes=single), minus its speculative
decoding flags, plus expert parallelism and the triton MoE runner for LoRA:

```bash
sglang serve zai-org/GLM-5.3 \
  --tp 4 --ep-size 4 \
  --mem-fraction-static 0.85 \
  --moe-runner-backend triton \
  --max-loras-per-batch 3 \
  --lora-paths lora1=/adapters/zai-org__GLM-5.3/lora1 \
               lora2=/adapters/zai-org__GLM-5.3/lora2 \
  --host 0.0.0.0 --port 30000
```
