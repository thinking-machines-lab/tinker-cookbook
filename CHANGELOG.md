# Changelog

A curated feed of notable changes to `tinker-cookbook`. Small bugfixes and minor argument additions are omitted—this is for changes worth knowing about.

## Format

Each entry includes:
- **Title**: A short, human-readable summary (not the commit message)
- **Date**: When it was merged
- **Type**: `new` (feature), `improvement` (enhancement to existing functionality), or `fix`
- **Tags**: What area it touches (e.g., `renderers`, `rl`, `supervised`, `eval`, `datasets`)
- **PR**: Link to the pull request

---

### [cookbook] Auto-generated model cards for HuggingFace Hub publishing ([#543](https://github.com/thinking-machines-lab/tinker-cookbook/pull/543))
**Date:** 2026-03-26
**Type:** new
**Tags:** weights

`publish_to_hf_hub` now accepts an optional `ModelCardConfig` to auto-generate a HuggingFace-compatible `README.md` with YAML frontmatter, usage snippets, and framework versions. Adapter vs merged model format is auto-detected from `adapter_config.json`. A standalone `generate_model_card` function is also available for previewing cards before publishing.

---

### [cookbook] Group per-iteration output files into subdirectories ([#517](https://github.com/thinking-machines-lab/tinker-cookbook/pull/517))
**Date:** 2026-03-25
**Type:** improvement
**Tags:** infrastructure, logging

Training output files (rollout summaries, logtree JSONs, HTML reports) are now grouped into per-iteration subdirectories under `log_path`, keeping the output directory clean.

---

### [cookbook] PEFT-format adapter serving with `build_lora_adapter` ([#533](https://github.com/thinking-machines-lab/tinker-cookbook/pull/533))
**Date:** 2026-03-24
**Type:** new
**Tags:** weights, adapters

New `build_lora_adapter` function exports trained LoRA weights in PEFT format, enabling adapter serving without merging into the base model. Includes Nemotron-3 adapter serving support and vLLM e2e tests ([#539](https://github.com/thinking-machines-lab/tinker-cookbook/pull/539)).

---

### [cookbook] ActionExtra TypedDict for Env.step extensibility ([#538](https://github.com/thinking-machines-lab/tinker-cookbook/pull/538))
**Date:** 2026-03-24
**Type:** new
**Tags:** rl

Introduces `ActionExtra` TypedDict so environments can pass additional structured data through `Env.step()` without breaking the existing interface.

---

### [cookbook] Multimodal tool result support ([#526](https://github.com/thinking-machines-lab/tinker-cookbook/pull/526))
**Date:** 2026-03-24
**Type:** new
**Tags:** renderers, tools

Tool results can now include images and other multimodal content, not just text.

---

### [cookbook] Unified training telemetry with `trace_iteration` and `scope_span` ([#522](https://github.com/thinking-machines-lab/tinker-cookbook/pull/522), [#477](https://github.com/thinking-machines-lab/tinker-cookbook/pull/477))
**Date:** 2026-03-24
**Type:** improvement
**Tags:** infrastructure, logging

All training loops (SL, RL, DPO, distillation) now share a unified telemetry system built on `trace_iteration` and `scope_span`. Generates per-iteration Gantt charts and W&B metrics via `@scope` ([#453](https://github.com/thinking-machines-lab/tinker-cookbook/pull/453)).

---

### [cookbook] Add version lower bounds and exclude compromised litellm ([#532](https://github.com/thinking-machines-lab/tinker-cookbook/pull/532))
**Date:** 2026-03-24
**Type:** fix
**Tags:** infrastructure, dependencies

Adds minimum version bounds for key dependencies and excludes compromised `litellm` versions, protecting users from known supply chain issues.

---

### [cookbook] Handle context limits and max-tokens truncation in multi-turn RL ([#506](https://github.com/thinking-machines-lab/tinker-cookbook/pull/506))
**Date:** 2026-03-24
**Type:** improvement
**Tags:** rl

Multi-turn RL environments now gracefully handle context limit exhaustion and max-tokens truncation rather than crashing.

---

### [cookbook] Fix LoRA merging for Qwen3.5 models ([#528](https://github.com/thinking-machines-lab/tinker-cookbook/pull/528), [#529](https://github.com/thinking-machines-lab/tinker-cookbook/pull/529))
**Date:** 2026-03-23
**Type:** fix
**Tags:** weights

Fixes split QKV fusion and tied vision embeddings issues when merging LoRA adapters for Qwen3.5 models. Also refactors `_merge.py` into per-model merge modules for maintainability.

---

### [cookbook] SFT hyperparameter sweep with results for 3 models ([#496](https://github.com/thinking-machines-lab/tinker-cookbook/pull/496))
**Date:** 2026-03-23
**Type:** new
**Tags:** supervised, recipes

Published SFT hyperparameter sweep results covering learning rate, batch size, and schedule across 3 model families. Useful as a starting point for tuning.

---

### [cookbook] Clean up public API surface for release ([#516](https://github.com/thinking-machines-lab/tinker-cookbook/pull/516))
**Date:** 2026-03-23
**Type:** improvement
**Tags:** infrastructure

Audit and cleanup of the public API: removed internal symbols from `__all__`, consolidated re-exports, and ensured a clean `import tinker_cookbook` surface.

---

### [cookbook] Diagnostic logs for MessageEnv and AgentToolMessageEnv ([#518](https://github.com/thinking-machines-lab/tinker-cookbook/pull/518), [#521](https://github.com/thinking-machines-lab/tinker-cookbook/pull/521))
**Date:** 2026-03-22
**Type:** new
**Tags:** rl, logging

`MessageEnv.step()` can now return diagnostic logs via `MessageStepResult`, and `AgentToolMessageEnv` populates them automatically. Useful for debugging multi-turn agent training.

---

### [cookbook] Warn when renderer is not recommended for the model ([#509](https://github.com/thinking-machines-lab/tinker-cookbook/pull/509))
**Date:** 2026-03-21
**Type:** improvement
**Tags:** renderers

A warning is now emitted when a renderer that isn't recommended for the given model is used, helping catch renderer mismatch bugs early.

---

### [cookbook] Rollout error resilience for RL training ([#497](https://github.com/thinking-machines-lab/tinker-cookbook/pull/497))
**Date:** 2026-03-20
**Type:** improvement
**Tags:** rl

RL training can now survive individual rollout failures (e.g., sandbox timeouts) without aborting the entire batch. Failed rollouts are logged and skipped.

---

### [cookbook] Slim core deps and split recipe extras ([#437](https://github.com/thinking-machines-lab/tinker-cookbook/pull/437))
**Date:** 2026-03-20
**Type:** improvement
**Tags:** infrastructure

Core `tinker-cookbook` dependencies are significantly slimmed. Recipe-specific deps (Modal, Inspect AI, etc.) are now under optional extras like `pip install tinker-cookbook[inspect]` ([#380](https://github.com/thinking-machines-lab/tinker-cookbook/pull/380)).

---

### [cookbook] `cleanup()` lifecycle method on EnvGroupBuilder ([#505](https://github.com/thinking-machines-lab/tinker-cookbook/pull/505))
**Date:** 2026-03-20
**Type:** new
**Tags:** rl

`EnvGroupBuilder` now has a `cleanup()` method called at the end of training, so environment backends (e.g., sandbox pools) can release resources gracefully.

---

### [cookbook] Fix crash when all RL advantages are zero ([#507](https://github.com/thinking-machines-lab/tinker-cookbook/pull/507))
**Date:** 2026-03-20
**Type:** fix
**Tags:** rl

Fixed a crash when an entire batch has zero advantage (all trajectories identical reward). The batch is now skipped with a warning.

---

### [cookbook] Fix async RL training hang on data exhaustion ([#480](https://github.com/thinking-machines-lab/tinker-cookbook/pull/480))
**Date:** 2026-03-18
**Type:** fix
**Tags:** rl

Fixed a deadlock where async RL training would hang when the dataset was exhausted. Uses cascading shutdown to cleanly terminate worker threads.

---

### [cookbook] Nemotron-3 model and renderer ([#492](https://github.com/thinking-machines-lab/tinker-cookbook/pull/492))
**Date:** 2026-03-18
**Type:** new
**Tags:** models, renderers

Adds support for NVIDIA Nemotron-3 model family with a dedicated renderer and LR config. Includes downstream compatibility tests ([#495](https://github.com/thinking-machines-lab/tinker-cookbook/pull/495)).

---

### [cookbook] Centralized exception hierarchy ([#489](https://github.com/thinking-machines-lab/tinker-cookbook/pull/489))
**Date:** 2026-03-18
**Type:** new
**Tags:** infrastructure

New structured exception hierarchy (`TinkerCookbookError`, `RolloutError`, `RendererError`, etc.) with picklability guarantees for distributed execution.

---

### [cookbook] Deprecation framework for API evolution ([#486](https://github.com/thinking-machines-lab/tinker-cookbook/pull/486))
**Date:** 2026-03-18
**Type:** new
**Tags:** infrastructure

New `@deprecated()` decorator and `warn_deprecated()` helper with `removal_version` enforcement. Enables smooth API transitions with clear migration paths.

---

### [cookbook] Quantized export with FP8 expert quantization ([#478](https://github.com/thinking-machines-lab/tinker-cookbook/pull/478))
**Date:** 2026-03-17
**Type:** new
**Tags:** weights

New quantized weight export supporting FP8 quantization of MoE expert layers, reducing model size for deployment.

---

### [cookbook] Shard-by-shard merging and modular merge architecture ([#476](https://github.com/thinking-machines-lab/tinker-cookbook/pull/476))
**Date:** 2026-03-17
**Type:** improvement
**Tags:** weights

LoRA merge now processes one shard at a time instead of loading all weights into memory. Modular architecture with per-model merge modules makes adding new model families easier.

---

### [cookbook] Tag-based versioning with hatch-vcs and nightly builds ([#439](https://github.com/thinking-machines-lab/tinker-cookbook/pull/439))
**Date:** 2026-03-17
**Type:** new
**Tags:** infrastructure

Package version is now derived from git tags via hatch-vcs. Nightly builds publish dev versions automatically. PyPI publishing triggers on `v*` tags via GitHub Actions ([#430](https://github.com/thinking-machines-lab/tinker-cookbook/pull/430)).

---

### [cookbook] PEP 561 `py.typed` marker ([#483](https://github.com/thinking-machines-lab/tinker-cookbook/pull/483))
**Date:** 2026-03-17
**Type:** new
**Tags:** infrastructure

Added `py.typed` marker so downstream projects using mypy/pyright get type information from `tinker-cookbook`.

---

### [cookbook] Consolidate streaming and response normalization into base Renderer ([#451](https://github.com/thinking-machines-lab/tinker-cookbook/pull/451))
**Date:** 2026-03-17
**Type:** improvement
**Tags:** renderers

Streaming token parsing and response normalization logic moved from individual renderers into the base `Renderer` class. Reduces per-model boilerplate significantly.

---

### [cookbook] Downstream compatibility tests for public API contracts ([#474](https://github.com/thinking-machines-lab/tinker-cookbook/pull/474))
**Date:** 2026-03-16
**Type:** new
**Tags:** testing

New test suite that imports and exercises the public API surface, catching accidental breakage of downstream consumers.

---

### [cookbook] `include_reasoning` option for Inspect AI integration ([#456](https://github.com/thinking-machines-lab/tinker-cookbook/pull/456))
**Date:** 2026-03-16
**Type:** new
**Tags:** eval

Inspect AI evaluators can now optionally include model reasoning (thinking traces) in their evaluation context.

---

### [cookbook] `weights/` subpackage for weight lifecycle ([#461](https://github.com/thinking-machines-lab/tinker-cookbook/pull/461))
**Date:** 2026-03-16
**Type:** new
**Tags:** weights

New `tinker_cookbook/weights/` subpackage consolidating weight download, merge, quantization, and publishing into a cohesive module.

---

### [cookbook] Support `HF_TRUST_REMOTE_CODE` env var for custom tokenizers ([#460](https://github.com/thinking-machines-lab/tinker-cookbook/pull/460))
**Date:** 2026-03-14
**Type:** new
**Tags:** models, renderers

Renderers now respect the `HF_TRUST_REMOTE_CODE` environment variable, enabling use of custom tokenizers that require `trust_remote_code=True` without code changes.

---

### [cookbook] Fix `gate_up_proj` interleave index in merge script ([#459](https://github.com/thinking-machines-lab/tinker-cookbook/pull/459))
**Date:** 2026-03-14
**Type:** fix
**Tags:** weights

Fixed a bug where the interleave index for `gate_up_proj` was always 0 during LoRA merging, which could silently produce incorrect merged weights for gated MLP models.

---

### [cookbook] LiteLLM custom provider for Tinker sampling ([#458](https://github.com/thinking-machines-lab/tinker-cookbook/pull/458))
**Date:** 2026-03-14
**Type:** new
**Tags:** infrastructure

New LiteLLM custom provider that routes sampling requests through Tinker, allowing existing LiteLLM-based tooling (e.g., agent frameworks) to use Tinker-hosted models.

---

### [cookbook] CheckpointRecord dataclass for typed checkpoint bookkeeping ([#450](https://github.com/thinking-machines-lab/tinker-cookbook/pull/450))
**Date:** 2026-03-13
**Type:** new
**Tags:** infrastructure

New `CheckpointRecord` dataclass replaces ad-hoc dicts for tracking checkpoint metadata. Backward-compatible with external checkpoint formats ([#471](https://github.com/thinking-machines-lab/tinker-cookbook/pull/471)).

---

### [cookbook] Fix final checkpoint batch field that breaks resume ([#448](https://github.com/thinking-machines-lab/tinker-cookbook/pull/448))
**Date:** 2026-03-13
**Type:** fix
**Tags:** infrastructure

Fixed a bug where the batch field in the final checkpoint was incorrect, causing training resume to start from the wrong position.

---

### [cookbook] Fix checkpoint loading for DPO and on-policy distillation ([#446](https://github.com/thinking-machines-lab/tinker-cookbook/pull/446), [#447](https://github.com/thinking-machines-lab/tinker-cookbook/pull/447))
**Date:** 2026-03-13
**Type:** fix
**Tags:** supervised, rl

DPO and on-policy distillation checkpoint loading now matches SFT/RL behavior, properly restoring optimizer state and step count.

---

### [cookbook] Extract shared streaming parser and restructure renderer tests ([#431](https://github.com/thinking-machines-lab/tinker-cookbook/pull/431))
**Date:** 2026-03-13
**Type:** improvement
**Tags:** renderers, testing

Common streaming parsing logic extracted into a shared `StreamingParser`. Renderer tests reorganized by model family for better maintainability.

---

### [cookbook] Keep final SFT and RL checkpoints indefinitely ([#424](https://github.com/thinking-machines-lab/tinker-cookbook/pull/424))
**Date:** 2026-03-12
**Type:** improvement
**Tags:** infrastructure

The final checkpoint from a training run is now saved with no TTL expiry, ensuring trained weights aren't auto-deleted.

---

### [cookbook] Pluggable rollout executor via `concurrent.futures.Executor` ([#425](https://github.com/thinking-machines-lab/tinker-cookbook/pull/425))
**Date:** 2026-03-12
**Type:** new
**Tags:** rl

RL rollout execution can now use any `concurrent.futures.Executor`, enabling distributed rollout computation across multiple machines.

---

### [cookbook] Rollout summary JSONL and logtree JSON exports ([#389](https://github.com/thinking-machines-lab/tinker-cookbook/pull/389), [#428](https://github.com/thinking-machines-lab/tinker-cookbook/pull/428))
**Date:** 2026-03-12
**Type:** new
**Tags:** rl, logging

RL training now writes machine-readable `*_rollout_summaries.jsonl` (per-trajectory metadata) and `*_logtree.json` (full rollout transcripts) alongside the existing HTML reports.

---

### [cookbook] Pickle support for Renderer and env builders ([#422](https://github.com/thinking-machines-lab/tinker-cookbook/pull/422), [#423](https://github.com/thinking-machines-lab/tinker-cookbook/pull/423))
**Date:** 2026-03-11
**Type:** new
**Tags:** infrastructure

Renderers, `ChromaTool`, and `VerifiersEnvGroupBuilder` are now picklable, enabling distributed rollout execution with process pools.

---

### [cookbook] Multi-turn on-policy distillation for Harbor environments ([#411](https://github.com/thinking-machines-lab/tinker-cookbook/pull/411))
**Date:** 2026-03-10
**Type:** new
**Tags:** rl, recipes

On-policy distillation now supports multi-turn environments (e.g., Harbor terminal tasks), distilling teacher behavior across interactive trajectories.

---

### [cookbook] Standardize recipe entrypoints, log paths, and CLI config ([#405](https://github.com/thinking-machines-lab/tinker-cookbook/pull/405))
**Date:** 2026-03-09
**Type:** improvement
**Tags:** recipes

All recipes now share a consistent CLI config pattern, standardized log path structure, and unified entrypoint conventions.

---

### [cookbook] Fix TinkerMessageCompleter dropping tool_calls ([#403](https://github.com/thinking-machines-lab/tinker-cookbook/pull/403))
**Date:** 2026-03-09
**Type:** fix
**Tags:** renderers, tools

Fixed a bug where `TinkerMessageCompleter` was silently dropping tool calls from model responses.

---

### [cookbook] Qwen3.5 support ([#397](https://github.com/thinking-machines-lab/tinker-cookbook/pull/397))
**Date:** 2026-03-06
**Type:** new
**Tags:** models

Adds Qwen3.5 to the model lineup with renderer support, LR config, and LoRA merge compatibility.

---

### [cookbook] Harbor RL recipe for sandboxed terminal-bench training ([#377](https://github.com/thinking-machines-lab/tinker-cookbook/pull/377))
**Date:** 2026-03-04
**Type:** new
**Tags:** recipes, rl

New recipe for training agents on terminal tasks using Harbor sandboxed environments. Includes eval standardization ([#463](https://github.com/thinking-machines-lab/tinker-cookbook/pull/463)).

---

### [cookbook] Strip thinking from history for Kimi K2 and K2.5 renderers ([#384](https://github.com/thinking-machines-lab/tinker-cookbook/pull/384), [#393](https://github.com/thinking-machines-lab/tinker-cookbook/pull/393))
**Date:** 2026-03-01 to 2026-03-03
**Type:** new
**Tags:** renderers

Kimi K2 and K2.5 renderers now support `strip_thinking_from_history`, matching the existing Qwen3 option. Controls whether `<think>` blocks are preserved in multi-turn history.

---

### [cookbook] Persist renderer metadata on training runs ([#382](https://github.com/thinking-machines-lab/tinker-cookbook/pull/382))
**Date:** 2026-02-24
**Type:** improvement
**Tags:** infrastructure, eval

Training runs now save renderer metadata to the checkpoint. Evals auto-resolve the correct renderer from checkpoint metadata, eliminating manual renderer selection.

---

### [cookbook] Support Qwen3VL in adapter merge ([#360](https://github.com/thinking-machines-lab/tinker-cookbook/pull/360))
**Date:** 2026-02-23
**Type:** new
**Tags:** weights, models

`merge_tinker_adapter_to_hf_model` now supports Qwen3VL vision-language models, handling their unique weight structure during LoRA merge.

---

### [cookbook] ifBench RLVR recipe for instruction following ([#276](https://github.com/thinking-machines-lab/tinker-cookbook/pull/276))
**Date:** 2026-02-22
**Type:** new
**Tags:** recipes, rl

New recipe for RLVR training on instruction following benchmarks using the ifBench dataset.

---

### [cookbook] Fix empty token chunk causing 400 errors ([#376](https://github.com/thinking-machines-lab/tinker-cookbook/pull/376))
**Date:** 2026-02-19
**Type:** fix
**Tags:** supervised, rl

Fixed a bug where empty token chunks in model inputs would cause 400 errors from the Tinker API. Empty chunks are now filtered out before submission.

---

### [cookbook] Kimi K2.5 support ([#352](https://github.com/thinking-machines-lab/tinker-cookbook/pull/352), [#357](https://github.com/thinking-machines-lab/tinker-cookbook/pull/357), [#359](https://github.com/thinking-machines-lab/tinker-cookbook/pull/359))
**Date:** 2026-02-05 to 2026-02-10
**Type:** new
**Tags:** models, renderers

Adds Kimi K2.5 model family with text and vision rendering support.

---

### [cookbook] Library for training tool-use agents ([#311](https://github.com/thinking-machines-lab/tinker-cookbook/pull/311))
**Date:** 2026-02-05
**Type:** new
**Tags:** tools, rl

New library for training tool-use agents with structured tool calling, conversation management, and evaluation.

---

### [cookbook] Remove ToolCallPart/UnparsedToolCallPart from ContentPart ([#353](https://github.com/thinking-machines-lab/tinker-cookbook/pull/353))
**Date:** 2026-02-05
**Type:** improvement
**Tags:** renderers

**Breaking:** `ToolCallPart` and `UnparsedToolCallPart` are no longer part of the `ContentPart` union type. Tool calls now live exclusively in `message["tool_calls"]` / `message["unparsed_tool_calls"]`, simplifying content iteration.

---

### [cookbook] Custom renderer and tokenizer registration ([#349](https://github.com/thinking-machines-lab/tinker-cookbook/pull/349))
**Date:** 2026-02-05
**Type:** new
**Tags:** renderers

Users can now register custom renderers and tokenizers, enabling support for models not in the built-in lineup.

---

### [cookbook] Support structured content in ConversationFormatter ([#343](https://github.com/thinking-machines-lab/tinker-cookbook/pull/343))
**Date:** 2026-02-05
**Type:** improvement
**Tags:** renderers

`ConversationFormatter` now handles structured content (thinking parts, tool calls) alongside plain text.

---

### [cookbook] Temperature parameter for TinkerMessageCompleter ([#336](https://github.com/thinking-machines-lab/tinker-cookbook/pull/336))
**Date:** 2026-02-05
**Type:** new
**Tags:** rl, tools

`TinkerMessageCompleter` now accepts a `temperature` parameter, giving users control over sampling temperature during multi-turn RL and evaluation.

---

### [cookbook] Fix XSS vulnerability in logtree HTML ([#337](https://github.com/thinking-machines-lab/tinker-cookbook/pull/337))
**Date:** 2026-02-05
**Type:** fix
**Tags:** infrastructure

Fixed a cross-site scripting vulnerability in logtree HTML reports where model output could contain executable scripts.

---

### [cookbook] `build_supervised_examples` and `LAST_ASSISTANT_TURN` ([#341](https://github.com/thinking-machines-lab/tinker-cookbook/pull/341))
**Date:** 2026-02-03
**Type:** new
**Tags:** supervised, renderers

New `build_supervised_examples` (plural) helper generates multiple training examples from a single conversation by splitting at assistant turns. `LAST_ASSISTANT_TURN` trains only on the final assistant response.

---

### [cookbook] Streaming parsing for Kimi K2 renderer ([#319](https://github.com/thinking-machines-lab/tinker-cookbook/pull/319))
**Date:** 2026-01-30
**Type:** new
**Tags:** renderers

Kimi K2 renderer now supports streaming token-by-token parsing, matching the existing capability in Qwen3 and DeepSeek V3 renderers.

---

### [cookbook] Reuse KL reference client instead of recreating per minibatch ([#332](https://github.com/thinking-machines-lab/tinker-cookbook/pull/332))
**Date:** 2026-01-30
**Type:** fix
**Tags:** rl

DPO and RL training now reuse the KL penalty reference sampling client across minibatches instead of creating a new one each time, reducing overhead.

---

### [cookbook] Cap training steps with `max_step` parameter ([#328](https://github.com/thinking-machines-lab/tinker-cookbook/pull/328))
**Date:** 2026-01-28
**Type:** new
**Tags:** rl, supervised

Adds optional `max_step` config parameter to cap training steps in on-policy distillation. When set, trains for `min(max_step, dataset_length)`. Default `None` preserves existing behavior.

---

### [cookbook] Configurable KL penalty reference model ([#326](https://github.com/thinking-machines-lab/tinker-cookbook/pull/326))
**Date:** 2026-01-27
**Type:** new
**Tags:** rl

Makes the KL penalty reference model configurable in RL training. Users can now specify a different base model or a checkpoint for the KL penalty computation, rather than using the default.

---

### [cookbook] Checkpoints now have 7-day TTL by default ([#324](https://github.com/thinking-machines-lab/tinker-cookbook/pull/324))
**Date:** 2026-01-27
**Type:** improvement
**Tags:** infrastructure

Checkpoints are now set to auto-expire after 7 days by default, helping users avoid unexpected storage costs.

---

### [cookbook] Support for dedicated capacity ([#315](https://github.com/thinking-machines-lab/tinker-cookbook/pull/315))
**Date:** 2026-01-21
**Type:** new
**Tags:** infrastructure

Adds support for dedicated capacity in training configurations.

---

### [cookbook] Configurable loss function parameters with `loss_fn_config` ([#156](https://github.com/thinking-machines-lab/tinker-cookbook/pull/156))
**Date:** 2026-01-16
**Type:** new
**Tags:** rl, supervised

New `loss_fn_config` parameter allows passing additional configuration to loss functions (e.g., KL penalty coefficients, clipping thresholds) without changing the function signature.

---

### [cookbook] Modal sandbox backend for code execution ([#278](https://github.com/thinking-machines-lab/tinker-cookbook/pull/278), [#291](https://github.com/thinking-machines-lab/tinker-cookbook/pull/291), [#300](https://github.com/thinking-machines-lab/tinker-cookbook/pull/300), [#302](https://github.com/thinking-machines-lab/tinker-cookbook/pull/302))
**Date:** 2026-01-07 to 2026-01-15
**Type:** new
**Tags:** sandboxes, rl

Adds Modal as an alternative sandbox backend for code execution alongside SandboxFusion. Includes:
- `ModalSandbox` and `ModalSandboxPool` for managing sandboxes
- Warm pool maintenance with configurable timeouts
- Rate limiting to respect Modal account limits
- Async API calls for better performance
- Documentation for both sandbox backends

See `tinker_cookbook/sandbox/` for the new module structure.

---

### [cookbook] Fix streaming dataset batch skipping ([#295](https://github.com/thinking-machines-lab/tinker-cookbook/pull/295))
**Date:** 2026-01-19
**Type:** fix
**Tags:** supervised

HuggingFace's shuffle is deterministic, so batch skipping now works correctly with streaming datasets. Forward skipping through batches no longer causes data inconsistencies.

---

### [cookbook] Fix supervised metrics from OptimStepResponse ([#286](https://github.com/thinking-machines-lab/tinker-cookbook/pull/286))
**Date:** 2026-01-20
**Type:** fix
**Tags:** supervised

Previously, optimization metrics (like gradient norms) from `OptimStepResponse` were being dropped in `finish_batch`. Metrics are now properly captured and merged into the step's metrics dictionary.

---

### [cookbook] Adapter to base-model merge script ([#292](https://github.com/thinking-machines-lab/tinker-cookbook/pull/292))
**Date:** 2026-01-08
**Type:** new
**Tags:** tools

New script to merge LoRA/adapter weights back into the base model.

---

### [cookbook] Fix inspect_utils for list content from parse_response ([#299](https://github.com/thinking-machines-lab/tinker-cookbook/pull/299))
**Date:** 2026-01-12
**Type:** fix
**Tags:** eval

Fixed `inspect_utils.py` which assumed `parse_response` always returns string content. Renderers like `Qwen3Renderer` return list content (with `ThinkingPart`, `ToolCallPart`, etc.) when responses contain `<think>` or `<tool_call>` blocks. Now uses `renderers.get_text_content()` which handles both formats.

---

### [cookbook] Fix Kimi K2 and DeepSeek V3 renderer parsing ([#279](https://github.com/thinking-machines-lab/tinker-cookbook/pull/279), [#285](https://github.com/thinking-machines-lab/tinker-cookbook/pull/285))
**Date:** 2026-01-05 to 2026-01-07
**Type:** fix
**Tags:** renderers

Fixes tool declaration rendering for Kimi K2 and Qwen3 to match HuggingFace templates. Also fixes DeepSeekV3ThinkingRenderer to properly parse thinking traces via a round-trip test ensuring `build_supervised_example` and `parse_response` correspondence.

---

### [sdk] Torch is now an optional dependency ([#15](https://github.com/thinking-machines-lab/tinker/pull/15))
**Date:** 2026-01-20
**Type:** improvement
**Tags:** dependencies

Moves torch to an optional dependency in the SDK. Applications that don't need torch for training can now use the SDK without installing it. Import guards added to `training_client.py`.

---

### Major renderer overhaul: tool calling, structured content ([#220](https://github.com/thinking-machines-lab/tinker-cookbook/pull/220), [#221](https://github.com/thinking-machines-lab/tinker-cookbook/pull/221), [#238](https://github.com/thinking-machines-lab/tinker-cookbook/pull/238), [#243](https://github.com/thinking-machines-lab/tinker-cookbook/pull/243), [#244](https://github.com/thinking-machines-lab/tinker-cookbook/pull/244), [#250](https://github.com/thinking-machines-lab/tinker-cookbook/pull/250))
**Date:** 2025-12-26 to 2025-12-28
**Type:** improvement
**Tags:** renderers, rl

A series of PRs that significantly improve the renderer system:

**Tool calling support:** New `ToolSpec` type for defining tools and `create_conversation_prefix_with_tools()` API on all renderers. Tool call parsing supported for Qwen3, DeepSeek V3, and Kimi K2. `UnparsedToolCall` captures tool calls that fail to parse.

**Structured message content:** The `Message.thinking` field is removed (**breaking**). Thinking content is now represented as `ThinkingPart` in the content list, alongside `TextPart`, `ImagePart`, and `ToolCallPart`. Use `get_text_content(message)` to extract text after `parse_response`.

**Clearer field names:** `RenderedMessage` fields renamed (**breaking**): `prefix` → `header`, `content` → `output`, `suffix` → `stop_overlap`. `Renderer` changed from Protocol to ABC.

**Sequence extension property:** New `has_extension_property` on `Renderer` indicates whether consecutive timesteps can be merged for O(T) instead of O(T²) compute in multi-turn RL.

**Modular architecture:** `renderers.py` split into `tinker_cookbook/renderers/` package with per-model modules (`qwen3.py`, `deepseek_v3.py`, `kimi_k2.py`, etc.). Imports unchanged.

**HF compatibility:** Various fixes to match HuggingFace chat templates, with expanded test coverage using random conversation generation.

---

### Qwen3 thinking blocks can now be preserved in history ([#142](https://github.com/thinking-machines-lab/tinker-cookbook/pull/142))
**Date:** 2025-12-06
**Type:** new
**Tags:** renderers, rl

The Qwen3Renderer now has a `strip_thinking_from_history` option. By default (`True`), it strips `<think>...</think>` blocks from previous assistant turns—matching how Qwen3 was trained. Set it to `False` if you're doing multi-turn RL and want to use sequence extension: preserving thinking lets turns merge into one sequence, reducing compute cost.

---

### Disable checkpoint saving with `save_every=0` ([#149](https://github.com/thinking-machines-lab/tinker-cookbook/pull/149))
**Date:** 2025-12-06
**Type:** improvement
**Tags:** supervised, rl

Setting `save_every=0` now disables checkpoint saving entirely (previously it crashed with a divide-by-zero). Useful for quick test runs where you don't need checkpoints.

---

### xmux: launch experiment sweeps in tmux ([#138](https://github.com/thinking-machines-lab/tinker-cookbook/pull/138))
**Date:** 2025-12-02
**Type:** new
**Tags:** tools

New `xmux` utility for running experiment sweeps. It spawns parallel jobs in a tmux session where you can monitor each experiment's progress in separate windows. See `tinker_cookbook/xmux/examples/` for usage.

---

### Optimizer state now loads correctly on resume ([#140](https://github.com/thinking-machines-lab/tinker-cookbook/pull/140), [#141](https://github.com/thinking-machines-lab/tinker-cookbook/pull/141))
**Date:** 2025-12-02
**Type:** fix
**Tags:** supervised, rl

Training resumption now properly loads optimizer state (momentum, etc.) alongside model weights. Previously, `load_state()` didn't restore the optimizer, which could affect training dynamics after a checkpoint resume.

---

### Tracing support for supervised training ([#88](https://github.com/thinking-machines-lab/tinker-cookbook/pull/88))
**Date:** 2025-11-21
**Type:** new
**Tags:** supervised, tools

Set `enable_trace=True` to generate trace events during supervised training. Visualize with Perfetto to see where time is spent. Run `python -m tinker_cookbook.utils.trace` to convert the trace file.

---

### Code RL recipe with DeepCoder ([#83](https://github.com/thinking-machines-lab/tinker-cookbook/pull/83))
**Date:** 2025-11-18
**Type:** new
**Tags:** recipes, rl

New recipe for RL on competitive programming problems using the DeepCoder dataset. Code execution is sandboxed via Sandbox Fusion. See `tinker_cookbook/recipes/code_rl/`.

---

### Configurable temperature for RL sampling ([#86](https://github.com/thinking-machines-lab/tinker-cookbook/pull/86))
**Date:** 2025-11-17
**Type:** new
**Tags:** rl

Temperature is now a configurable parameter in RL configs. Previously hardcoded to 1.0.

---

### Per-message training control with `TrainOnWhat.CUSTOMIZED` ([#85](https://github.com/thinking-machines-lab/tinker-cookbook/pull/85))
**Date:** 2025-11-14
**Type:** new
**Tags:** supervised, renderers

New `TrainOnWhat.CUSTOMIZED` option lets you set a `trainable: bool` field on each message to control which messages get loss applied. Useful for training on specific turns in a conversation.

---

### Interactive environment debugging with `play_w_env` ([#76](https://github.com/thinking-machines-lab/tinker-cookbook/pull/76))
**Date:** 2025-11-07
**Type:** new
**Tags:** rl, tools

New utility to "role-play" as the policy and interact with an Environment. Useful for debugging reward functions and environment logic. See `tinker_cookbook/recipes/multiplayer_rl/twenty_questions/play.py` for an example.

---
