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
