# Plan: Upgrade parse_response to Extract ThinkingPart

## Overview

This document outlines a plan to update all `parse_response` methods to extract thinking content into `ThinkingPart` elements, providing a consistent structured representation across all thinking-capable renderers.

## Current State

| Renderer | Thinking Format | parse_response Behavior |
|----------|----------------|------------------------|
| KimiK2Renderer | `<think>...</think>` | ✅ Extracts to `[ThinkingPart, TextPart]` |
| Qwen3Renderer | `<think>...</think>` | ❌ Returns string with tags embedded |
| DeepSeekV3ThinkingRenderer | `<think>...</think>` | ❌ Returns string with tags embedded |
| GptOssRenderer | `<\|channel\|>analysis<\|message\|>...<\|end\|>` | ❌ Returns raw string |
| Qwen3DisableThinkingRenderer | N/A (no thinking) | N/A |
| DeepSeekV3DisableThinkingRenderer | N/A (no thinking) | N/A |
| RoleColonRenderer | N/A | N/A |
| Llama3Renderer | N/A | N/A |

## Goal

All thinking-capable renderers should return `Message` with content as `list[ContentPart]` when thinking is present:
```python
{
    "role": "assistant",
    "content": [
        ThinkingPart(type="thinking", thinking="reasoning here..."),
        TextPart(type="text", text="visible response"),
    ]
}
```

## Renderers to Update

### 1. Qwen3Renderer

**Current format:** `<think>reasoning</think>visible response`

**Parsing logic:**
```python
think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
if think_match:
    thinking = think_match.group(1)
    remaining = content[think_match.end():].lstrip()
    # Build list with ThinkingPart + TextPart
```

**Edge cases:**
- Empty `<think></think>` block → omit ThinkingPart, return just TextPart
- No thinking block → return string content unchanged (backward compatible)
- Multiple `<think>` blocks → only first one is thinking (rest is in text) 
    JDS: no, in this case use multiple ThinkingPart and TextPart blocks.
- Tool calls after thinking → handle tool call parsing on remaining content
    JDS: in general, avoid making assumptions about the ordering of tool calling, thinking, and text parts.

### 2. DeepSeekV3ThinkingRenderer

**Current format:** `<think>reasoning</think>visible response`

**Parsing logic:** Same as Qwen3

**Edge cases:** Same as Qwen3, plus:
- Tool calls use `<｜tool▁calls▁begin｜>` format
- Need to strip tool calls from content after extracting thinking
    JDS: follow same guidelines as Qwen.

### 3. GptOssRenderer

**Format (from Harmony spec):** GptOss uses a multi-message protocol where a single completion can contain MULTIPLE messages with different channels:

```
<|channel|>analysis<|message|>reasoning step 1<|end|>
<|start|>assistant<|channel|>analysis<|message|>reasoning step 2<|end|>
<|start|>assistant<|channel|>final<|message|>visible response<|return|>
```

**Channels:**
- `analysis` - Chain-of-thought (CoT), NOT shown to users
- `commentary` - Tool calls (`to=functions.x`) and user-visible preambles
- `final` - User-facing answer

**Key complexity:** Unlike Qwen3/DeepSeek which have a single `<think>...</think>` block, GptOss can have:
- Multiple `analysis` messages (interleaved reasoning)
- `commentary` messages for tool calls
- A `final` message for visible output

**Parsing logic:**
```python
# Parse all messages from the completion
# Each message: <|channel|>{channel}<|message|>{content}<|end|> or <|return|>/<|call|>

# Collect all analysis content → ThinkingPart
# Collect final content → TextPart
# Collect commentary with to=functions.x → tool_calls

import re

messages = []
# Pattern for each message segment
pattern = r"<\|channel\|>(\w+)(?:\s+to=([^\s<]+))?(?:<\|constrain\|>\w+)?<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|call\|>)"

for match in re.finditer(pattern, content, re.DOTALL):
    channel = match.group(1)
    recipient = match.group(2)  # e.g., "functions.get_weather"
    msg_content = match.group(3)
    messages.append((channel, recipient, msg_content))

# Aggregate by channel
analysis_parts = [m[2] for m in messages if m[0] == "analysis"]
final_parts = [m[2] for m in messages if m[0] == "final"]
tool_calls = [(m[1], m[2]) for m in messages if m[0] == "commentary" and m[1] and m[1].startswith("functions.")]

# Build content list
content_parts = []
if analysis_parts:
    combined_thinking = "\n".join(analysis_parts)
    content_parts.append(ThinkingPart(type="thinking", thinking=combined_thinking))
if final_parts:
    content_parts.append(TextPart(type="text", text=final_parts[-1]))  # Use last final
```

**Edge cases:**
- Multiple `analysis` messages → concatenate into single ThinkingPart
    JDS: no, you can return multiple blocks in whatever order you want.
- No `analysis` messages → no ThinkingPart
    JDS: right
- `commentary` without `to=functions.x` → user-visible preamble (include in text?)
- Stop on `<|call|>` → tool call, not `<|return|>`

**Open question:** How to handle `commentary` preambles (user-visible planning text before tool calls)? Options:
- Include in TextPart before tool call
- Add a new ContentPart type for preambles
- Ignore for now (focus on analysis/final)
    JDS: just return a bunch of parts in whatever order they appear. Let me know if we need new ContentPart types.

## Implementation Plan

### Phase 1: Update Qwen3Renderer.parse_response

```python
def parse_response(self, response: list[int]) -> tuple[Message, bool]:
    assistant_message, parse_success = parse_response_for_stop_token(
        response, self.tokenizer, self._end_message_token
    )
    if not parse_success:
        return assistant_message, False

    content = assistant_message["content"]
    assert isinstance(content, str)

    # Extract thinking if present
    think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if think_match:
        thinking = think_match.group(1)
        remaining_content = content[think_match.end():].lstrip()

        # Build content list
        content_parts: list[ContentPart] = []
        if thinking:  # Omit empty thinking
            content_parts.append(ThinkingPart(type="thinking", thinking=thinking))
        content_parts.append(TextPart(type="text", text=remaining_content))
        assistant_message["content"] = content_parts
        content = remaining_content  # For tool call parsing below

    # ... rest of tool call parsing unchanged ...
```

### Phase 2: Update DeepSeekV3ThinkingRenderer.parse_response

Same pattern as Qwen3, but tool call stripping happens after thinking extraction.

### Phase 3: Update GptOssRenderer.parse_response

More complex due to channel-based format. Need to:
1. Parse both analysis and final channels
2. Build ThinkingPart from analysis content
3. Build TextPart from final content

### Phase 4: Add/Update Tests

Add tests for each renderer:
- `test_qwen3_parse_response_extracts_thinking`
- `test_deepseek_parse_response_extracts_thinking`
- `test_gpt_oss_parse_response_extracts_thinking`
- `test_parse_response_no_thinking_returns_string` (backward compat)
- `test_parse_response_empty_thinking_omitted`

## Questions for Review

1. **Backward compatibility:** Should we always return list content when parsing, or only when thinking is present?
   - Option A: Always return list (consistent but breaking)
   - Option B: Only return list when thinking present (less breaking)

   **Recommendation:** Option B - only return list when thinking is present. This minimizes breaking changes for code that expects string content from non-thinking responses.

   JDS: Agree

2. **Tool calls with thinking:** When both thinking and tool calls are present, should the TextPart contain the tool call tags, or should they be stripped?

   **Current behavior (KimiK2):** Tool calls are stripped from content and put in `tool_calls` field. TextPart contains only non-tool-call text.

   **Recommendation:** Follow KimiK2's pattern - strip tool calls from TextPart.

   JDS: agree

3. **Multiple thinking blocks:** What if the model outputs multiple `<think>` blocks?

   **Recommendation:** Only treat the first `<think>...</think>` as thinking. Any subsequent blocks are part of the visible text (likely a model error).

   JDS: as I said above, let's just parse things into a bunch of parts in whatever order they appear.

## Risk Assessment

**Low risk:**
- Qwen3Renderer changes (clear format, similar to KimiK2)
- DeepSeekV3ThinkingRenderer changes (same format as Qwen3)

**High risk:**
- GptOssRenderer changes (complex multi-message protocol, significantly different from other renderers)

**Low impact:**
- Non-thinking renderers unchanged
- Disable-thinking variants unchanged

## Implementation Order

1. **Qwen3Renderer** (most common, clear format, matches KimiK2 pattern)
2. **DeepSeekV3ThinkingRenderer** (same `<think>` format as Qwen3)
3. Tests for Qwen3 and DeepSeek
4. **GptOssRenderer** (complex multi-message protocol - consider deferring)
   - Significantly more complex than other renderers
   - Requires parsing multiple messages with different channels
   - May want to implement as a separate PR

## Estimated Changes

**Qwen3 + DeepSeek (straightforward):**
- `renderers.py`: ~40 lines added/modified
- `test_renderers.py`: ~60 lines added (new tests)

**GptOss (complex, if included):**
- `renderers.py`: ~80 additional lines (multi-message parsing)
- `test_renderers.py`: ~40 additional lines
- Consider helper function for Harmony message parsing


JDS: start out with Qwen3, then I'll review, before you proceed to the rest. You might need to write some utilities to do parsing, as this is obviously more complicated than what's in the current code.