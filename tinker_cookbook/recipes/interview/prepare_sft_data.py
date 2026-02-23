"""
Post-process DeepMath samples to create SFT data with interleaved progress tool calls.

Takes sampled thinking traces and splits them into ~500 token chunks, then uses the
model to generate brief progress summaries for each chunk. The result is training data
where the model learns to emit <tool_call> blocks (calling notify_user) interleaved
with its <think> blocks, giving users progress updates during long reasoning.

Usage:
    python -m tinker_cookbook.recipes.interview.prepare_sft_data
"""

import asyncio
import json
import logging
from pathlib import Path

import tinker
from dotenv import load_dotenv

from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import Message, ToolSpec
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
INPUT_PATH = Path("/tmp/tinker-examples/interview/deepmath_samples.json")
OUTPUT_PATH = Path("/tmp/tinker-examples/interview/deepmath_sft_data.json")
CHUNK_TARGET_TOKENS = 500

NOTIFY_USER_TOOL: ToolSpec = {
    "name": "notify_user",
    "description": "Send a brief progress update to the user about what you're currently working on.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "A brief, user-friendly progress update (1-2 sentences)",
            }
        },
        "required": ["message"],
    },
}


def split_thinking_into_chunks(thinking: str, tokenizer) -> list[str]:
    """Split a thinking trace into chunks of ~CHUNK_TARGET_TOKENS tokens at sentence boundaries.

    Walks forward to the target token count, then scans for the next sentence boundary
    ('. ', '.\\n', or '\\n\\n'). Returns a list of text chunks.
    """
    tokens = tokenizer.encode(thinking, add_special_tokens=False)
    if len(tokens) <= CHUNK_TARGET_TOKENS:
        return [thinking]

    chunks: list[str] = []
    consumed = 0  # tokens consumed so far

    while consumed < len(tokens):
        remaining = len(tokens) - consumed
        if remaining <= int(CHUNK_TARGET_TOKENS * 1.3):
            # Last chunk — take everything remaining
            chunk_text = tokenizer.decode(tokens[consumed:])
            chunks.append(chunk_text)
            break

        # Decode from current position to target + some overshoot to find a boundary
        search_end = min(consumed + CHUNK_TARGET_TOKENS + 200, len(tokens))
        candidate_text = tokenizer.decode(tokens[consumed:search_end])

        # Find sentence boundary after the target length
        target_text = tokenizer.decode(tokens[consumed : consumed + CHUNK_TARGET_TOKENS])
        search_start = len(target_text)

        best_break = -1
        for boundary in [". ", ".\n", "\n\n"]:
            pos = candidate_text.find(boundary, search_start)
            if pos != -1:
                # Include the boundary delimiter in this chunk
                break_pos = pos + len(boundary)
                if best_break == -1 or break_pos < best_break:
                    best_break = break_pos

        if best_break == -1:
            # No sentence boundary found; fall back to the target length
            chunk_text = target_text
        else:
            chunk_text = candidate_text[:best_break]

        chunks.append(chunk_text)
        # Figure out how many tokens we actually consumed
        chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
        consumed += len(chunk_tokens)

    return chunks


async def generate_summaries(
    chunks: list[str],
    sampling_client,
    summary_renderer: renderers.Renderer,
) -> list[str]:
    """Generate progress summaries for a list of thinking chunks.

    Prompts the model to write a 1-2 sentence user-friendly progress update for each
    chunk. All requests fire in parallel. Uses a non-thinking renderer so the model
    produces direct answers without <think> blocks.
    """
    stop_sequences = summary_renderer.get_stop_sequences()
    sample_params = tinker.SamplingParams(
        max_tokens=100,
        temperature=0.3,
        stop=stop_sequences,
    )

    async def summarize_one(chunk: str) -> str:
        messages: list[Message] = [
            {
                "role": "user",
                "content": (
                    "Below is a chunk of mathematical reasoning from a model's thinking trace. "
                    "Summarize in 1-2 sentences what this reasoning is working on, as a brief "
                    "user-friendly progress update. Do NOT use LaTeX. Be concise.\n\n"
                    f"---\n{chunk}\n---"
                ),
            },
        ]
        prompt = summary_renderer.build_generation_prompt(messages)
        result = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=sample_params,
        )
        response_tokens = result.sequences[0].tokens
        parsed, _ = summary_renderer.parse_response(response_tokens)
        content = parsed["content"]
        if isinstance(content, list):
            return "".join(part["text"] for part in content if part["type"] == "text").strip()
        return str(content).strip()

    summaries = await asyncio.gather(*[summarize_one(chunk) for chunk in chunks])
    return list(summaries)


def assemble_conversation(
    question: str,
    thinking_chunks: list[str],
    summaries: list[str],
    visible_response: str,
    renderer: renderers.Renderer,
) -> list[Message]:
    """Assemble a conversation with interleaved <think> and <tool_call> blocks.

    Returns a list of messages: system (with tool spec), user, and a single assistant
    message containing the interleaved content.
    """
    # System message with tool declarations
    prefix_messages = renderer.create_conversation_prefix_with_tools(
        tools=[NOTIFY_USER_TOOL],
    )

    # User message
    user_message: Message = {
        "role": "user",
        "content": question + " Write your answer in \\boxed{} format.",
    }

    # Build assistant content with interleaved <think> and <tool_call> blocks
    assistant_parts: list[str] = []
    for i, chunk in enumerate(thinking_chunks):
        assistant_parts.append(f"<think>{chunk}</think>")
        # Add a tool call after each chunk except the last
        if i < len(thinking_chunks) - 1 and i < len(summaries):
            tool_call_json = json.dumps(
                {"name": "notify_user", "arguments": {"message": summaries[i]}},
            )
            assistant_parts.append(f"\n<tool_call>\n{tool_call_json}\n</tool_call>\n")

    # Append the visible response after the final think block
    assistant_parts.append(visible_response)

    assistant_message: Message = {
        "role": "assistant",
        "content": "".join(assistant_parts),
    }

    return prefix_messages + [user_message, assistant_message]


def message_to_serializable(msg: Message) -> dict:
    """Convert a Message TypedDict to a plain dict for JSON serialization."""
    result: dict = {"role": msg["role"], "content": msg["content"]}
    for key in ("tool_calls", "unparsed_tool_calls", "trainable", "tool_call_id", "name"):
        if key in msg:
            result[key] = msg[key]  # type: ignore[literal-required]
    return result


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Load samples
    logger.info(f"Loading samples from {INPUT_PATH}...")
    with open(INPUT_PATH) as f:
        samples = json.load(f)
    logger.info(f"Loaded {len(samples)} samples")

    # Set up renderers and tokenizer
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    # Non-thinking renderer for summary generation (avoids <think> blocks in summaries)
    summary_renderer = renderers.get_renderer("qwen3_disable_thinking", tokenizer=tokenizer)

    # Create sampling client for generating summaries
    logger.info(f"Creating sampling client for {MODEL_NAME}...")
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)

    conversations = []
    for i, sample in enumerate(samples):
        thinking = sample["thinking"]
        visible_response = sample["response"]
        question = sample["question"]

        # Split thinking into chunks
        chunks = split_thinking_into_chunks(thinking, tokenizer)
        logger.info(
            f"[{i + 1}/{len(samples)}] Split thinking ({len(thinking)} chars) "
            f"into {len(chunks)} chunks"
        )

        # Generate progress summaries (only for non-final chunks)
        chunks_to_summarize = chunks[:-1] if len(chunks) > 1 else []
        if chunks_to_summarize:
            summaries = await generate_summaries(
                chunks_to_summarize, sampling_client, summary_renderer
            )
            logger.info(f"  Generated {len(summaries)} progress summaries")
        else:
            summaries = []

        # Assemble conversation
        conversation = assemble_conversation(
            question=question,
            thinking_chunks=chunks,
            summaries=summaries,
            visible_response=visible_response,
            renderer=renderer,
        )
        conversations.append(
            {
                "index": i,
                "question": question,
                "num_think_chunks": len(chunks),
                "num_summaries": len(summaries),
                "messages": [message_to_serializable(m) for m in conversation],
            }
        )

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(conversations, f, indent=2)
    logger.info(f"Saved {len(conversations)} conversations to {OUTPUT_PATH}")

    # Print summary statistics
    for conv in conversations:
        print(
            f"  [{conv['index']}] {conv['num_think_chunks']} chunks, "
            f"{conv['num_summaries']} summaries, "
            f"{len(conv['messages'])} messages"
        )


if __name__ == "__main__":
    asyncio.run(main())
