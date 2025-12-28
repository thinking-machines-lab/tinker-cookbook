"""Renderer for Llama 3 chat format."""

import json

import tinker

from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    ensure_text,
    parse_response_for_stop_token,
)


class Llama3Renderer(Renderer):
    """Renderer for Llama 3 Instruct models.

    Format::

        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

        What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Tool calls use JSON format: {"name": "func", "parameters": {"arg": "value"}}

    Note: We intentionally differ from HF's stock Llama template:

    - HF prepends "Cutting Knowledge Date..." to system messages; we don't
      (add manually if needed)
    - HF drops assistant content when tool_calls are present; we preserve it
    - HF double-encodes tool args via |tojson; we use clean single-encoding

    These differences are intentional - the stock Llama format has quirks not
    worth matching. Our format works with vLLM's Llama tool parser which accepts
    both single and double encoding.
    """

    @property
    def has_extension_property(self) -> bool:
        """Llama3 satisfies the extension property - no content is stripped from history."""
        return True

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        # Determine role for header
        # Tool responses use "ipython" role in Llama 3 format
        original_role = message["role"]
        role = "ipython" if original_role == "tool" else original_role

        header_str = f"<|start_header_id|>{role}<|end_header_id|>\n\n"

        # Build output content
        content = ensure_text(message["content"])

        # Tool results are wrapped in {"output": ...} to match vLLM's Llama template
        if original_role == "tool":
            output_str = json.dumps({"output": content})
        else:
            output_str = content

        # Handle tool calls in assistant messages
        # JSON format: {"name": "func_name", "parameters": {"arg": "value"}}
        # Note: HF template double-encodes args via |tojson, but we use single-encoding
        # which is cleaner and still works with vllm's parser.
        if "tool_calls" in message and message["tool_calls"]:
            tool_call_strs = []
            for tool_call in message["tool_calls"]:
                func_name = tool_call.function.name
                args = tool_call.function.arguments  # Already a JSON string
                tool_call_strs.append(f'{{"name": "{func_name}", "parameters": {args}}}')
            output_str += "".join(tool_call_strs)

        output_str += "<|eot_id|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_str, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        (token,) = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        return token

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def _parse_llama_tool_calls(
        self, content: str
    ) -> tuple[list[ToolCall], list[UnparsedToolCall], str]:
        """Parse tool calls from Llama 3 JSON format.

        Format: {"name": "func_name", "parameters": {"arg": "value"}}

        Known limitation: This parser treats ANY JSON object with "name" and
        "parameters"/"arguments" keys as a tool call, even if it's regular assistant
        content (e.g., user asked for a JSON schema). This can incorrectly strip
        JSON from content and turn it into a tool call. We require parameters to be
        a dict to mitigate this, but the ambiguity remains.

        We're leaving this behavior as-is because we don't know if anyone is using
        Llama 3 for serious tool calling work. If this becomes an issue, consider
        adding a delimiter/marker that the model is instructed to use, e.g.,
        <tool_call>...</tool_call>.

        Returns:
            Tuple of (successfully parsed tool calls, failed parses, remaining content).
        """
        tool_calls: list[ToolCall] = []
        unparsed_tool_calls: list[UnparsedToolCall] = []
        remaining_content = content

        # Use JSON decoder to find and parse tool call objects
        decoder = json.JSONDecoder()
        search_start = 0

        while search_start < len(content):
            brace_pos = content.find("{", search_start)
            if brace_pos == -1:
                break

            try:
                obj, end_idx = decoder.raw_decode(content[brace_pos:])
                raw_text = content[brace_pos : brace_pos + end_idx]

                # Check if it's a tool call (has name and parameters/arguments)
                # We require parameters to be a dict to avoid treating arbitrary JSON
                # (e.g., user asking for a JSON schema) as tool calls.
                if "name" in obj and ("parameters" in obj or "arguments" in obj):
                    func_name = obj["name"]
                    params = obj.get("parameters") or obj.get("arguments")

                    if isinstance(params, dict):
                        tool_calls.append(
                            ToolCall(
                                function=ToolCall.FunctionBody(
                                    name=func_name, arguments=json.dumps(params)
                                ),
                            )
                        )
                        remaining_content = remaining_content.replace(raw_text, "", 1)
                    # If params is not a dict, leave the JSON in content (don't treat as tool call)

                search_start = brace_pos + end_idx
            except (json.JSONDecodeError, KeyError):
                search_start = brace_pos + 1

        return tool_calls, unparsed_tool_calls, remaining_content.strip()

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"]

        # Parse tool calls and get remaining content with tool calls stripped
        tool_calls, unparsed_tool_calls, remaining_content = self._parse_llama_tool_calls(content)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if unparsed_tool_calls:
            assistant_message["unparsed_tool_calls"] = unparsed_tool_calls

        # Update content to remaining content (tool calls stripped)
        if tool_calls or unparsed_tool_calls:
            assistant_message["content"] = remaining_content

        return assistant_message, True

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create system message with Llama 3 tool specifications.

        Note: HF's Llama template puts tool instructions in the user message, not
        system. We use a system message for simplicity. The JSON format matches
        what HF expects: {"name": function name, "parameters": dictionary of args}.
        """
        tools_text = ""
        if tools:
            tool_lines = "\n\n".join(json.dumps(tool, indent=2) for tool in tools)
            # Wording matches HF template's tool instruction
            tools_text = f"""You have access to the following functions. To call a function, please respond with JSON for a function call.

Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}.
Do not use variables.

{tool_lines}

"""

        return [Message(role="system", content=tools_text + system_prompt)]
