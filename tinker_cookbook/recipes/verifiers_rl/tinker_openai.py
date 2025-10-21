"""
OpenAI-compatible client backed by Tinker sampling.

Implements OpenAI client semantics for:
- chat.completions.create(...)
- completions.create(...)

Returns OpenAI types (ChatCompletion / Completion) constructed from sampled tokens.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, overload, Literal

import tinker
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from openai import AsyncOpenAI
from openai.resources.chat import AsyncChat as OpenAIAsyncChat
from openai.resources.chat.completions import AsyncCompletions as OpenAIAsyncChatCompletions
from openai.resources.completions import AsyncCompletions as OpenAIAsyncCompletions
from openai._streaming import AsyncStream

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import Tokenizer


GenerationHook = Callable[
    [List[renderers.Message], tinker.ModelInput, List[int], List[float]], None
]


def convert_oai_messages_to_renderer_messages(
    messages: List[Dict[str, Any]],
) -> List[renderers.Message]:
    out: List[renderers.Message] = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        # extract text from list of content parts if necessary
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    if "text" in part:
                        text_parts.append(str(part["text"]))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "".join(text_parts)
        else:
            content = str(content)
        out.append(renderers.Message(role=role, content=content))
    return out


class TinkerAsyncOpenAIClient(AsyncOpenAI):
    """
    OpenAI-compatible async client that routes calls to a Tinker SamplingClient.
    """

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        tokenizer: Tokenizer,
    ) -> None:
        super().__init__(api_key="tinker", base_url="http://localhost")
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.hook: Optional[GenerationHook] = None

    def set_generation_hook(self, hook: Optional[GenerationHook]) -> None:
        self.hook = hook

    def set_sampling_client(self, sampling_client: tinker.SamplingClient) -> None:
        self.sampling_client = sampling_client

    @property
    def chat(self) -> OpenAIAsyncChat:
        return TinkerAsyncChat(self)

    @property
    def completions(self) -> OpenAIAsyncCompletions:
        return TinkerCompletions(self)


class TinkerChatCompletions(OpenAIAsyncChatCompletions):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @overload
    async def create(
        self, *args: Any, stream: Literal[True], **kwargs: Any
    ) -> AsyncStream[Any]: ...

    @overload
    async def create(
        self, *args: Any, stream: Literal[False] = False, **kwargs: Any
    ) -> ChatCompletion: ...

    @overload
    async def create(self, *args: Any, stream: bool, **kwargs: Any) -> ChatCompletion: ...

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion | AsyncStream[Any]:
        model = kwargs.get("model", "tinker")
        messages = kwargs.get("messages", [])
        if kwargs.get("stream", False):
            raise ValueError("stream=True not supported by TinkerAsyncOpenAIClient")
        sampling_args = {k: v for k, v in kwargs.items() if k not in ("model", "messages", "tools")}

        # prepare prompt
        conv_messages = convert_oai_messages_to_renderer_messages(messages)
        stop = sampling_args.get("stop", self._parent.renderer.get_stop_sequences())
        max_tokens = sampling_args.get("max_tokens") or sampling_args.get("max_completion_tokens")

        model_input = self._parent.renderer.build_generation_prompt(conv_messages)
        sample = await self._parent.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=float(sampling_args.get("temperature", 1.0)),
                max_tokens=int(max_tokens or 128),
                top_p=float(sampling_args.get("top_p", 1.0)),
                top_k=int(sampling_args.get("top_k", -1)),
                stop=stop,
            ),
        )
        seq = sample.sequences[0]
        tokens: List[int] = seq.tokens
        logprobs: List[float] = seq.logprobs or [0.0] * len(tokens)

        if self._parent.hook is not None:
            self._parent.hook(conv_messages, model_input, tokens, logprobs)

        # build ChatCompletion via pydantic validation using renderer parsing
        assistant_message, parse_success = self._parent.renderer.parse_response(tokens)
        content_text = assistant_message["content"]
        finish_reason = "stop" if parse_success else "length"
        response_dict: Dict[str, Any] = {
            "id": "tinker-chatcmpl",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content_text},
                    "finish_reason": finish_reason,
                    "logprobs": {
                        "content": [
                            {"token": f"token_id:{tid}", "logprob": float(lp), "top_logprobs": []}
                            for tid, lp in zip(tokens, logprobs)
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": model_input.length,
                "completion_tokens": len(tokens),
                "total_tokens": model_input.length + len(tokens),
            },
        }
        return ChatCompletion.model_validate(response_dict)


class TinkerCompletions(OpenAIAsyncCompletions):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @overload
    async def create(
        self, *args: Any, stream: Literal[True], **kwargs: Any
    ) -> AsyncStream[Completion]: ...

    @overload
    async def create(
        self, *args: Any, stream: Literal[False] = False, **kwargs: Any
    ) -> Completion: ...

    @overload
    async def create(
        self, *args: Any, stream: bool, **kwargs: Any
    ) -> Completion | AsyncStream[Completion]: ...

    async def create(self, *args: Any, **kwargs: Any) -> Completion | AsyncStream[Completion]:
        stream = bool(kwargs.get("stream", False))
        model = kwargs.get("model", "tinker")
        prompt = kwargs.get("prompt", "")
        sampling_args = {k: v for k, v in kwargs.items() if k not in ("model", "prompt")}

        # Completion-mode: render prompt directly as text chunk
        model_input = tinker.ModelInput.from_ints(
            self._parent.tokenizer.encode(prompt, add_special_tokens=True)
        )
        sample = await self._parent.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=float(sampling_args.get("temperature", 1.0)),
                max_tokens=int(sampling_args.get("max_tokens", 128)),
                top_p=float(sampling_args.get("top_p", 1.0)),
                top_k=int(sampling_args.get("top_k", -1)),
            ),
        )
        seq = sample.sequences[0]
        tokens: List[int] = seq.tokens
        logprobs: List[float] = seq.logprobs or [0.0] * len(tokens)

        text = self._parent.tokenizer.decode(tokens)
        tokens_str = [f"token_id:{tid}" for tid in tokens]
        response_dict: Dict[str, Any] = {
            "id": "tinker-cmpl",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                    "logprobs": {
                        "tokens": tokens_str,
                        "token_logprobs": [float(lp) for lp in logprobs],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": model_input.length,
                "completion_tokens": len(tokens),
                "total_tokens": model_input.length + len(tokens),
            },
        }
        final = Completion.model_validate(response_dict)
        if stream:
            return TinkerAsyncCompletionStream(final)
        return final


class TinkerAsyncChat(OpenAIAsyncChat):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @property
    def completions(self) -> OpenAIAsyncChatCompletions:
        return TinkerChatCompletions(self._parent)


class TinkerAsyncCompletionStream(AsyncStream[Completion]):
    def __init__(self, final: Completion) -> None:
        self._final = final

    def __aiter__(self):
        self._done = True
        return self

    async def __anext__(self) -> Completion:
        raise StopAsyncIteration

    def __await__(self):
        async def _await_final():
            return self._final

        return _await_final().__await__()

    async def get_final_response(self) -> Completion:
        return self._final
