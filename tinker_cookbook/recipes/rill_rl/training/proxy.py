"""An OpenAI-compatible sampling proxy that records tokens, backed by a Tinker policy.

This is the integration seam between the production agent app and RL training. The app
points ``OPENAI_BASE_URL`` at this proxy; the proxy renders the chat request with the
model's renderer, samples from the current Tinker ``SamplingClient``, returns an
OpenAI ``ChatCompletion``, and **records the exact prompt tokens, sampled tokens, and
logprobs** so the trainer can build training data from real on-policy rollouts.

Requests are routed per rollout: ``POST /v1/{rollout_id}/chat/completions``. Every turn
of one rollout shares its ``rollout_id``, so the trainer can group all of a rollout's
turns (and the GRPO group they belong to) without the app knowing anything about training.

The trainer runs this proxy embedded in-process and talks to it through the
:class:`SamplingProxy` object directly for control (``set_policy`` / ``pop_captures``),
while the app reaches the chat endpoint over HTTP.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import tinker
from fastapi import FastAPI, Request

from tinker_cookbook import renderers


@dataclass
class TurnCapture:
    """One sampled turn: the exact tokens the policy saw and produced."""

    prompt_tokens: list[int]
    sampled_tokens: list[int]
    logprobs: list[float]


@dataclass
class _State:
    sampling_client: tinker.SamplingClient | None = None
    default_max_tokens: int = 2048
    temperature: float = 1.0
    captures: dict[str, list[TurnCapture]] = field(default_factory=dict)


class SamplingProxy:
    """Holds the current policy + captured tokens and exposes an OpenAI-compatible app."""

    def __init__(self, renderer: renderers.Renderer, *, default_max_tokens: int = 2048):
        self.renderer = renderer
        self._state = _State(default_max_tokens=default_max_tokens)
        self._lock = threading.Lock()
        self.app = self._build_app()

    # ---- control surface (called by the trainer in-process) ----
    def set_policy(
        self, sampling_client: tinker.SamplingClient, *, temperature: float | None = None
    ):
        with self._lock:
            self._state.sampling_client = sampling_client
            if temperature is not None:
                self._state.temperature = temperature

    def reset_captures(self) -> None:
        with self._lock:
            self._state.captures = {}

    def pop_captures(self, ids: list[str] | None = None) -> dict[str, list[TurnCapture]]:
        """Remove and return captures. With ``ids``, pop only those rollout ids (so
        overlapping batches in async training don't mix); otherwise drain everything."""
        with self._lock:
            if ids is None:
                caps = self._state.captures
                self._state.captures = {}
                return caps
            return {rid: self._state.captures.pop(rid) for rid in ids if rid in self._state.captures}

    def _record(self, rollout_id: str, cap: TurnCapture) -> None:
        with self._lock:
            self._state.captures.setdefault(rollout_id, []).append(cap)

    # ---- OpenAI-compatible HTTP surface (called by the app) ----
    def _build_app(self) -> FastAPI:
        app = FastAPI(title="RILL sampling proxy")
        counter = {"n": 0}

        async def _complete(rollout_id: str, body: dict) -> dict:
            sampling_client = self._state.sampling_client
            if sampling_client is None:
                return {"error": {"message": "proxy has no policy set"}}

            messages = body.get("messages", [])
            max_tokens = (
                body.get("max_completion_tokens")
                or body.get("max_tokens")
                or self._state.default_max_tokens
            )
            temperature = body.get("temperature")
            model_input = self.renderer.build_generation_prompt(messages)
            sampling_params = tinker.types.SamplingParams(
                max_tokens=int(max_tokens),
                temperature=self._state.temperature if temperature is None else float(temperature),
                stop=self.renderer.get_stop_sequences(),
            )
            sample = await sampling_client.sample_async(
                prompt=model_input, num_samples=1, sampling_params=sampling_params
            )
            seq = sample.sequences[0]
            sampled_tokens: list[int] = list(seq.tokens)
            logprobs: list[float] = list(seq.logprobs or [0.0] * len(sampled_tokens))

            self._record(
                rollout_id,
                TurnCapture(model_input.to_ints(), sampled_tokens, logprobs),
            )

            message, termination = self.renderer.parse_response(sampled_tokens)
            content = renderers.get_text_content(message)
            finish = "stop" if termination.is_clean else "length"
            counter["n"] += 1
            prompt_len = model_input.length
            return {
                "id": f"rill-{counter['n']}",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "rill-policy"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": len(sampled_tokens),
                    "total_tokens": prompt_len + len(sampled_tokens),
                },
            }

        @app.post("/v1/{rollout_id}/chat/completions")
        async def chat_with_id(rollout_id: str, request: Request):
            return await _complete(rollout_id, await request.json())

        @app.post("/v1/chat/completions")
        async def chat_default(request: Request):
            return await _complete("default", await request.json())

        return app
