"""A local ``/inference/v1/generate`` endpoint backed by Tinker sampling.

The verifiers v1 train client (``vf.TrainClientConfig``) renders each turn to
token IDs with Prime Intellect's ``renderers`` package and POSTs them to a vLLM
``/inference/v1/generate`` endpoint, expecting sampled token IDs and per-token
logprobs back. That endpoint is the one seam where an inference engine plugs
into the v1 rollout stack, so this module serves its wire format over a
``tinker.SamplingClient`` — everything else (harness programs, interception,
renderer bridging, trace/token bookkeeping) is verifiers running natively.

The server binds a random loopback port; ``train_client_config()`` returns the
``vf.TrainClientConfig`` that points a rollout at it.
"""

from __future__ import annotations

import logging
import uuid

import tinker
import verifiers.v1 as vf
from aiohttp import web

logger = logging.getLogger(__name__)


_SAMPLING_KEYS = frozenset(
    {"max_tokens", "temperature", "top_p", "top_k", "seed", "stop_token_ids"}
)
"""Request keys mapped onto ``tinker.SamplingParams``."""
_WIRE_KEYS = frozenset({"logprobs", "skip_special_tokens"})
"""Keys the renderers client always sends for token-in/token-out bookkeeping; the
endpoint always returns per-token logprobs and raw token ids, so they carry no choice."""


class TinkerGenerateServer:
    """Serves vLLM's token-in/token-out generate API over Tinker sampling.

    The sampling client is swappable (``set_sampling_client``) so an RL loop
    can point the same endpoint at each new policy checkpoint without
    rebuilding the verifiers client stack that connects to it.
    """

    def __init__(self, sampling_client: tinker.SamplingClient | None = None) -> None:
        self._sampling_client = sampling_client
        self._runner: web.AppRunner | None = None
        self.base_url = ""

    def set_sampling_client(self, sampling_client: tinker.SamplingClient) -> None:
        """Point the endpoint at a new policy checkpoint."""
        self._sampling_client = sampling_client

    async def start(self) -> None:
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_post("/inference/v1/generate", self._handle_generate)
        app.router.add_get("/v1/models", self._handle_models)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        host, port = self._runner.addresses[0][:2]
        # The `/v1` suffix matches vLLM's OpenAI-style base URL; the train
        # client strips it before POSTing to `/inference/v1/generate` and keeps
        # it for `GET /v1/models`.
        self.base_url = f"http://{host}:{port}/v1"
        logger.info("Tinker generate endpoint listening on %s", self.base_url)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self.base_url = ""

    async def __aenter__(self) -> TinkerGenerateServer:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    def train_client_config(self, renderer_model_name: str) -> vf.TrainClientConfig:
        """The verifiers client config that samples through this endpoint.

        ``renderer_model_name`` pins the tokenizer/renderer that the train
        client builds prompts with; it must match the Tinker base model being
        sampled, or token IDs would come from one vocabulary and go to another.
        """
        if not self.base_url:
            raise RuntimeError("server is not running; call start() first")
        return vf.TrainClientConfig.model_validate(
            {
                "base_url": self.base_url,
                # No key is checked; the var is named so the default
                # PRIME_API_KEY config resolution does not kick in for this
                # loopback endpoint.
                "api_key_var": "TINKER_GENERATE_API_KEY",
                "renderer_model_name": renderer_model_name,
            }
        )

    async def _handle_models(self, request: web.Request) -> web.Response:
        # Served so the train client's max_model_len discovery gets a clean
        # answer. No `max_model_len` is advertised (Tinker does not expose
        # one), so the client-side overflow pre-flight stays disabled and
        # overlong prompts surface as engine errors instead.
        del request
        return web.json_response({"object": "list", "data": []})

    async def _handle_generate(self, request: web.Request) -> web.Response:
        if self._sampling_client is None:
            return web.json_response(
                {"error": "no sampling client attached to the generate endpoint"},
                status=503,
            )
        body = await request.json()
        token_ids = body.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            return web.json_response(
                {"error": "request must carry a non-empty `token_ids` list"}, status=400
            )
        if body.get("features"):
            return web.json_response(
                {"error": "multimodal `features` are not supported by the Tinker endpoint"},
                status=400,
            )
        sampling_params = body.get("sampling_params") or {}
        unsupported = sorted(set(sampling_params) - _SAMPLING_KEYS - _WIRE_KEYS)
        if unsupported:
            # verifiers passes provider-specific sampling keys through; silently
            # dropping one would run a different distribution than requested.
            return web.json_response(
                {"error": f"unsupported sampling_params for the Tinker endpoint: {unsupported}"},
                status=400,
            )
        stop = sampling_params.get("stop_token_ids") or None
        max_tokens = sampling_params.get("max_tokens")
        params = tinker.SamplingParams(
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            temperature=float(sampling_params.get("temperature", 1.0)),
            top_p=float(sampling_params.get("top_p", 1.0)),
            top_k=int(sampling_params.get("top_k", -1)),
            seed=sampling_params.get("seed"),
            stop=[int(t) for t in stop] if stop else None,
        )
        try:
            result = await self._sampling_client.sample_async(
                prompt=tinker.ModelInput.from_ints([int(t) for t in token_ids]),
                num_samples=1,
                sampling_params=params,
            )
        except Exception as e:
            # Surface as a 500 so the verifiers client records a model error on
            # the trace (and retries per its policy) instead of crashing the
            # rollout stack.
            logger.exception("Tinker sampling failed")
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
        sequence = result.sequences[0]
        completion_ids = [int(t) for t in sequence.tokens]
        logprobs = sequence.logprobs
        if logprobs is None or len(logprobs) != len(completion_ids):
            return web.json_response(
                {"error": "Tinker sampling returned no per-token logprobs"}, status=500
            )
        return web.json_response(
            {
                "request_id": uuid.uuid4().hex,
                "choices": [
                    {
                        "index": 0,
                        "token_ids": completion_ids,
                        # Tinker's stop_reason is already "stop" | "length",
                        # matching vLLM's finish_reason vocabulary.
                        "finish_reason": sequence.stop_reason,
                        "logprobs": {
                            "content": [
                                {"token": f"token_id:{token}", "logprob": float(logprob)}
                                for token, logprob in zip(completion_ids, logprobs)
                            ]
                        },
                    }
                ],
            }
        )
