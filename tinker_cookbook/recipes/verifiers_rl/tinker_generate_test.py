"""Tests for the Tinker-backed /inference/v1/generate endpoint (no API calls)."""

from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

try:
    import verifiers.v1 as _verifiers  # noqa: F401

    _has_verifiers = True
except ImportError:
    _has_verifiers = False

pytestmark = pytest.mark.skipif(not _has_verifiers, reason="verifiers not installed")


class _FakeSamplingClient:
    """Records sample_async calls and returns a fixed sequence."""

    def __init__(self, stop_reason: str = "stop") -> None:
        self.calls: list[dict[str, Any]] = []
        self.stop_reason = stop_reason

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls.append(
            {"prompt": prompt, "num_samples": num_samples, "sampling_params": sampling_params}
        )
        sequence = SimpleNamespace(
            tokens=[7, 8, 9], logprobs=[-0.1, -0.2, -0.3], stop_reason=self.stop_reason
        )
        return SimpleNamespace(sequences=[sequence])


async def _post_generate(server, body: dict) -> tuple[int, dict]:
    url = server.base_url.removesuffix("/v1") + "/inference/v1/generate"
    async with aiohttp.ClientSession() as session, session.post(url, json=body) as response:
        return response.status, await response.json()


@pytest.mark.asyncio
async def test_generate_returns_vllm_wire_format() -> None:
    from tinker_cookbook.recipes.verifiers_rl.tinker_generate import TinkerGenerateServer

    fake = _FakeSamplingClient()
    async with TinkerGenerateServer(fake) as server:  # type: ignore[arg-type]
        status, data = await _post_generate(
            server,
            {
                "model": "tinker",
                "token_ids": [1, 2, 3],
                "sampling_params": {
                    "max_tokens": 32,
                    "temperature": 0.7,
                    "stop_token_ids": [42],
                    # Always sent by the renderer client; must be ignored.
                    "logprobs": 1,
                    "skip_special_tokens": False,
                },
            },
        )

    assert status == 200
    [choice] = data["choices"]
    assert choice["token_ids"] == [7, 8, 9]
    assert choice["finish_reason"] == "stop"
    # The renderer client requires one logprob entry per completion token,
    # each tagged "token_id:<id>".
    entries = choice["logprobs"]["content"]
    assert [entry["token"] for entry in entries] == ["token_id:7", "token_id:8", "token_id:9"]
    assert [entry["logprob"] for entry in entries] == [-0.1, -0.2, -0.3]

    [call] = fake.calls
    assert call["prompt"].to_ints() == [1, 2, 3]
    assert call["num_samples"] == 1
    params = call["sampling_params"]
    assert params.max_tokens == 32
    assert params.temperature == 0.7
    assert params.stop == [42]


@pytest.mark.asyncio
async def test_generate_maps_length_stop_reason() -> None:
    from tinker_cookbook.recipes.verifiers_rl.tinker_generate import TinkerGenerateServer

    async with TinkerGenerateServer(_FakeSamplingClient(stop_reason="length")) as server:  # type: ignore[arg-type]
        _, data = await _post_generate(
            server, {"model": "tinker", "token_ids": [1], "sampling_params": {}}
        )
    assert data["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_generate_rejects_bad_requests() -> None:
    from tinker_cookbook.recipes.verifiers_rl.tinker_generate import TinkerGenerateServer

    async with TinkerGenerateServer(_FakeSamplingClient()) as server:  # type: ignore[arg-type]
        status, _ = await _post_generate(server, {"model": "tinker", "token_ids": []})
        assert status == 400
        status, _ = await _post_generate(
            server, {"model": "tinker", "token_ids": [1], "features": {"mm_hashes": []}}
        )
        assert status == 400


@pytest.mark.asyncio
async def test_generate_without_sampling_client_is_503() -> None:
    from tinker_cookbook.recipes.verifiers_rl.tinker_generate import TinkerGenerateServer

    async with TinkerGenerateServer() as server:
        status, _ = await _post_generate(server, {"model": "tinker", "token_ids": [1]})
    assert status == 503


@pytest.mark.asyncio
async def test_train_client_config_points_at_the_server() -> None:
    import verifiers.v1 as vf

    from tinker_cookbook.recipes.verifiers_rl.tinker_generate import TinkerGenerateServer

    server = TinkerGenerateServer()
    with pytest.raises(RuntimeError, match="not running"):
        server.train_client_config("Qwen/Qwen3.5-4B")
    async with server:
        config = server.train_client_config("Qwen/Qwen3.5-4B")
        assert isinstance(config, vf.TrainClientConfig)
        assert config.base_url == server.base_url
        assert config.renderer_model_name == "Qwen/Qwen3.5-4B"

        # The models listing the train client probes for max_model_len.
        async with (
            aiohttp.ClientSession() as session,
            session.get(server.base_url + "/models") as response,
        ):
            assert response.status == 200
            assert (await response.json())["data"] == []


@pytest.mark.asyncio
async def test_generate_rejects_unsupported_sampling_params() -> None:
    """A sampling key the endpoint cannot honor is an error, never a silent drop."""
    from tinker_cookbook.recipes.verifiers_rl.tinker_generate import TinkerGenerateServer

    fake = _FakeSamplingClient()
    async with TinkerGenerateServer(fake) as server:  # type: ignore[arg-type]
        status, data = await _post_generate(
            server,
            {
                "token_ids": [1, 2, 3],
                "sampling_params": {
                    "max_tokens": 8,
                    "logprobs": 1,
                    "skip_special_tokens": False,
                    "repetition_penalty": 1.1,
                    "stop": ["</answer>"],
                },
            },
        )

    assert status == 400
    assert "repetition_penalty" in data["error"] and "stop" in data["error"]
    assert fake.calls == []
