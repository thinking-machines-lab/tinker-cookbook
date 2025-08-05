import asyncio
import time
from functools import cache

import httpx
import pytest
import tinker
import torch
from tinker.types import AdamParams, ModelInput
from tinker_backend.scripts.start_vllm import wait_for_server_health
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook_internal.test_util import (
    cleanup_process,
    start_tinker_api,
    start_vllm,
    wait_for_server_ready,
)


@cache
def get_reference_document():
    """Download PyTorch's forward_ad.py file from a specific commit."""
    url = "https://raw.githubusercontent.com/pytorch/pytorch/a10b765bf159a86fb2a0ad693c6b72e0c691e60b/torch/autograd/forward_ad.py"
    response = httpx.get(url)
    response.raise_for_status()
    return response.text


async def _run_kl_test(
    model_name: str,
    timeout_sec: float,
) -> dict:
    async def _inner():
        tstart = time.time()
        print(f"========== Testing {model_name} ==========")
        service_client = tinker.ServiceClient(base_url="http://localhost:8080")
        training_client = await service_client.create_lora_training_client_async(
            base_model=model_name
        )
        # First sample something
        tokenizer = training_client.get_tokenizer()
        tokens = torch.tensor(tokenizer.encode(get_reference_document()))
        weights = torch.ones_like(tokens)
        weights[0] = 0.0
        datum = datum_from_tokens_weights(tokens, weights)
        num_updates = 3
        for iteration in range(num_updates + 1):
            fwd_bwd_future = await training_client.forward_backward_async(
                [datum], loss_fn="cross_entropy"
            )
            # Use zero LR on the last iteration because we we'll use those training logprobs
            # to compare against the sampling logprobs, and the sampling client will use the
            # post-update model
            optim_step_future = await training_client.optim_step_async(
                adam_params=AdamParams(learning_rate=1e-4 if iteration < num_updates else 0.0)
            )
            fwd_bwd_result = await fwd_bwd_future.result_async()
            _optim_step_result = await optim_step_future.result_async()
        training_logprobs = fwd_bwd_result.loss_fn_outputs[0]["logprobs"].to_torch()  # pyright: ignore[reportPossiblyUnboundVariable]
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name="tmp-checkpoint"
        )
        logprobs_response = await sampling_client.compute_logprobs_async(
            ModelInput.from_ints(tokens.tolist())
        )
        sampling_logprobs = torch.tensor(logprobs_response[1:])
        mse = ((sampling_logprobs - training_logprobs) ** 2).mean()

        dur = time.time() - tstart
        print(f"Time taken: {dur:.1f} seconds")
        result = {
            "model_name": model_name,
            "mse[sample, train]": mse.item(),
            "time": dur,
        }
        print(result)
        return result

    try:
        return await asyncio.wait_for(_inner(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        print(f"ERROR: Timeout after {timeout_sec} seconds for model {model_name}")
        return {"model_name": model_name, "error": "TimeoutError"}


@pytest.mark.skip(reason="Skipping while fixing")
@pytest.mark.need_2_gpus
def test_kl():
    tinker_api_process = start_tinker_api(world_size=2, num_op_shards=2)
    time.sleep(5)
    vllm_process = start_vllm(num_op_shards=2)
    time.sleep(5)
    wait_for_server_ready()
    asyncio.run(wait_for_server_health("localhost", 8001, 120.0))
    try:
        result = asyncio.run(
            _run_kl_test(
                model_name="meta-llama/Llama-3.2-1B",
                timeout_sec=300.0,
            )
        )
        assert result["mse[sample, train]"] < 5e-3
    finally:
        cleanup_process(tinker_api_process)
        cleanup_process(vllm_process)


if __name__ == "__main__":
    test_kl()
