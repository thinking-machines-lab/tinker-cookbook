import asyncio
import logging
import time
from functools import cache

import chz
import httpx
import pandas as pd
import tinker
import torch
from tinker import AdamParams, ModelInput
from tinker_cookbook.supervised.common import datum_from_model_input_weights


@cache
def get_reference_document():
    """Download PyTorch's forward_ad.py file from a specific commit."""
    url = "https://raw.githubusercontent.com/pytorch/pytorch/a10b765bf159a86fb2a0ad693c6b72e0c691e60b/torch/autograd/forward_ad.py"
    response = httpx.get(url)
    response.raise_for_status()
    return response.text


async def get_row(
    model_name: str,
    service_client: tinker.ServiceClient,
    timeout_sec: float,
    saved_path_for_trainer: str | None = None,
    saved_path_for_sampler: str | None = None,
) -> dict:
    async def _inner():
        tstart = time.time()
        print(f"========== Testing {model_name} ==========")
        training_client = await service_client.create_lora_training_client_async(
            base_model=model_name
        )
        if saved_path_for_trainer is not None:
            await training_client.load_state_async(saved_path_for_trainer)
        # First sample something
        tokenizer = training_client.get_tokenizer()
        tokens = tokenizer.encode(get_reference_document())
        model_input = ModelInput.from_ints(tokens)
        weights = torch.ones(len(tokens), dtype=torch.float32)
        weights[0] = 0.0
        datum = datum_from_model_input_weights(model_input, weights)
        for _ in range(3 if saved_path_for_trainer is None else 0):
            fwd_bwd_future = await training_client.forward_backward_async(
                [datum], loss_fn="cross_entropy"
            )
            optim_step_future = await training_client.optim_step_async(
                adam_params=AdamParams(learning_rate=1e-3)
            )
            _fwd_bwd_result = await fwd_bwd_future.result_async()
            _optim_step_result = await optim_step_future.result_async()
        fwd_future = await training_client.forward_async([datum], loss_fn="cross_entropy")
        fwd_result = await fwd_future.result_async()
        training_logprobs = fwd_result.loss_fn_outputs[0]["logprobs"].to_torch()
        if saved_path_for_sampler is None:
            state_for_trainer_future = await training_client.save_state_async(name="tmp-checkpoint")
            state_for_trainer = await state_for_trainer_future.result_async()
            print(f"Saved state for trainer: {state_for_trainer.path}")
            sampling_client = await training_client.save_weights_and_get_sampling_client_async(
                name="tmp-checkpoint"
            )
        else:
            sampling_client = training_client.create_sampling_client(
                model_path=saved_path_for_sampler
            )
        logprobs_response = await sampling_client.compute_logprobs_async(model_input)
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


@chz.chz
class Config:
    base_url: str | None = None
    print_models: bool = False
    model_names: list[str] | None = None
    model_name_filter: list[str] | None = chz.field(default_factory=lambda: ["loadtest"])
    state_for_trainer: str | None = None
    state_for_sampler: str | None = None


async def main(config: Config):
    logging.basicConfig(level=logging.INFO)
    service_client = tinker.ServiceClient(base_url=config.base_url)

    if config.model_names is None:
        server_capabilities = await service_client.get_server_capabilities_async()
        model_names = [
            model_info.model_name
            for model_info in server_capabilities.supported_models
            if model_info.model_name is not None
        ]
        if config.print_models:
            print("Available models:")
            for model_name in model_names:
                print(f"- {model_name}")
            return
    else:
        model_names = list(config.model_names)

    def should_do_model(model_name: str) -> bool:
        if not config.model_name_filter:
            return True
        return not any(x in model_name for x in config.model_name_filter)

    model_names = [x for x in sorted(model_names) if should_do_model(x)]
    print(f"Model names: {model_names}")
    timeout_sec = 300.0
    rows = await asyncio.gather(
        *[
            get_row(
                model_name,
                service_client,
                timeout_sec,
                config.state_for_trainer,
                config.state_for_sampler,
            )
            for model_name in model_names
        ]
    )

    df = pd.DataFrame(rows)
    # Ensure df has all required columns with NaN for missing values
    required_columns = ["model_name", "mse[sample, train]", "time", "error"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[required_columns]

    df.to_csv("/tmp/sampling_training_logprobs.csv", index=False)
    print(df.to_markdown())


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
