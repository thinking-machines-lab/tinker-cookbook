"""End-to-end lifecycle test: train → save → download → build.

Trains a tiny SFT model for 1 step, saves the checkpoint, downloads
the adapter via weights.download(), and builds a merged HF model via
weights.build_hf_model().

Requires TINKER_API_KEY and network access. Skipped otherwise.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import cast

import datasets
import pytest
import tinker

from tinker_cookbook import renderers
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.weights import build_hf_model, download


@pytest.mark.integration
class TestFullLifecycle:
    """Train 1 step → save → download → build merged HF model."""

    MODEL_NAME = "Qwen/Qwen3-8B"
    RENDERER_NAME = "qwen3"
    BATCH_SIZE = 4
    MAX_LENGTH = 512
    LORA_RANK = 8

    def _train_and_save(self, log_path: str) -> str:
        """Train for 1 step and return the sampler checkpoint tinker:// path."""
        tokenizer = get_tokenizer(self.MODEL_NAME)
        renderer = renderers.get_renderer(self.RENDERER_NAME, tokenizer)

        # Load a small slice of data
        dataset = datasets.load_dataset("allenai/tulu-3-sft-mixture")
        dataset = cast(datasets.DatasetDict, dataset)
        train_ds = dataset["train"].take(self.BATCH_SIZE)

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(row["messages"], renderer, self.MAX_LENGTH)

        sft_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.BATCH_SIZE, map_fn=map_fn
        )

        async def _run() -> str:
            sc = tinker.ServiceClient()
            tc = await sc.create_lora_training_client_async(
                base_model=self.MODEL_NAME,
                rank=self.LORA_RANK,
            )

            # Train 1 step
            batch = sft_dataset.get_batch(0)
            fwd_bwd = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            await fwd_bwd.result_async()
            optim = await tc.optim_step_async({"learning_rate": 1e-4})
            await optim.result_async()

            # Save checkpoint
            sampler_resp = await tc.save_weights_for_sampler_async("lifecycle_test")
            result = await sampler_resp.result_async()
            return result.path

        return asyncio.run(_run())

    def test_train_download_build(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_path = str(root / "logs")
            Path(log_path).mkdir()

            # Step 1: Train and save
            tinker_path = self._train_and_save(log_path)
            assert tinker_path.startswith("tinker://"), f"Unexpected path format: {tinker_path}"

            # Step 2: Download
            adapter_dir = download(
                tinker_path=tinker_path,
                output_dir=str(root / "adapter"),
            )
            adapter_path = Path(adapter_dir)
            assert (adapter_path / "adapter_model.safetensors").exists(), (
                f"adapter_model.safetensors not found in {adapter_dir}"
            )
            assert (adapter_path / "adapter_config.json").exists(), (
                f"adapter_config.json not found in {adapter_dir}"
            )

            # Step 3: Build merged HF model
            output_path = str(root / "merged")
            build_hf_model(
                base_model=self.MODEL_NAME,
                adapter_path=adapter_dir,
                output_path=output_path,
            )

            # Verify output looks like a valid HF model
            out = Path(output_path)
            assert (out / "config.json").exists(), "config.json missing from merged model"
            assert any(out.glob("*.safetensors")), "No safetensors files in merged model"
            assert (out / "tokenizer.json").exists() or (out / "tokenizer_config.json").exists(), (
                "Tokenizer files missing from merged model"
            )
