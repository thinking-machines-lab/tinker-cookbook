"""Test for checkpoint resume functionality in supervised training."""

import asyncio
import contextlib
import json
import os
import tempfile
from unittest.mock import patch

from tinker_cookbook import renderers
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tests.test_utils import create_mock_logger_with_jsonl
from tinker_cookbook.utils.file_utils import read_jsonl


class StopTrainingException(Exception):
    """Exception to stop training at a specific step."""

    pass


def checkpoint_resume():
    interrupt_step = 8
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = tmpdir
        os.makedirs(log_path, exist_ok=True)

        # Use the real NoRobots dataset with a small batch size
        model_name = "meta-llama/Llama-3.2-1B"
        common_config = ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=model_name,
            renderer_name="role_colon",
            max_length=1024,
            batch_size=32,
            train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )

        # Create config
        config = train.Config(
            log_path=log_path,
            model_name=model_name,
            dataset_builder=chat_datasets.NoRobotsBuilder(common_config=common_config),
            num_epochs=1,
            save_every=5,
            eval_every=0,
            infrequent_eval_every=0,
            wandb_project=None,
            lora_rank=16,
            learning_rate=1e-5,
        )

        # Ensure interrupt happens after checkpoint
        assert interrupt_step > config.save_every, (
            f"interrupt_step ({interrupt_step}) must be > save_every ({config.save_every}) "
            "to test checkpoint resume"
        )

        # First run - stop at interrupt_step
        with patch("tinker_cookbook.utils.ml_log.setup_logging") as mock_setup_logging:
            mock_logger = create_mock_logger_with_jsonl(
                log_path=log_path,
                interrupt_at_step=interrupt_step,
                interrupt_exception_class=StopTrainingException,
            )
            mock_setup_logging.return_value = mock_logger

            # Run until exception
            with contextlib.suppress(StopTrainingException):
                asyncio.run(train.main(config))

        # Verify checkpoint was saved at step 5
        checkpoint_file = os.path.join(log_path, "checkpoints.jsonl")
        assert os.path.exists(checkpoint_file), "Checkpoint file should exist"

        with open(checkpoint_file, "r") as f:
            checkpoints = [json.loads(line) for line in f]
        assert len(checkpoints) > 0, "Should have at least one checkpoint"
        assert checkpoints[0]["name"] == "000005", "First checkpoint should be at step 5"

        # Read first run metrics
        first_run_metrics = read_jsonl(os.path.join(log_path, "metrics.jsonl"))

        # Second run - resume from checkpoint
        with patch("tinker_cookbook.utils.ml_log.setup_logging") as mock_setup_logging:
            mock_logger2 = create_mock_logger_with_jsonl(
                log_path=log_path,
                metrics_filename="metrics_run2.jsonl",
                interrupt_at_step=interrupt_step,
                interrupt_exception_class=StopTrainingException,
            )
            mock_setup_logging.return_value = mock_logger2

            with contextlib.suppress(StopTrainingException):
                asyncio.run(train.main(config))

        # Read second run metrics
        second_run_metrics = read_jsonl(os.path.join(log_path, "metrics_run2.jsonl"))

        # Extract losses
        first_losses = {
            m["step"]: m["train_mean_nll"] for m in first_run_metrics if "train_mean_nll" in m
        }
        second_losses = {
            m["step"]: m["train_mean_nll"] for m in second_run_metrics if "train_mean_nll" in m
        }

        overlap_steps = [5, 6, 7]
        # Check that steps 6 and 7 have approximately the same losses in both runs
        # (We resumed from checkpoint at step 5, so steps 6 and 7 should be similar)
        for step in overlap_steps:
            assert step in first_losses, f"Step {step} missing from first run"
            assert step in second_losses, f"Step {step} missing from second run"

            # Losses should be very close (within 5% relative difference)
            loss1 = first_losses[step]
            loss2 = second_losses[step]
            relative_diff = abs(loss1 - loss2) / max(abs(loss1), abs(loss2))
            assert relative_diff < 0.01, (
                f"Loss at step {step} should be similar: "
                f"{loss1} vs {loss2} (relative diff: {relative_diff:.2%})"
            )

        print("✓ Test passed: training resumed correctly from checkpoint")
        print(f"  First run losses at steps 5-7: {[first_losses[i] for i in overlap_steps]}")
        print(f"  Second run losses at steps 5-7: {[second_losses[i] for i in overlap_steps]}")


if __name__ == "__main__":
    checkpoint_resume()
