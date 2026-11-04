"""Tests for the reference model that create_dpo_clients sets up."""

from typing import cast
from unittest.mock import MagicMock

import tinker

from tinker_cookbook.preference.train_dpo import Config, create_dpo_clients
from tinker_cookbook.supervised.types import ChatDatasetBuilder


def _config(reference_model_name: str | None) -> Config:
    return Config(
        log_path="unused-log-path",
        model_name="meta-llama/Llama-3.1-8B",
        recipe_name="test",
        dataset_builder=cast(ChatDatasetBuilder, MagicMock()),
        reference_model_name=reference_model_name,
    )


def test_explicit_reference_model_is_used_as_the_reference():
    service_client = MagicMock()

    _, reference_client = create_dpo_clients(
        _config("meta-llama/Llama-3.1-8B-Instruct"), cast(tinker.ServiceClient, service_client)
    )

    service_client.create_sampling_client.assert_called_once_with(
        base_model="meta-llama/Llama-3.1-8B-Instruct"
    )
    assert reference_client is service_client.create_sampling_client.return_value
    lora_client = service_client.create_lora_training_client.return_value
    lora_client.save_weights_and_get_sampling_client.assert_not_called()


def test_reference_defaults_to_the_initial_training_weights():
    service_client = MagicMock()

    _, reference_client = create_dpo_clients(
        _config(None), cast(tinker.ServiceClient, service_client)
    )

    lora_client = service_client.create_lora_training_client.return_value
    assert reference_client is lora_client.save_weights_and_get_sampling_client.return_value
    service_client.create_sampling_client.assert_not_called()
