from __future__ import annotations

from typing import cast

import tinker

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.preference.train_dpo import Config, create_dpo_clients
from tinker_cookbook.supervised.types import ChatDatasetBuilder


class _Future:
    def __init__(self, value: object) -> None:
        self.value = value

    def result(self) -> object:
        return self.value


class _TrainingClient:
    """Stands for the weights named by ``weights``; its sampling client samples those weights."""

    def __init__(self, weights: str) -> None:
        self.weights = weights

    def save_weights_and_get_sampling_client(self) -> _Future:
        return _Future(f"sampler of {self.weights}")


class _ServiceClient:
    def create_lora_training_client(
        self, base_model: str, rank: int, user_metadata: dict[str, str] | None = None
    ) -> _Future:
        return _Future(_TrainingClient(f"new adapter on {base_model}"))

    def create_training_client_from_state(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> _Future:
        return _Future(_TrainingClient(path))

    def create_training_client_from_state_with_optimizer(
        self, path: str, user_metadata: dict[str, str] | None = None
    ) -> _Future:
        return _Future(_TrainingClient(path))

    def create_sampling_client(self, base_model: str) -> str:
        return f"sampler of {base_model}"


def _clients(
    load_checkpoint_path: str | None, resume_from: str | None
) -> tuple[_TrainingClient, object]:
    config = Config(
        log_path="/tmp/unused",
        model_name="some/model",
        recipe_name="test",
        dataset_builder=cast(ChatDatasetBuilder, None),
        load_checkpoint_path=load_checkpoint_path,
    )
    resume_info = (
        checkpoint_utils.CheckpointRecord(name="000020", batch=20, state_path=resume_from)
        if resume_from
        else None
    )
    training_client, reference_client = create_dpo_clients(
        config, cast(tinker.ServiceClient, _ServiceClient()), resume_info
    )
    return cast(_TrainingClient, training_client), reference_client


def test_a_new_run_takes_its_reference_from_the_weights_it_starts_with() -> None:
    training_client, reference = _clients(None, None)
    assert training_client.weights == "new adapter on some/model"
    assert reference == "sampler of new adapter on some/model"
    training_client, reference = _clients("tinker://sft/weights/final", None)
    assert training_client.weights == "tinker://sft/weights/final"
    assert reference == "sampler of tinker://sft/weights/final"


def test_a_resumed_run_keeps_the_reference_it_started_with() -> None:
    resumed = "tinker://dpo/weights/000020"
    # From the base model: training continues from the saved state, and the reference
    # is still the base model, not the half-trained weights.
    training_client, reference = _clients(None, resumed)
    assert training_client.weights == resumed
    assert reference == "sampler of some/model"
    # From a checkpoint: the reference is that checkpoint.
    training_client, reference = _clients("tinker://sft/weights/final", resumed)
    assert training_client.weights == resumed
    assert reference == "sampler of tinker://sft/weights/final"
