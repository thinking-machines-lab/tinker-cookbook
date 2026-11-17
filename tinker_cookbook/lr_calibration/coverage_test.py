from types import SimpleNamespace
from typing import cast

import tinker

from tinker_cookbook.lr_calibration.coverage import (
    base_model_name,
    missing_models,
    one_id_per_base_model,
    trainable_base_models,
)

LISTED = [
    ("a", True),
    ("a:peft:131072", True),
    ("b:full:65536", True),
    ("c", False),
    ("d:peft:65536", True),
    ("d:peft:32768", True),
    ("e:peft:65536:preview", True),
]


def _client() -> tinker.RestClient:
    caps = SimpleNamespace(
        supported_models=[SimpleNamespace(model_name=n, trainable=t) for n, t in LISTED]
    )
    client = SimpleNamespace(get_server_capabilities=lambda: SimpleNamespace(result=lambda: caps))
    return cast(tinker.RestClient, client)


def test_base_model_name():
    assert base_model_name("org/m:peft:262144") == "org/m"
    assert base_model_name("org/m") == "org/m"


def test_one_id_per_base_model():
    names = ["a:peft:131072", "a", "b:peft:65536", "b:peft:32768", "c:peft:1"]
    assert one_id_per_base_model(names) == ["a", "b:peft:32768", "c:peft:1"]


def test_trainable_base_models_skips_full_and_untrainable_variants():
    assert trainable_base_models(_client()) == ["a", "d:peft:32768"]


def test_missing_models_matches_entries_by_base_name():
    assert missing_models({"models": {"d": {}}}, _client()) == ["a"]
