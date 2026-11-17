"""Where the calibration table lives, and which Tinker models it should cover."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import tinker

CALIBRATION_FILE = Path(__file__).with_name("calibration_data.json")


def base_model_name(model_name: str) -> str:
    """Strip Tinker's variant suffix (e.g. ``:peft:262144``); table entries use the base name."""
    return model_name.split(":", 1)[0]


def one_id_per_base_model(model_names: list[str]) -> list[str]:
    """Keep one id per base model, so each table entry is calibrated once: the plain id if it is
    listed, else the first ``:peft:`` id in sorted order."""
    by_base: dict[str, list[str]] = {}
    for name in model_names:
        by_base.setdefault(base_model_name(name), []).append(name)
    return [base if base in names else sorted(names)[0] for base, names in by_base.items()]


def trainable_base_models(rest_client: tinker.RestClient) -> list[str]:
    """One id per trainable base model that the service lists."""
    caps = rest_client.get_server_capabilities().result()
    names = []
    for m in caps.supported_models:
        if not m.trainable or m.model_name is None:
            continue
        # Only the plain id and its ``:peft:<context>`` variants are general LoRA models;
        # longer suffixes mark special-purpose variants of them.
        _, _, suffix = m.model_name.partition(":")
        if suffix and not re.fullmatch(r"peft:\d+", suffix):
            continue
        names.append(m.model_name)
    return sorted(one_id_per_base_model(names))


def missing_models(table: dict[str, Any], rest_client: tinker.RestClient) -> list[str]:
    """Trainable models that have no entry in the calibration table."""
    return [
        m for m in trainable_base_models(rest_client) if base_model_name(m) not in table["models"]
    ]
