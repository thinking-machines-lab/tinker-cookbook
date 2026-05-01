"""Training shape selection for cookbook recipes.

Two paths:

* **Explicit** — caller provides ``training_shape_id``, calls
  ``resolve_training_profile()`` on the SDK to get the full profile
  (including ``deployment_shape_version``).  No selection logic needed.

* **Auto-select** — ``auto_select_training_shape()`` picks a validated
  training shape from the control plane based on base model, trainer
  mode, and context length.  Returns a shape ID; caller resolves the
  profile the same way.

The deployment shape always comes from the training shape profile's
``deployment_shape_version``.  There is no separate deployment shape
selection — if a training shape lacks a deployment shape link, it is
broken and should be fixed.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal
from urllib.parse import urlencode

from fireworks.training.sdk.trainer import TrainerJobManager
logger = logging.getLogger(__name__)

_TRAINING_SHAPE_VERSION_PARENT = "accounts/-/trainingShapes/-"
_ORDER_BY_CREATE_TIME_DESC = "create_time desc"
_TRAINING_SHAPE_VERSION_RE = re.compile(r"/versions/[^/]+$")

_TRAINER_MODE_BY_CODE = {
    1: "POLICY_TRAINER",
    2: "FORWARD_ONLY",
    3: "LORA_TRAINER",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def auto_select_training_shape(
    trainer_mgr: TrainerJobManager,
    *,
    base_model: str,
    trainer_role: Literal["policy", "reference"] = "policy",
    lora_rank: int = 0,
    max_seq_len: int | None = None,
    public_only: bool = False,
    shape_account: str | None = None,
) -> str:
    """Auto-select a validated training shape ID.

    Returns the training shape resource name (without ``/versions/...``).
    Caller should then call ``trainer_mgr.resolve_training_profile(shape_id)``
    to get the full profile including ``deployment_shape_version``.

    Raises ``ValueError`` if no matching shape is found.
    """
    expected_mode = _expected_trainer_mode(trainer_role, lora_rank)
    parent = (
        f"accounts/{shape_account}/trainingShapes/-"
        if shape_account
        else _TRAINING_SHAPE_VERSION_PARENT
    )

    # Try exact base_model match first.
    candidates = _list_and_filter(
        trainer_mgr,
        parent=parent,
        filter_expr=_combine_filters(
            f'snapshot.base_model="{base_model}"',
            f'snapshot.trainer_mode="{expected_mode}"',
            "latest_validated=true",
            "public=true" if public_only else "",
        ),
        expected_mode=expected_mode,
        max_seq_len=max_seq_len,
    )
    if candidates:
        return _pick_best(candidates, max_seq_len)

    # Fallback: compatible model_type + parameter_count bucket.
    model_ctx = _fetch_model_context(trainer_mgr, base_model)
    lo, hi = _param_count_bounds(model_ctx["parameter_count"])
    candidates = _list_and_filter(
        trainer_mgr,
        parent=parent,
        filter_expr=_combine_filters(
            f'snapshot.model_type="{model_ctx["model_type"]}"',
            f"snapshot.parameter_count>={lo}",
            f"snapshot.parameter_count<={hi}",
            f'snapshot.trainer_mode="{expected_mode}"',
            "latest_validated=true",
            "public=true" if public_only else "",
        ),
        expected_mode=expected_mode,
        max_seq_len=max_seq_len,
    )
    if candidates:
        return _pick_best(candidates, max_seq_len)

    mode_label = {"LORA_TRAINER": "LoRA", "FORWARD_ONLY": "reference"}.get(
        expected_mode, "full-tune"
    )
    raise ValueError(
        f"No validated training shape matched base_model={base_model!r}, "
        f"mode={mode_label!r}, max_seq_len={max_seq_len!r}. "
        f"Provide an explicit training_shape_id."
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _expected_trainer_mode(
    trainer_role: Literal["policy", "reference"],
    lora_rank: int,
) -> str:
    if lora_rank > 0:
        return "LORA_TRAINER"
    if trainer_role == "reference":
        return "FORWARD_ONLY"
    return "POLICY_TRAINER"


def _list_and_filter(
    client: TrainerJobManager,
    *,
    parent: str,
    filter_expr: str,
    expected_mode: str,
    max_seq_len: int | None,
) -> list[dict]:
    """List training shape versions and filter by mode + context length."""
    versions = _list_paginated(client, parent=parent, filter_expr=filter_expr)
    result = []
    for v in versions:
        snap = (v.get("snapshot") or {})
        mode = _normalize_trainer_mode(
            snap.get("trainerMode") or snap.get("trainer_mode")
        )
        if mode != expected_mode:
            continue
        ctx_len = _int_val(snap, "maxSupportedContextLength", "max_supported_context_length")
        if max_seq_len is not None and ctx_len < max_seq_len:
            continue
        name = v.get("name", "")
        shape_id = _TRAINING_SHAPE_VERSION_RE.sub("", name) or name
        result.append({"shape_id": shape_id, "ctx_len": ctx_len})
    return result


def _pick_best(candidates: list[dict], max_seq_len: int | None) -> str:
    """Pick the candidate with smallest sufficient context length."""
    if max_seq_len is None:
        return candidates[0]["shape_id"]
    return min(candidates, key=lambda c: c["ctx_len"])["shape_id"]


def _fetch_model_context(client: TrainerJobManager, base_model: str) -> dict:
    resp = client._get(f"/v1/{base_model}", timeout=30)
    if not resp.is_success:
        raise RuntimeError(
            f"Failed to fetch model details for {base_model!r} (HTTP {resp.status_code})"
        )
    data = resp.json() or {}
    details = data.get("baseModelDetails") or data.get("base_model_details") or {}
    model_type = (
        details.get("modelType") or details.get("model_type")
        or data.get("modelType") or data.get("model_type")
    )
    param_count = (
        _try_int(details.get("parameterCount") or details.get("parameter_count"))
        or _try_int(data.get("parameterCount") or data.get("parameter_count"))
        or 0
    )
    if not model_type or param_count <= 0:
        raise ValueError(f"Base model {base_model!r} missing model_type or parameter_count")
    return {"model_type": model_type, "parameter_count": param_count}


def _list_paginated(
    client: TrainerJobManager,
    *,
    parent: str,
    filter_expr: str,
) -> list[dict]:
    resources: list[dict] = []
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {
            "filter": filter_expr,
            "orderBy": _ORDER_BY_CREATE_TIME_DESC,
            "pageSize": 200,
        }
        if page_token:
            params["pageToken"] = page_token
        resp = client._get(f"/v1/{parent}/versions?{urlencode(params)}", timeout=30)
        if not resp.is_success:
            raise RuntimeError(
                f"Failed to list training shape versions (HTTP {resp.status_code})"
            )
        data = resp.json() or {}
        items = (
            data.get("trainingShapeVersions")
            or data.get("training_shape_versions")
            or []
        )
        resources.extend(items)
        page_token = data.get("nextPageToken") or data.get("next_page_token")
        if not page_token:
            break
    return resources


def _param_count_bounds(param_count: int) -> tuple[int, int]:
    one_b = 1_000_000_000
    bucket = one_b if param_count < 10 * one_b else 10 * one_b
    lo = (param_count // bucket) * bucket
    return lo, lo + bucket


def _normalize_trainer_mode(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    return _TRAINER_MODE_BY_CODE.get(int(value))


def _combine_filters(*parts: str) -> str:
    return " AND ".join(p for p in parts if p)


def _int_val(d: dict, *keys: str) -> int:
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return int(v)
    return 0


def _try_int(v: Any) -> int:
    if v in (None, ""):
        return 0
    return int(v)
