"""Pre-flight validation for cookbook configs.

Application-level checks only (credentials, base_model format, dataset).
Infra-field validation lives in the SDK's ``TrainerJobConfig.validate()``.
"""

from __future__ import annotations

import logging
import re

try:
    from fireworks.training.sdk.errors import format_sdk_error, DOCS_SDK
except ImportError:
    from fireworks.training.sdk.errors import format_sdk_error
    DOCS_SDK = ""
try:
    from fireworks.training.sdk import validate_output_model_id
except ImportError:
    _OUTPUT_MODEL_ID_RE = re.compile(
        r"^(accounts/[^/]+/models/[A-Za-z0-9-]+|[A-Za-z0-9-]+)$"
    )

    def validate_output_model_id(_value: str) -> list[str]:
        if _OUTPUT_MODEL_ID_RE.match(_value):
            return []
        return [
            format_sdk_error(
                "Invalid output_model_id",
                f"'{_value}' doesn't match expected format.",
                "Use either a short model name (letters / numbers / '-') or "
                "a full resource name like accounts/ACCOUNT/models/MODEL_NAME.\n"
                "  Example: promoted-orpo-model\n"
                "  Example: accounts/pyroworks/models/my-model",
            )
        ]
from tinker_cookbook.fireworks.utils.config import DeployConfig, WeightSyncConfig

logger = logging.getLogger(__name__)


def validate_config(
    base_model: str,
    dataset: str,
    hotload: WeightSyncConfig | None = None,
    deploy: DeployConfig | None = None,
    output_model_id: str | None = None,
) -> None:
    """Pre-flight validation. Catches misconfiguration before provisioning GPUs."""
    errors: list[str] = []

    if not base_model:
        errors.append(
            format_sdk_error(
                "Missing base_model",
                "No base model specified.",
                "Set base_model (e.g. 'accounts/fireworks/models/qwen3-8b').",
            )
        )
    elif not base_model.startswith("accounts/"):
        errors.append(
            format_sdk_error(
                "Invalid base_model format",
                f"'{base_model}' doesn't match expected format.",
                "Use format: accounts/ACCOUNT/models/MODEL_NAME\n"
                "  Example: accounts/fireworks/models/qwen3-8b",
            )
        )

    if not dataset:
        errors.append(
            format_sdk_error(
                "Missing dataset",
                "No dataset path or URL specified.",
                "Set dataset to a local path or URL to a JSONL file.",
            )
        )

    if output_model_id is not None:
        errors.extend(validate_output_model_id(output_model_id))

    if errors:
        raise RuntimeError("\n\n".join(errors))


def validate_preflight(
    args,
    fw_api_key: str | None,
    *,
    skip_credential_check: bool = False,
) -> None:
    """Catch common CLI configuration issues before creating expensive resources."""
    errors: list[str] = []

    if not skip_credential_check:
        if not fw_api_key:
            errors.append(
                format_sdk_error(
                    "Missing FIREWORKS_API_KEY",
                    "No API key found in --fireworks-api-key or FIREWORKS_API_KEY env var.",
                    "export FIREWORKS_API_KEY='your-key-here'\n"
                    "  Get your API key at: https://fireworks.ai/account/api-keys",
                    docs_url=DOCS_SDK,
                )
            )

    if errors:
        raise RuntimeError("\n\n".join(errors))

    validate_config(
        base_model=getattr(args, "base_model", "") or "",
        dataset=getattr(args, "dataset", "") or "",
        hotload=WeightSyncConfig(
            weight_sync_interval=getattr(args, "weight_sync_interval", 0),
            weight_sync_before_training=getattr(args, "weight_sync_before_training", False),
        ),
        deploy=DeployConfig(
            deployment_id=getattr(args, "hot_load_deployment_id", None),
        ),
        output_model_id=getattr(args, "output_model_id", None),
    )
