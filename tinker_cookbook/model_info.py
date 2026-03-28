"""
This module associates model names with metadata, which helps  training code choose good defaults.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache

from tinker_cookbook.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Common renderer tuples, defined once to reduce repetition.
# Tuples (not lists) because these are shared across ModelAttributes instances
# — a mutable list would risk silent cross-model corruption if mutated.
_LLAMA3 = ("llama3",)
_ROLE_COLON = ("role_colon",)
_QWEN3 = ("qwen3", "qwen3_disable_thinking")
_QWEN3_INSTRUCT = ("qwen3_instruct",)
_QWEN3_VL = ("qwen3_vl",)
_QWEN3_VL_INSTRUCT = ("qwen3_vl_instruct",)
_QWEN3_5 = ("qwen3_5", "qwen3_5_disable_thinking")
_DEEPSEEKV3 = ("deepseekv3", "deepseekv3_thinking")
_GPT_OSS = ("gpt_oss_no_sysprompt", "gpt_oss_medium_reasoning")
_KIMI_K2 = ("kimi_k2",)
_KIMI_K25 = ("kimi_k25", "kimi_k25_disable_thinking")
_NEMOTRON3 = ("nemotron3", "nemotron3_disable_thinking")


@dataclass
class ModelAttributes:
    """Metadata describing a model's organization, version, size, and recommended renderers."""

    organization: str  # meta-llama, Qwen, etc.
    version_str: str  # just the version number e.g. "3.1", "2.5"
    size_str: str  # size of the model e.g. "8B", "72B", "1.5B"
    is_chat: bool  # is chat/instruct model
    recommended_renderers: tuple[str, ...]  # first entry is the most recommended
    is_vl: bool = False  # is vision-language model


@cache
def get_llama_info() -> dict[str, ModelAttributes]:
    org = "meta-llama"
    return {
        "Llama-3.2-1B-Instruct": ModelAttributes(org, "3.2", "1B", True, _LLAMA3),
        "Llama-3.2-3B-Instruct": ModelAttributes(org, "3.2", "3B", True, _LLAMA3),
        "Llama-3.1-8B-Instruct": ModelAttributes(org, "3.1", "8B", True, _LLAMA3),
        "Llama-3.2-1B": ModelAttributes(org, "3.2", "1B", False, _ROLE_COLON),
        "Llama-3.2-3B": ModelAttributes(org, "3.2", "3B", False, _ROLE_COLON),
        "Llama-3.1-8B": ModelAttributes(org, "3.1", "8B", False, _ROLE_COLON),
        "Llama-3.1-70B": ModelAttributes(org, "3.1", "70B", False, _ROLE_COLON),
        "Llama-3.3-70B-Instruct": ModelAttributes(org, "3.3", "70B", True, _LLAMA3),
    }


@cache
def get_qwen_info() -> dict[str, ModelAttributes]:
    org = "Qwen"
    return {
        "Qwen3-VL-30B-A3B-Instruct": ModelAttributes(
            org, "3", "30B-A3B", True, _QWEN3_VL_INSTRUCT, is_vl=True
        ),
        "Qwen3-VL-235B-A22B-Instruct": ModelAttributes(
            org, "3", "235B-A22B", True, _QWEN3_VL_INSTRUCT, is_vl=True
        ),
        "Qwen3-4B-Base": ModelAttributes(org, "3", "4B", False, _ROLE_COLON),
        "Qwen3-8B-Base": ModelAttributes(org, "3", "8B", False, _ROLE_COLON),
        "Qwen3-14B-Base": ModelAttributes(org, "3", "14B", False, _ROLE_COLON),
        "Qwen3-30B-A3B-Base": ModelAttributes(org, "3", "30B-A3B", False, _ROLE_COLON),
        "Qwen3-0.6B": ModelAttributes(org, "3", "0.6B", True, _QWEN3),
        "Qwen3-1.7B": ModelAttributes(org, "3", "1.7B", True, _QWEN3),
        "Qwen3-4B": ModelAttributes(org, "3", "4B", True, _QWEN3),
        "Qwen3-8B": ModelAttributes(org, "3", "8B", True, _QWEN3),
        "Qwen3-14B": ModelAttributes(org, "3", "14B", True, _QWEN3),
        "Qwen3-32B": ModelAttributes(org, "3", "32B", True, _QWEN3),
        "Qwen3-30B-A3B": ModelAttributes(org, "3", "30B-A3B", True, _QWEN3),
        "Qwen3-4B-Instruct-2507": ModelAttributes(org, "3", "4B", True, _QWEN3_INSTRUCT),
        "Qwen3-30B-A3B-Instruct-2507": ModelAttributes(org, "3", "30B-A3B", True, _QWEN3_INSTRUCT),
        "Qwen3-235B-A22B-Instruct-2507": ModelAttributes(
            org, "3", "235B-A22B", True, _QWEN3_INSTRUCT
        ),
        "Qwen3.5-4B": ModelAttributes(org, "3.5", "4B", True, _QWEN3_5, is_vl=True),
        "Qwen3.5-27B": ModelAttributes(org, "3.5", "27B", True, _QWEN3_5, is_vl=True),
        "Qwen3.5-35B-A3B": ModelAttributes(org, "3.5", "35B-A3B", True, _QWEN3_5, is_vl=True),
        "Qwen3.5-397B-A17B": ModelAttributes(org, "3.5", "397B-A17B", True, _QWEN3_5, is_vl=True),
    }


@cache
def get_deepseek_info() -> dict[str, ModelAttributes]:
    org = "deepseek-ai"
    return {
        "DeepSeek-V3.1": ModelAttributes(org, "3", "671B-A37B", True, _DEEPSEEKV3),
        "DeepSeek-V3.1-Base": ModelAttributes(org, "3", "671B-A37B", False, _ROLE_COLON),
    }


@cache
def get_gpt_oss_info() -> dict[str, ModelAttributes]:
    org = "openai"
    return {
        "gpt-oss-20b": ModelAttributes(org, "1", "21B-A3.6B", True, _GPT_OSS),
        "gpt-oss-120b": ModelAttributes(org, "1", "117B-A5.1B", True, _GPT_OSS),
    }


@cache
def get_moonshot_info() -> dict[str, ModelAttributes]:
    org = "moonshotai"
    return {
        "Kimi-K2-Thinking": ModelAttributes(org, "K2", "1T-A32B", True, _KIMI_K2),
        "Kimi-K2.5": ModelAttributes(org, "K2.5", "1T-A32B", True, _KIMI_K25, is_vl=True),
    }


@cache
def get_nvidia_info() -> dict[str, ModelAttributes]:
    org = "nvidia"
    return {
        "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": ModelAttributes(
            org, "3", "30B-A3B", True, _NEMOTRON3
        ),
        "NVIDIA-Nemotron-3-Super-120B-A12B-BF16": ModelAttributes(
            org, "3", "120B-A12B", True, _NEMOTRON3
        ),
    }


def get_model_attributes(model_name: str) -> ModelAttributes:
    model_name = model_name.split(":")[0]
    org, model_version_full = model_name.split("/")
    model_version_full = model_version_full.split(":")[0]
    if org == "meta-llama":
        return get_llama_info()[model_version_full]
    elif org == "Qwen":
        return get_qwen_info()[model_version_full]
    elif org == "deepseek-ai":
        return get_deepseek_info()[model_version_full]
    elif org == "openai":
        return get_gpt_oss_info()[model_version_full]
    elif org == "moonshotai":
        return get_moonshot_info()[model_version_full]
    elif org == "nvidia":
        return get_nvidia_info()[model_version_full]
    else:
        raise ConfigurationError(f"Unknown model: {model_name}")


def get_recommended_renderer_names(model_name: str) -> list[str]:
    """
    Return a list of renderers that are designed for the model.
    Used so we can emit a warning if you use a non-recommended renderer.
    The first result is the most recommended renderer for the model.
    """
    return list(get_model_attributes(model_name).recommended_renderers)


def get_recommended_renderer_name(model_name: str) -> str:
    """
    Return the most recommended renderer for the model.
    """
    return get_recommended_renderer_names(model_name)[0]


def warn_if_renderer_not_recommended(model_name: str, renderer_name: str | None) -> None:
    """
    Log a warning if ``renderer_name`` is not in the recommended list for ``model_name``.

    Silently returns if ``renderer_name`` is None (caller is using the default) or if
    ``model_name`` is not in the model registry.
    """
    if renderer_name is None:
        return
    try:
        recommended = get_recommended_renderer_names(model_name)
    except (ConfigurationError, KeyError, ValueError):
        # Unknown model — nothing to validate against.
        return
    if renderer_name not in recommended:
        logger.warning(
            "Renderer %r is not recommended for model %r. "
            "Recommended renderer(s): %s. "
            "Using an incompatible renderer can silently degrade training quality "
            "(e.g., prefilling tokens the model was never trained on).",
            renderer_name,
            model_name,
            ", ".join(repr(r) for r in recommended),
        )
