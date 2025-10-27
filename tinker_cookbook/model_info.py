"""
This module associates model names with metadata, which helps  training code choose good defaults.
"""

from dataclasses import dataclass
from functools import cache


@dataclass
class ModelAttributes:
    organization: str  # meta-llama, Qwen, etc.
    version_str: str  # just the version number e.g. "3.1", "2.5"
    size_str: str  # size of the model e.g. "8B", "72B", "1.5B"
    is_chat: bool  # is chat/instruct model
    is_vl: bool = False  # is vision-language model


@cache
def get_llama_info() -> dict[str, ModelAttributes]:
    org = "meta-llama"
    return {
        "Llama-3.2-1B-Instruct": ModelAttributes(org, "3.2", "1B", True),
        "Llama-3.2-3B-Instruct": ModelAttributes(org, "3.2", "3B", True),
        "Llama-3.1-8B-Instruct": ModelAttributes(org, "3.1", "8B", True),
        "Llama-3.2-1B": ModelAttributes(org, "3.2", "1B", False),
        "Llama-3.2-3B": ModelAttributes(org, "3.2", "3B", False),
        "Llama-3.1-8B": ModelAttributes(org, "3.1", "8B", False),
        "Llama-3.1-70B": ModelAttributes(org, "3.1", "70B", False),
        "Llama-3.3-70B-Instruct": ModelAttributes(org, "3.3", "70B", True),
    }


def get_qwen_info() -> dict[str, ModelAttributes]:
    org = "Qwen"
    return {
        "Qwen3-4B-Base": ModelAttributes(org, "3", "4B", False),
        "Qwen3-8B-Base": ModelAttributes(org, "3", "8B", False),
        "Qwen3-14B-Base": ModelAttributes(org, "3", "14B", False),
        "Qwen3-30B-A3B-Base": ModelAttributes(org, "3", "30B-A3B", False),
        "Qwen3-0.6B": ModelAttributes(org, "3", "0.6B", True),
        "Qwen3-1.7B": ModelAttributes(org, "3", "1.7B", True),
        "Qwen3-4B": ModelAttributes(org, "3", "4B", True),
        "Qwen3-8B": ModelAttributes(org, "3", "8B", True),
        "Qwen3-14B": ModelAttributes(org, "3", "14B", True),
        "Qwen3-32B": ModelAttributes(org, "3", "32B", True),
        "Qwen3-30B-A3B": ModelAttributes(org, "3", "30B-A3B", True),
        "Qwen3-4B-Instruct-2507": ModelAttributes(org, "3", "4B", True),
        "Qwen3-30B-A3B-Instruct-2507": ModelAttributes(org, "3", "30B-A3B", True),
        "Qwen3-235B-A22B-Instruct-2507": ModelAttributes(org, "3", "235B-A22B", True),
    }


def get_deepseek_info() -> dict[str, ModelAttributes]:
    org = "deepseek-ai"
    return {
        "DeepSeek-V3.1": ModelAttributes(org, "3", "671B-A37B", True),
        "DeepSeek-V3.1-Base": ModelAttributes(org, "3", "671B-A37B", False),
    }


def get_gpt_oss_info() -> dict[str, ModelAttributes]:
    org = "openai"
    return {
        "gpt-oss-20b": ModelAttributes(org, "1", "21B-A3.6B", True),
        "gpt-oss-120b": ModelAttributes(org, "1", "117B-A5.1B", True),
    }


def get_model_attributes(model_name: str) -> ModelAttributes:
    org, model_version_full = model_name.split("/")
    if org == "meta-llama":
        return get_llama_info()[model_version_full]
    elif org == "Qwen":
        return get_qwen_info()[model_version_full]
    elif org == "deepseek-ai":
        return get_deepseek_info()[model_version_full]
    elif org == "openai":
        return get_gpt_oss_info()[model_version_full]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_recommended_renderer_names(model_name: str) -> list[str]:
    """
    Return a list of renderers that are designed for the model.
    Used so we can emit a warning if you use a non-recommended renderer.
    The first result is the most recommended renderer for the model.
    """
    attributes = get_model_attributes(model_name)
    if not attributes.is_chat:
        return ["role_colon"]
    elif attributes.organization == "meta-llama":
        return ["llama3"]
    elif attributes.organization == "Qwen":
        if attributes.version_str == "3":
            if "-Instruct" in model_name:
                return ["qwen3_instruct"]
            else:
                return ["qwen3", "qwen3_disable_thinking"]
        else:
            raise ValueError(f"Unknown model: {model_name}")
    elif attributes.organization == "deepseek-ai":
        return ["deepseekv3_disable_thinking", "deepseekv3"]
    elif attributes.organization == "openai":
        return ["gpt_oss_no_sysprompt", "gpt_oss_medium_reasoning"]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_recommended_renderer_name(model_name: str) -> str:
    """
    Return the most recommended renderer for the model.
    """
    return get_recommended_renderer_names(model_name)[0]
