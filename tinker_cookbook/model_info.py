"""
This module associates model names with metadata, which helps  training code choose good defaults.
"""

from dataclasses import dataclass
from typing import cast


@dataclass
class ModelAttributes:
    organization: str  # meta-llama, Qwen, etc.
    version_str: str  # just the version number e.g. "3.1", "2.5"
    size_str: str  # size of the model e.g. "8B", "72B", "1.5B"
    is_chat: bool  # is chat/instruct model
    is_vl: bool = False  # is vision-language model


def get_model_attributes(model_name: str) -> ModelAttributes:
    org, model_version_full = model_name.split("/")
    if org == "meta-llama":
        return {
            "Llama-3.2-1B-Instruct": ModelAttributes(org, "3.2", "1B", True),
            "Llama-3.2-3B-Instruct": ModelAttributes(org, "3.2", "3B", True),
            "Llama-3.1-8B-Instruct": ModelAttributes(org, "3.1", "8B", True),
            "Llama-3.2-1B": ModelAttributes(org, "3.2", "1B", False),
            "Llama-3.2-3B": ModelAttributes(org, "3.2", "3B", False),
            "Llama-3.1-8B": ModelAttributes(org, "3.1", "8B", False),
            "Llama-3.1-70B": ModelAttributes(org, "3.1", "70B", False),
        }[model_version_full]
    elif org == "Qwen":
        return {
            "Qwen2.5-VL-3B-Instruct": ModelAttributes(org, "2.5", "3B", True, is_vl=True),
            "Qwen2.5-VL-7B-Instruct": ModelAttributes(org, "2.5", "7B", True, is_vl=True),
            "Qwen2.5-VL-32B-Instruct": ModelAttributes(org, "2.5", "32B", True, is_vl=True),
            "Qwen2.5-VL-72B-Instruct": ModelAttributes(org, "2.5", "72B", True, is_vl=True),
            "Qwen3-0.6B-Base": ModelAttributes(org, "3", "0.6B", False),
            "Qwen3-1.7B-Base": ModelAttributes(org, "3", "1.7B", False),
            "Qwen3-4B-Base": ModelAttributes(org, "3", "4B", False),
            "Qwen3-8B-Base": ModelAttributes(org, "3", "8B", False),
            "Qwen3-14B-Base": ModelAttributes(org, "3", "14B", False),
            "Qwen3-32B-Base": ModelAttributes(org, "3", "32B", False),
            "Qwen3-30B-A3B-Base": ModelAttributes(org, "3", "30B-A3B", False),
            "Qwen3-0.6B": ModelAttributes(org, "3", "0.6B", True),
            "Qwen3-1.7B": ModelAttributes(org, "3", "1.7B", True),
            "Qwen3-4B": ModelAttributes(org, "3", "4B", True),
            "Qwen3-8B": ModelAttributes(org, "3", "8B", True),
            "Qwen3-14B": ModelAttributes(org, "3", "14B", True),
            "Qwen3-32B": ModelAttributes(org, "3", "32B", True),
        }[model_version_full]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_recommended_renderer_names(model_name: str) -> list[str]:
    """
    Used so we can emit a warning if you use a non-recommended renderer.
    The first result is the most recommended renderer for the model.
    """
    attributes = get_model_attributes(model_name)
    if not attributes.is_chat:
        return ["role_colon"]
    elif attributes.organization == "meta-llama":
        return ["llama3"]
    elif attributes.organization == "Qwen":
        return ["qwen2p5", "qwen3"]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_recommended_renderer_name(model_name: str) -> str:
    return get_recommended_renderer_names(model_name)[0]


def main():
    import tinker

    sc = tinker.ServiceClient()
    for sm in sc.get_server_capabilities().supported_models:
        name = cast(str, sm.model_name)  # TODO remove cast after fixing type
        try:
            attributes = get_model_attributes(name)
        except KeyError:
            print(f"Unknown model: {name}")
            continue
        print(f"{name}: {attributes}")


if __name__ == "__main__":
    main()
