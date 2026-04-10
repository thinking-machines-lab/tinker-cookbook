"""
Utilities for working with tokenizers. Create new types to avoid needing to import AutoTokenizer and PreTrainedTokenizer.


Avoid importing AutoTokenizer and PreTrainedTokenizer until runtime, because they're slow imports.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    # this import takes a few seconds, so avoid it on the module import when possible
    from transformers import PreTrainedTokenizer

    Tokenizer: TypeAlias = PreTrainedTokenizer
else:
    # make it importable from other files as a type in runtime
    Tokenizer: TypeAlias = Any

# Global registry for custom tokenizer factories
_CUSTOM_TOKENIZER_REGISTRY: dict[str, Callable[[], Tokenizer]] = {}


def register_tokenizer(
    name: str,
    factory: Callable[[], Tokenizer],
) -> None:
    """Register a custom tokenizer factory.

    Once registered, :func:`get_tokenizer` will call ``factory()`` instead
    of loading from HuggingFace when the given ``name`` is requested.

    Args:
        name (str): The tokenizer name (typically a HuggingFace model ID
            like ``"Foo/foo_tokenizer"``).
        factory (Callable[[], Tokenizer]): A callable that takes no arguments
            and returns a Tokenizer instance.

    Example::

        def my_tokenizer_factory():
            return MyCustomTokenizer()

        register_tokenizer("Foo/foo_tokenizer", my_tokenizer_factory)
    """
    _CUSTOM_TOKENIZER_REGISTRY[name] = factory


def get_registered_tokenizer_names() -> list[str]:
    """Return a list of all registered custom tokenizer names.

    Returns:
        list[str]: Names of all tokenizers registered via
            :func:`register_tokenizer`.
    """
    return list(_CUSTOM_TOKENIZER_REGISTRY.keys())


def is_tokenizer_registered(name: str) -> bool:
    """Check if a tokenizer name is registered.

    Args:
        name (str): The tokenizer name to check.

    Returns:
        bool: True if the name has been registered via
            :func:`register_tokenizer`.
    """
    return name in _CUSTOM_TOKENIZER_REGISTRY


def unregister_tokenizer(name: str) -> bool:
    """Unregister a custom tokenizer factory.

    Args:
        name: The tokenizer name to unregister.

    Returns:
        True if the tokenizer was unregistered, False if it wasn't registered.
    """
    if name in _CUSTOM_TOKENIZER_REGISTRY:
        del _CUSTOM_TOKENIZER_REGISTRY[name]
        return True
    return False


def get_tokenizer(model_name: str) -> Tokenizer:
    """Get a tokenizer by name.

    Checks the custom registry first (see :func:`register_tokenizer`),
    then falls back to HuggingFace ``AutoTokenizer``. HuggingFace
    tokenizers are cached after first load.

    Args:
        model_name (str): HuggingFace model identifier (e.g.
            ``"Qwen/Qwen3-8B"``) or a custom registered name.

    Returns:
        Tokenizer: A ``PreTrainedTokenizer`` instance.

    Example::

        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer("Qwen/Qwen3-8B")
        tokens = tokenizer.encode("Hello world")
    """
    # Check custom registry first (not cached, factory handles caching if needed)
    if (tokenizer := _CUSTOM_TOKENIZER_REGISTRY.get(model_name)) is not None:
        return tokenizer()

    return _get_hf_tokenizer(model_name)


@cache
def _get_hf_tokenizer(model_name: str) -> Tokenizer:
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    model_name = model_name.split(":")[0]

    # Local directory path — always trust custom tokenizer code bundled alongside.
    if os.path.isdir(model_name):
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Avoid gating of Llama 3 models:
    if model_name.startswith("meta-llama/Llama-3"):
        model_name = "thinkingmachineslabinc/meta-llama-3-instruct-tokenizer"

    kwargs: dict[str, Any] = {}
    if os.environ.get("HF_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes"):
        kwargs["trust_remote_code"] = True

    if model_name == "moonshotai/Kimi-K2-Thinking":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55"
    elif model_name == "moonshotai/Kimi-K2.5":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "2426b45b6af0da48d0dcce71bbce6225e5c73adc"

    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

    # Kimi models require the custom TikTokenTokenizer which overrides apply_chat_template
    # to format tool declarations as TypeScript. On some platforms (x86_64 + transformers
    # >=5.5), AutoTokenizer resolves to TokenizersBackend instead. Bypass AutoTokenizer
    # and directly load the custom class in that case.
    if (
        model_name.startswith("moonshotai/Kimi-K2")
        and "apply_chat_template" not in type(tokenizer).__dict__
    ):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        revision = kwargs.get("revision")
        cls = get_class_from_dynamic_module(
            "tokenization_kimi.TikTokenTokenizer", model_name, revision=revision
        )
        tokenizer = cls.from_pretrained(model_name, revision=revision)

    return tokenizer
