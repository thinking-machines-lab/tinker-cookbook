"""
Utilities for working with tokenizers. Create new types to avoid needing to import AutoTokenizer and PreTrainedTokenizer.


Avoid importing AutoTokenizer and PreTrainedTokenizer until runtime, because they're slow imports.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast, runtime_checkable

# ---------------------------------------------------------------------------
# Optional ``tml_renderers`` import shim.
#
# ``tml_renderers`` is not part of the base cookbook dependency set; install
# the ``inkling`` extra (``pip install 'tinker-cookbook[inkling]'``) to use
# Inkling. The ``TML_RENDERERS_SOURCE_DIR`` escape hatch only exists so a
# source checkout can be used without installing.
# ---------------------------------------------------------------------------


def ensure_tml_renderers_importable() -> None:
    """Make ``tml_renderers`` importable, falling back to ``TML_RENDERERS_SOURCE_DIR``."""
    if importlib.util.find_spec("tml_renderers") is not None:
        return

    if env_path := os.environ.get("TML_RENDERERS_SOURCE_DIR"):
        source_dir = Path(env_path).expanduser()
        if (source_dir / "tml_renderers").exists():
            source_str = str(source_dir)
            inserted = False
            if source_str not in sys.path:
                sys.path.insert(0, source_str)
                inserted = True
            if importlib.util.find_spec("tml_renderers") is not None:
                return
            if inserted:
                sys.path.remove(source_str)

    raise ModuleNotFoundError(
        "Could not import optional dependency 'tml_renderers'. "
        "Install it with: uv pip install 'tinker-cookbook[inkling]', "
        "or set TML_RENDERERS_SOURCE_DIR to a directory containing the "
        "tml_renderers package."
    )


if TYPE_CHECKING:
    # this import takes a few seconds, so avoid it on the module import when possible
    from transformers import PreTrainedTokenizer

    Tokenizer: TypeAlias = PreTrainedTokenizer
else:
    # make it importable from other files as a type in runtime
    Tokenizer: TypeAlias = Any

# Global registry for custom tokenizer factories
_CUSTOM_TOKENIZER_REGISTRY: dict[str, Callable[[], Tokenizer]] = {}


class TmlTokenizer(Protocol):
    """Structural type for the underlying ``tml_renderers`` tokenizer.

    Cookbook never constructs this object (``tml_renderers.tokenizers`` does),
    and ``tml_renderers`` is an optional, lazily imported dependency, so its
    concrete type is not referenceable here. This protocol pins the small
    surface the cookbook adapter and renderer actually rely on, making the
    contract explicit instead of ``Any`` duck-typing.
    """

    bos_token: str
    eos_token: str

    def encode_ordinary(self, text: str) -> Sequence[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...

    def encode_special(self, text: str) -> int: ...


@runtime_checkable
class SupportsTmlTokenizer(Protocol):
    """Tokenizer facade that exposes an underlying :class:`TmlTokenizer`."""

    tml_tokenizer: TmlTokenizer


class TmlRenderersTokenizerAdapter:
    """Small tokenizer facade for ``tml_renderers`` models used through cookbook."""

    eos_token_id: int | None = None
    tml_tokenizer: TmlTokenizer

    def __init__(self, name_or_path: str):
        ensure_tml_renderers_importable()
        tokenizers = importlib.import_module("tml_renderers.tokenizers")

        self.name_or_path = name_or_path
        self.tml_tokenizer = tokenizers.o200k_base_chat()
        self.bos_token = self.tml_tokenizer.bos_token
        self.eos_token = self.tml_tokenizer.eos_token
        try:
            self.eos_token_id = int(self.tml_tokenizer.encode_special(self.eos_token))
        except Exception:
            self.eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False, **_: Any) -> list[int]:
        if add_special_tokens:
            raise ValueError("tml_renderers tokenizer does not support HF-style add_special_tokens")
        return list(self.tml_tokenizer.encode_ordinary(text))

    def decode(self, token_ids: list[int] | tuple[int, ...], **_: Any) -> str:
        return str(self.tml_tokenizer.decode(list(token_ids)))


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
    then returns a tml-renderers tokenizer for Inkling, or falls back to
    HuggingFace ``AutoTokenizer``. HuggingFace tokenizers are cached after first load.

    Args:
        model_name (str): HuggingFace model identifier (e.g.
            ``"Qwen/Qwen3-8B"``) or a custom registered name.

    Returns:
        Tokenizer: A tml-renderers tokenizer for Inkling, otherwise a
            HuggingFace ``PreTrainedTokenizer``.

    Example::

        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer("Qwen/Qwen3-8B")
        tokens = tokenizer.encode("Hello world")
    """
    # Check custom registry first (not cached, factory handles caching if needed)
    if (tokenizer := _CUSTOM_TOKENIZER_REGISTRY.get(model_name)) is not None:
        return tokenizer()

    base_model_name = model_name.split(":", 1)[0]
    if base_model_name == "thinkingmachines/Inkling":
        # Duck-typed facade (encode/decode/eos_token_id); not a PreTrainedTokenizer.
        return cast(Tokenizer, TmlRenderersTokenizerAdapter(model_name))

    return _get_hf_tokenizer(model_name)


# Pinned revisions for Kimi K2 tokenizers, loaded directly via the custom
# TikTokenTokenizer class (see _get_hf_tokenizer).
_KIMI_TOKENIZER_REVISIONS: dict[str, str] = {
    "moonshotai/Kimi-K2-Thinking": "a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55",
    "moonshotai/Kimi-K2.5": "2426b45b6af0da48d0dcce71bbce6225e5c73adc",
    "moonshotai/Kimi-K2.6": "b5aabbfb20227ed42becbf5541dbffd213942c58",
}


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

    # Kimi K2 models require the custom TikTokenTokenizer, which overrides
    # apply_chat_template to format tool declarations as TypeScript. AutoTokenizer
    # cannot be relied on to resolve it: on some platforms (x86_64 + certain
    # transformers releases) the model is forced through TokenizersBackend, and on
    # K2.5/K2.6 AutoTokenizer.from_pretrained raises outright ("Couldn't instantiate
    # the backend tokenizer ...") instead of using Kimi's tokenization_kimi auto-map.
    # Load the custom class directly so the result is correct regardless of the
    # installed transformers version.
    if model_name.startswith("moonshotai/Kimi-K2"):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        revision = _KIMI_TOKENIZER_REVISIONS.get(model_name)
        cls = get_class_from_dynamic_module(
            "tokenization_kimi.TikTokenTokenizer", model_name, revision=revision
        )
        return cls.from_pretrained(model_name, revision=revision)

    return AutoTokenizer.from_pretrained(model_name, **kwargs)
