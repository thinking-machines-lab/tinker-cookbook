"""
Utilities for working with tokenizers. Create new types to avoid needing to import AutoTokenizer and PreTrainedTokenizer.


Avoid importing AutoTokenizer and PreTrainedTokenizer until runtime, because they're slow imports.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    # this import takes a few seconds, so avoid it on the module import when possible
    from transformers.tokenization_utils import PreTrainedTokenizer

    Tokenizer: TypeAlias = PreTrainedTokenizer
else:
    # make it importable from other files as a type in runtime
    Tokenizer: TypeAlias = Any


# Llama 3 Instruct chat template (the mirror tokenizer doesn't include it)
# Only needed by unit tests -- we don't use chat templates in this library
# (TODO: fix the mirrored tokenizer to include the chat template)

LLAMA3_CHAT_TEMPLATE = """\
{{- bos_token }}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""


@cache
def get_tokenizer(model_name: str) -> Tokenizer:
    """
    This function is functionally equivalent to AutoTokenizer.from_pretrained except it
    - uses the thinkingmachineslabinc mirror for Llama 3 models to avoid needing the HF token
      (which was intended to gate access to the model weights, not the tokenizer)
    - avoids the slow import of AutoTokenizer
    """
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    # Avoid gating of Llama 3 models:
    is_llama3 = model_name.startswith("meta-llama/Llama-3")
    if is_llama3:
        model_name = "thinkingmachineslabinc/meta-llama-3-tokenizer"

    kwargs: dict[str, Any] = {}
    if model_name == "moonshotai/Kimi-K2-Thinking":
        kwargs["trust_remote_code"] = True
        kwargs["revision"] = "612681931a8c906ddb349f8ad0f582cb552189cd"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, **kwargs, local_files_only=True
    )

    # The Llama 3 mirror tokenizer doesn't include the chat template, so add it
    if is_llama3 and tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    return tokenizer
