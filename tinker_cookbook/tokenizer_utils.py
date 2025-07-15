"""
Utilities for working with tokenizers. Create new types to avoid needing to import AutoTokenizer and PreTrainedTokenizer.


Avoid importing AutoTokenizer and PreTrainedTokenizer until runtime, because they're slow imports.
"""

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.tokenization_utils import PreTrainedTokenizer

type Tokenizer = "PreTrainedTokenizer"


@cache
def get_tokenizer(model_name: str) -> "Tokenizer":
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)
