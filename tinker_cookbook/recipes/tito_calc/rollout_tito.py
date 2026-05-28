"""TITO-driven training-sample construction.

Builds ``(token_ids, weights)`` using *only* ``tokenizer.apply_chat_template``.
No imports from ``tinker_cookbook.renderers``.

Two modes, in preference order:

1. **Template-native masking** — when the chat template carries
   ``{% generation %}`` / ``{% endgeneration %}`` markers (Laguna XS.2, TRL's
   training templates, etc.), ``apply_chat_template(..., return_assistant_tokens_mask=True)``
   produces tokens *and* the assistant-only loss mask in one call. No
   per-family Python at all.
2. **Per-family header split** — fallback for templates without generation
   markers. Splits each message's render delta into ``[header] + [output]``
   using a known role-header string (e.g. ``<|im_start|>{role}\\n`` for Qwen3).
   That's a 2-line lookup, not a 100-line ``render_message``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from transformers import AutoTokenizer

from tinker_cookbook.recipes.tito_calc.env import Message

TrainOnWhatStr = Literal[
    "ALL_ASSISTANT_MESSAGES",
    "LAST_ASSISTANT_MESSAGE",
    "LAST_ASSISTANT_TURN",
    "ALL_MESSAGES",
]


@dataclass(frozen=True)
class FamilyHeaders:
    """Per-family header-string format, used only for the loss-mask split
    *fallback* (when the template lacks ``{% generation %}`` markers)."""

    bos_str: str
    header_template: str

    def header_str(self, role: str) -> str:
        return self.header_template.format(role=role)


LLAMA3 = FamilyHeaders(
    bos_str="<|begin_of_text|>",
    header_template="<|start_header_id|>{role}<|end_header_id|>\n\n",
)
QWEN3 = FamilyHeaders(
    bos_str="",
    header_template="<|im_start|>{role}\n",
)


def template_has_generation_markers(chat_template: str | None) -> bool:
    if not chat_template:
        return False
    return "{% generation %}" in chat_template or "{%- generation -%}" in chat_template


def build_via_tito(
    messages: list[Message],
    model_name: str,
    *,
    family: FamilyHeaders | None = None,
    chat_template: str | None = None,
    train_on_what: TrainOnWhatStr = "ALL_ASSISTANT_MESSAGES",
) -> tuple[list[int], list[int]]:
    """Build ``(token_ids, weights)`` without any tinker_cookbook.renderers code.

    When the template carries ``{% generation %}`` markers, this is one
    ``apply_chat_template`` call. Otherwise we fall back to a per-message
    header-split using ``family`` (required only in the fallback case).
    """
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    effective_template = chat_template or tok.chat_template

    # --- preferred path: generation markers in the template ---
    if template_has_generation_markers(effective_template):
        out = tok.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **({"chat_template": chat_template} if chat_template else {}),
        )
        ids: list[int] = list(out["input_ids"])
        mask: list[int] = [int(m) for m in out["assistant_masks"]]
        if train_on_what == "ALL_ASSISTANT_MESSAGES":
            return ids, mask
        # The template hands us "every assistant token." For other modes, we
        # need per-message boundaries; fall through to the header-split path.

    # --- fallback path: per-message header split ---
    if family is None:
        raise ValueError(
            "Template lacks {% generation %} markers; pass `family` so we can "
            "split each message's delta into header vs output for loss masking."
        )

    def render(msgs: list[Message]) -> list[int]:
        kwargs = {"chat_template": chat_template} if chat_template else {}
        return tok.apply_chat_template(msgs, tokenize=True, return_dict=False, **kwargs)

    full_ids = render(messages)
    prefix_lens: list[int] = [0]
    for i in range(1, len(messages) + 1):
        ids_i = render(messages[:i])
        prefix_lens.append(len(ids_i))

    weights = [0] * len(full_ids)
    bos_ids: list[int] = (
        tok.encode(family.bos_str, add_special_tokens=False) if family.bos_str else []
    )

    last_user_idx = max(
        (i for i, m in enumerate(messages) if m["role"] == "user"), default=-1
    )

    for i, msg in enumerate(messages):
        start = prefix_lens[i]
        end = prefix_lens[i + 1]
        role = msg["role"]
        header_start = start + (len(bos_ids) if i == 0 else 0)
        header_ids = tok.encode(family.header_str(role), add_special_tokens=False)
        output_start = min(header_start + len(header_ids), end)

        is_assistant = role == "assistant"
        is_last_message = i == len(messages) - 1
        is_after_last_user = last_user_idx == -1 or i > last_user_idx

        if train_on_what == "ALL_ASSISTANT_MESSAGES":
            has_weight = is_assistant
        elif train_on_what == "LAST_ASSISTANT_MESSAGE":
            has_weight = is_assistant and is_last_message
        elif train_on_what == "LAST_ASSISTANT_TURN":
            has_weight = is_assistant and is_after_last_user
        elif train_on_what == "ALL_MESSAGES":
            has_weight = True
        else:
            raise ValueError(f"unknown train_on_what: {train_on_what}")

        if has_weight:
            for j in range(output_start, end):
                weights[j] = 1

    return full_ids, weights


def prefix_preserved(
    messages: list[Message],
    model_name: str,
    chat_template: str | None = None,
) -> bool:
    """True iff every cumulative render extends the final one byte-for-byte."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    kwargs = {"chat_template": chat_template} if chat_template else {}
    full = tok.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)
    for i in range(1, len(messages) + 1):
        ids_i = tok.apply_chat_template(
            messages[:i], tokenize=True, return_dict=False, **kwargs
        )
        if full[: len(ids_i)] != ids_i:
            return False
    return True
