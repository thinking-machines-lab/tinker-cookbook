from __future__ import annotations


def resolve_session_display_name(
    *,
    session_name: str | None,
    wandb_name: str | None,
) -> str | None:
    """Pick the human-readable ``display_name`` for a recipe's Tinker session.

    The display name shows up in the Tinker console and session listings; it does
    not need to be unique. Resolution order:

    1. ``session_name`` when the user set one explicitly.
    2. ``wandb_name`` when the run is logging to Weights & Biases, so the Tinker
       session and the W&B run share a name.
    3. ``None`` otherwise; the recipe name is already attached via ``user_metadata``.

    Empty strings are treated as unset.
    """
    if session_name:
        return session_name
    if wandb_name:
        return wandb_name
    return None
