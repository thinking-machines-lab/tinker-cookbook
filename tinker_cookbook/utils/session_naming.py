from __future__ import annotations

import re


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

    Recipe CLIs fill ``session_name`` themselves when the user does not pass one,
    using the run name they already generate for logging or, for recipes without
    one, :func:`default_cookbook_session_name`. This fallback chain therefore
    matters mainly for code that builds a training ``Config`` directly.
    """
    if session_name:
        return session_name
    if wandb_name:
        return wandb_name
    return None


def slugify(text: str) -> str:
    """Lowercase ``text`` and collapse anything outside ``[a-z0-9]`` into single hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def default_cookbook_session_name(*parts: str | None, wandb_name: str | None = None) -> str:
    """Build the default Tinker session name for a cookbook recipe.

    When ``wandb_name`` is set, it is returned unchanged so the Tinker session
    and the W&B run share a name. Pass only a name the user chose, not one the
    recipe generated as a fallback.

    Otherwise the result is a slug of ``parts`` prefixed with ``cookbook-`` so
    the session is recognizable as coming from the cookbook, for example
    ``default_cookbook_session_name("math-rl", "gsm8k")`` returns
    ``"cookbook-math-rl-gsm8k"``. Pass the recipe name first, followed by any
    variant that distinguishes runs of the same recipe (dataset, environment,
    pipeline stage). ``None`` parts are skipped.
    """
    if wandb_name:
        return wandb_name
    return "-".join(slug for part in ("cookbook", *parts) if part and (slug := slugify(part)))
