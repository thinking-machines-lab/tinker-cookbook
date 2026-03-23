"""Grid generation utilities for parameter sweeps."""

from itertools import product
from typing import Any


def grid(axes: dict[str, list[Any]] | None = None, /, **kwargs: list[Any]) -> list[dict[str, Any]]:
    """Generate all parameter combinations from named axes (cartesian product).

    Accepts either a dict or keyword arguments (or both, merged).

    Example::

        >>> grid(learning_rate=[1e-4, 3e-4], lora_rank=[32, 128])
        [
            {"learning_rate": 1e-4, "lora_rank": 32},
            {"learning_rate": 1e-4, "lora_rank": 128},
            {"learning_rate": 3e-4, "lora_rank": 32},
            {"learning_rate": 3e-4, "lora_rank": 128},
        ]

    Args:
        axes: Optional dict mapping parameter names to lists of values.
        **kwargs: Additional parameter axes as keyword arguments.

    Returns:
        List of dicts, one per combination.
    """
    merged: dict[str, list[Any]] = {}
    if axes is not None:
        merged.update(axes)
    merged.update(kwargs)

    if not merged:
        return [{}]

    for name, values in merged.items():
        if not isinstance(values, list):
            raise TypeError(
                f"Sweep axis '{name}' must be a list of values, got {type(values).__name__}. "
                f"Did you mean {name}=[{values!r}]?"
            )
        if len(values) == 0:
            raise ValueError(f"Sweep axis '{name}' must have at least one value.")

    names = list(merged.keys())
    all_values = [merged[n] for n in names]
    return [dict(zip(names, combo)) for combo in product(*all_values)]


def default_run_name(overrides: dict[str, Any]) -> str:
    """Generate a deterministic, readable directory name from sweep overrides.

    Example::

        >>> default_run_name({"learning_rate": 3e-4, "lora_rank": 32})
        'learning_rate=0.0003_lora_rank=32'
    """
    parts = []
    for key, value in overrides.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:g}")
        else:
            parts.append(f"{key}={value}")
    return "_".join(parts)
