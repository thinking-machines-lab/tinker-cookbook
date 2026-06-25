"""RILL: a small, deterministic DSL used as an out-of-distribution coding target.

The grammar (``rill.lark``) and the reference interpreter (``interp.py``) are the
language spec. ``run_rill`` runs a program and returns a :class:`Result` with the
captured ``emit`` output and a coarse, stable error category (``parse:`` / ``runtime:``
/ ``budget:``) suitable for reward shaping.
"""

from .interp import Result, run_rill

__all__ = ["Result", "run_rill"]
