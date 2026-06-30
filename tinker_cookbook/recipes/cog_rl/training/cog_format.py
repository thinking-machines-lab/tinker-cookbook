"""Render Python values the way Cog renders them — the adapter that lets us grade Cog
against references written in *any* language.

Two directions:

- ``cog_repr(v)`` formats a Python value exactly as Cog's ``emit`` prints it (so an
  expected output computed from a Python reference matches what a correct Cog ``solve``
  would print). It mirrors the interpreter's ``_to_str`` verbatim.
- ``cog_literal(v)`` renders a Python value as Cog *source* (a call argument), with proper
  string escaping for the grammar's ``\\n \\t \\" \\\\`` escapes.
- ``supported(v)`` is the type filter: Cog has int / text / flag / list (no float, dict,
  set, tuple, None as a value-to-print). We keep only tasks whose inputs and outputs land
  in that subset, so the I/O is reproducible in Cog.
"""

from __future__ import annotations

# Mirrors interp._to_str. Imported lazily/duplicated here to avoid a hard import at module
# load; kept identical on purpose (there is a test asserting they agree).
SUPPORTED_SCALARS = (bool, int, str)


def cog_repr(v) -> str:
    """Format ``v`` as Cog's ``emit`` would print it. Must match interp._to_str."""
    if v is None:
        return "void"
    if v is True:
        return "yes"
    if v is False:
        return "no"
    if isinstance(v, list):
        return "[" + ", ".join(cog_repr(x) for x in v) + "]"
    return str(v)


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")


def cog_literal(v) -> str:
    """Render ``v`` as a Cog source literal (e.g. a call argument)."""
    if v is None:
        return "void"
    if v is True:
        return "yes"
    if v is False:
        return "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return '"' + _escape(v) + '"'
    if isinstance(v, list):
        return "[" + ", ".join(cog_literal(x) for x in v) + "]"
    raise TypeError(f"unsupported literal type: {type(v).__name__}")


def cog_args(args: tuple | list) -> str:
    return ", ".join(cog_literal(a) for a in args)


def supported(v) -> bool:
    """True iff ``v`` is a value Cog can represent and print unambiguously.

    bool / int / text, or a (possibly nested) list of supported values. Floats are excluded
    (Cog reals exist but their print formatting differs from Python's ``str(float)``), as are
    dict / set / tuple / None-as-value.
    """
    if isinstance(v, bool):
        return True
    if isinstance(v, int):
        return True
    if isinstance(v, str):
        return True
    if isinstance(v, list):
        return all(supported(x) for x in v)
    return False
