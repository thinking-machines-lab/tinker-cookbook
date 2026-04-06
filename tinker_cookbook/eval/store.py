"""Backward-compatible re-export of EvalStore from stores/.

The canonical implementation now lives in ``tinker_cookbook.stores.eval_store``.

Note: Imports are lazy to avoid circular dependency with stores/eval_store.py
which imports from eval/benchmarks/_types which triggers eval/__init__.py.
"""


def __getattr__(name: str):
    """Lazy import to break circular dependency chain."""
    if name in ("EvalStore", "RunComparison", "RunMetadata"):
        from tinker_cookbook.stores.eval_store import EvalStore, RunComparison, RunMetadata

        mapping = {
            "EvalStore": EvalStore,
            "RunComparison": RunComparison,
            "RunMetadata": RunMetadata,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
