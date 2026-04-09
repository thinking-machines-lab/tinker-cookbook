"""Registry cache -- maps source URI sets to cached RunRegistry instances."""

import logging
import time
from typing import Any

from tinker_cookbook.stores import RunRegistry
from tinker_cookbook.stores.storage import LocalStorage, Storage, storage_from_uri

logger = logging.getLogger(__name__)

_cache: dict[frozenset[str], tuple[RunRegistry, float]] = {}
_default_registry: RunRegistry | None = None
_default_sources: list[Storage] = []


def init_default(storages: list[Storage]) -> RunRegistry:
    """Initialize with startup sources (from CLI). Returns the default registry."""
    global _default_registry, _default_sources
    _default_sources = storages
    _default_registry = RunRegistry(storages)
    _default_registry.refresh()
    return _default_registry


def get_registry(sources: list[str] | None = None) -> RunRegistry:
    """Get a registry for the given source URIs. Caches by source-set.

    If sources is empty/None, returns the default (startup) registry.
    """
    if not sources:
        if _default_registry is None:
            raise RuntimeError("Registry not initialized -- call init_default() first")
        return _default_registry

    key = frozenset(sources)
    if key in _cache:
        return _cache[key][0]

    storages = [storage_from_uri(s) for s in sources]
    registry = RunRegistry(storages)
    registry.refresh()
    _cache[key] = (registry, time.time())
    return registry


def refresh_registry(sources: list[str] | None = None) -> int:
    """Refresh the registry for the given sources. Returns run count."""
    if not sources:
        if _default_registry is None:
            raise RuntimeError("Registry not initialized")
        runs = _default_registry.refresh()
        return len(runs)

    key = frozenset(sources)
    if key in _cache:
        runs = _cache[key][0].refresh()
        return len(runs)

    # Not cached yet -- create and cache
    registry = get_registry(sources)
    return len(registry.get_runs())


def get_default_sources() -> list[dict[str, str]]:
    """Return the startup sources for UI defaults."""
    result = []
    for s in _default_sources:
        result.append({
            "url": s.url(),
            "type": "local" if isinstance(s, LocalStorage) else "cloud",
        })
    return result
