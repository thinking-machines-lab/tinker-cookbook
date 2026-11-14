"""Pinned p5.js + p5.brush library sources for the headless renderer.

The two scripts are fetched once from jsDelivr into a local cache and inlined
into every rendered page (the sandboxed page cannot load ``file://`` or
remote scripts). Set ``TINKER_PAINT_ASSETS_DIR`` to change the cache location,
or drop ``p5.min.js`` / ``p5.brush.js`` into it to work fully offline.
"""

from __future__ import annotations

import logging
import os
from functools import cache
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)

P5_VERSION = "1.11.3"
P5_BRUSH_VERSION = "1.1.4"  # last 1.x release; requires p5 ^1.11

P5_URL = f"https://cdn.jsdelivr.net/npm/p5@{P5_VERSION}/lib/p5.min.js"
P5_BRUSH_URL = f"https://cdn.jsdelivr.net/npm/p5.brush@{P5_BRUSH_VERSION}/dist/p5.brush.js"

# Sanity floors: a CDN error page is a few KB, the real files are far larger.
_MIN_BYTES = {"p5.min.js": 500_000, "p5.brush.js": 20_000}


def assets_dir() -> Path:
    configured = os.environ.get("TINKER_PAINT_ASSETS_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "tinker-cookbook" / "paint_rl"


def _fetch(url: str, target: Path) -> None:
    logger.info("Downloading %s -> %s", url, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        data = response.read()
    if len(data) < _MIN_BYTES[target.name]:
        raise RuntimeError(f"{url} returned {len(data)} bytes; expected a library file")
    target.write_bytes(data)


@cache
def get_library_sources() -> tuple[str, str]:
    """Return ``(p5_source, p5_brush_source)`` as JavaScript text."""
    directory = assets_dir()
    sources: list[str] = []
    for name, url in (("p5.min.js", P5_URL), ("p5.brush.js", P5_BRUSH_URL)):
        path = directory / name
        if not path.exists() or path.stat().st_size < _MIN_BYTES[name]:
            _fetch(url, path)
        sources.append(path.read_text(encoding="utf-8"))
    return sources[0], sources[1]
