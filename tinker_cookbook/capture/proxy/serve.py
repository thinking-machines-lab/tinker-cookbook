"""Capture proxy entry point.

Run as::

    python -m tinker_cookbook.capture.proxy.serve --port 7462 \\
        --base-model Qwen/Qwen3-8B [--model-path tinker://...] \\
        [--store-data-dir DIR]

Startup wires the capture pipeline end to end: ``ensure_daemon`` spawns (or reuses) the
local capture store, ``instrument_tinker`` is armed with a
``CaptureExporter`` feeding a ``StoreSink``, and every proxied request then
samples through the instrumented SDK inside its address's capture scope.
Shutdown drains the exporter (wait_pending, force_flush, shutdown) before the
process exits.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path
from typing import cast

import tinker
from aiohttp import web

from tinker_cookbook.capture.exporter import CaptureExporter
from tinker_cookbook.capture.instrument import instrument_tinker, uninstrument_tinker
from tinker_cookbook.capture.proxy.app import ProxyDeps, SamplingClientLike, make_app
from tinker_cookbook.capture.store.client import StoreSink, ensure_daemon
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

_DEFAULT_DATA_DIR = Path("~/.cache/tinker-capture").expanduser()
_DRAIN_TIMEOUT_SEC = 5.0


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Unknown hostnames may resolve anywhere; treat as non-loopback.
        return False


def validate_bind(host: str, auth_token: str | None) -> str | None:
    """Normalize the auth token and refuse an unauthenticated non-loopback bind.

    An empty or whitespace-only token (e.g. ``TINKER_PROXY_AUTH_TOKEN=""``)
    is treated as UNSET everywhere: it neither authorizes a non-loopback
    bind nor installs auth middleware that would reject every request on a
    documented tokenless loopback deployment.

    The proxy fronts a paid ``SamplingClient``; exposing it beyond loopback
    without a token would let any network peer spend Tinker credits.

    Returns:
        The normalized token (stripped), or None when unset/empty.

    Raises:
        SystemExit: When ``host`` is non-loopback and no token is set.
    """
    normalized = (auth_token or "").strip() or None
    if not _is_loopback_host(host) and normalized is None:
        raise SystemExit(
            f"refusing to bind {host!r} without authentication: this proxy spends "
            "Tinker credits. Pass --auth-token (or set TINKER_PROXY_AUTH_TOKEN), "
            "or bind a loopback host such as 127.0.0.1."
        )
    return normalized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Anthropic/OpenAI-compatible capture proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7462)
    parser.add_argument("--base-model", required=True, help="Base model (renderer + tokenizer)")
    parser.add_argument(
        "--model-path", default=None, help="tinker:// weights path (defaults to the base model)"
    )
    parser.add_argument(
        "--renderer-name",
        default=None,
        help="Renderer override (default: the base model's recommended renderer)",
    )
    parser.add_argument("--default-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--auth-token",
        default=os.environ.get("TINKER_PROXY_AUTH_TOKEN"),
        help="Require this token on every request (x-api-key or Bearer). "
        "Mandatory for non-loopback --host; defaults to $TINKER_PROXY_AUTH_TOKEN.",
    )
    parser.add_argument("--store-data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--flush-interval-sec", type=float, default=1.0)
    parser.add_argument("--max-queue-size", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    auth_token = validate_bind(args.host, args.auth_token)

    store_url = ensure_daemon(args.store_data_dir)
    exporter = CaptureExporter(
        StoreSink(store_url),
        max_queue_size=args.max_queue_size,
        max_batch_size=args.max_batch_size,
        flush_interval_sec=args.flush_interval_sec,
    )
    instrument_tinker(exporter)

    renderer_name = args.renderer_name or get_recommended_renderer_name(args.base_model)
    renderer = get_renderer(
        renderer_name, get_tokenizer(args.base_model), model_name=args.base_model
    )
    service_client = tinker.ServiceClient()
    if args.model_path is not None:
        sampling_client = service_client.create_sampling_client(model_path=args.model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=args.base_model)

    deps = ProxyDeps(
        renderer=renderer,
        # cast: the real SamplingClient satisfies the protocol at runtime, but
        # pyright cannot match pydantic's cached_property ``tokens`` against
        # the protocol's read-only property.
        sampling_client=cast(SamplingClientLike, sampling_client),
        model_label=args.model_path or args.base_model,
        default_max_tokens=args.default_max_tokens,
    )
    app = make_app(deps, auth_token=auth_token)

    async def _drain(app: web.Application) -> None:
        del app
        uninstrument_tinker()
        exporter.wait_pending(timeout=_DRAIN_TIMEOUT_SEC)
        exporter.force_flush(timeout=_DRAIN_TIMEOUT_SEC)
        exporter.shutdown()

    app.on_cleanup.append(_drain)
    print(
        f"capture proxy for {deps.model_label} (renderer {renderer_name}) "
        f"on http://{args.host}:{args.port}, store {store_url}",
        flush=True,
    )
    web.run_app(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
