"""Capture store daemon entry point.

Run as ``python -m tinker_cookbook.capture.store.daemon --data-dir DIR``.

Single-owner semantics: the daemon takes an exclusive ``flock`` on
``<data-dir>/daemon.lock`` before serving; a second daemon on the same data
dir exits immediately. After binding (``--port 0`` picks a free port), it
writes ``<data-dir>/daemon.json`` with ``{"port", "pid", "base_url"}`` so
clients can discover it. With ``--idle-shutdown-minutes N`` the daemon exits
after N minutes without requests.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import sys
import uuid
from pathlib import Path

from aiohttp import web

from tinker_cookbook.capture.store.app import make_app
from tinker_cookbook.capture.store.db import CaptureDB

DAEMON_INFO_FILENAME = "daemon.json"
LOCK_FILENAME = "daemon.lock"


def acquire_lock(data_dir: Path) -> int | None:
    """Take the exclusive data-dir flock. Returns the fd, or None if held."""
    fd = os.open(data_dir / LOCK_FILENAME, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


async def _serve(data_dir: Path, port: int, idle_shutdown_minutes: float | None) -> None:
    db = CaptureDB(data_dir / "capture.sqlite")
    identity = {
        "data_dir": str(data_dir.resolve()),
        "instance_token": uuid.uuid4().hex,
    }
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        loop.call_soon_threadsafe(stop_event.set)

    app = make_app(
        db,
        idle_shutdown_sec=(
            idle_shutdown_minutes * 60.0 if idle_shutdown_minutes is not None else None
        ),
        shutdown_cb=_request_stop,
        identity=identity,
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=port)
    await site.start()

    # Reaching into the private server is the only way to learn the bound
    # port when --port 0 asked the OS to pick one.
    server = site._server
    assert server is not None and server.sockets  # type: ignore[union-attr]
    bound_port = server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    info = {
        "port": bound_port,
        "pid": os.getpid(),
        "base_url": f"http://127.0.0.1:{bound_port}",
        **identity,
    }
    (data_dir / DAEMON_INFO_FILENAME).write_text(json.dumps(info))
    print(f"capture store daemon listening on {info['base_url']}", flush=True)

    try:
        await stop_event.wait()
    finally:
        await runner.cleanup()
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local capture store daemon")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    parser.add_argument("--idle-shutdown-minutes", type=float, default=None)
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = acquire_lock(data_dir)
    if lock_fd is None:
        print(f"another capture store daemon owns {data_dir}", file=sys.stderr)
        return 1
    try:
        asyncio.run(_serve(data_dir, args.port, args.idle_shutdown_minutes))
    finally:
        os.close(lock_fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
