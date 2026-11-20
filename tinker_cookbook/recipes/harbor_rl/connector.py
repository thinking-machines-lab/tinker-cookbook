# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "pydantic>=2.12", "websockets>=15.0.1"]
# ///
"""Training-side connector for Harbor's relay and Tinker's capture proxy.

Adapted from https://github.com/harbor-framework/harbor/blob/v0.26.0/examples/relay/connector.py
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal
from urllib.parse import unquote, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

logger = logging.getLogger(__name__)
HARBOR_API_VERSION = "2026-10-30"
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_REQUESTS = 8092
IDLE_TIMEOUT = 1800
HEARTBEAT_INTERVAL = 15
Headers = list[tuple[str, str]]
_STRIPPED_HEADERS = {
    "connection",
    "host",
    "content-length",
    "transfer-encoding",
    "te",
    "trailer",
    "upgrade",
    "keep-alive",
    "proxy-connection",
    "proxy-authenticate",
    "proxy-authorization",
    "authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "cookie",
    "set-cookie",
}


def forwarded_headers(headers: Headers) -> Headers:
    blocked = _STRIPPED_HEADERS | {
        item.strip().lower()
        for name, value in headers
        if name.lower() == "connection"
        for item in value.split(",")
    }
    return [
        (name.lower(), value)
        for name, value in headers
        if name.lower() not in blocked and not name.lower().startswith("x-harbor-")
    ]


Identifier = Annotated[str, Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_-]+$")]


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["request"]
    request_id: Identifier
    rollout_id: Identifier
    method: str = Field(pattern=r"^[A-Z]+$")
    target: str
    headers: Headers = Field(max_length=128)
    body_b64: str


class Cancel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["cancel"]
    request_id: Identifier
    reason: str


_MESSAGE = TypeAdapter(Annotated[InferenceRequest | Cancel, Field(discriminator="type")])


def _http_url(value: str) -> str:
    url = urlsplit(value)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Use an HTTP(S) URL without credentials, query, or fragment.")
    return value.rstrip("/")


def _requests_sse(request: httpx.Request, path: str) -> bool:
    if request.method != "POST" or path not in {"/v1/messages", "/v1/chat/completions"}:
        return False
    try:
        payload = json.loads(request.content)
    except (ValueError, RecursionError):
        return False
    return isinstance(payload, dict) and payload.get("stream") is True


@asynccontextmanager
async def _pending_response(
    client: httpx.AsyncClient, request: httpx.Request
) -> AsyncIterator[asyncio.Task[httpx.Response]]:
    """Keep one upstream request alive across heartbeat intervals and close it on exit."""
    task = asyncio.create_task(client.send(request, stream=True))
    try:
        yield task
    finally:
        task.cancel()
        try:
            response = await task
        except (asyncio.CancelledError, httpx.HTTPError):
            pass
        else:
            await response.aclose()


def _media_type(response: httpx.Response) -> str:
    return response.headers.get("content-type", "").partition(";")[0].strip().lower()


async def _proxy_error(response: httpx.Response) -> dict[str, object]:
    """Preserve a bounded proxy error, including context-window and rate-limit details."""
    fallback: dict[str, object] = {
        "type": "api_error",
        "message": f"Expected an SSE stream. Proxy returned HTTP {response.status_code}.",
    }
    if _media_type(response) != "application/json":
        return fallback
    body = bytearray()
    try:
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > 16 * 1024:
                return fallback
            body.extend(chunk)
        error = json.loads(body)["error"]
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error
    except (httpx.HTTPError, ValueError, TypeError, KeyError, RecursionError):
        pass
    return fallback


async def _committed_sse(pending: asyncio.Task[httpx.Response], path: str) -> AsyncIterator[bytes]:
    """The heartbeat already committed HTTP 200 and an uncompressed SSE body."""
    try:
        response = await pending
    except httpx.HTTPError as exc:
        error = {"type": "api_error", "message": f"Proxy forwarding failed ({type(exc).__name__})."}
    else:
        if response.status_code == 200 and _media_type(response) == "text/event-stream":
            async for chunk in response.aiter_bytes():
                yield chunk
            return
        error = await _proxy_error(response)

    if path == "/v1/messages":
        frame = "event: error\ndata: " + json.dumps({"type": "error", "error": error})
    else:
        frame = "data: " + json.dumps({"error": error})
    yield (frame + "\n\n").encode()


class Connector:
    def __init__(
        self,
        relay_url: str,
        proxy_url: str,
        *,
        session_id: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if session_id is None:
            session_id = str(uuid4())
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", session_id):
            raise ValueError("session_id must contain 1–200 letters, digits, '_' or '-'.")
        self.session_id = session_id
        url = urlsplit(_http_url(relay_url))
        self.url = urlunsplit(
            (
                "wss" if url.scheme == "https" else "ws",
                url.netloc,
                f"{url.path}/connect",
                urlencode({"session_id": session_id}),
                "",
            )
        )
        self.proxy_url = _http_url(proxy_url)
        path = urlsplit(self.proxy_url).path
        if not path.endswith("/r") and "/r/" not in path:
            self.proxy_url += "/r"
        self.api_key = api_key
        self.ready = asyncio.Event()

    async def run(self) -> None:
        """Run until cancelled; reconnect without replaying requests."""
        headers = {"Harbor-API-Version": HARBOR_API_VERSION}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(IDLE_TIMEOUT, connect=30, pool=30),
            limits=httpx.Limits(max_connections=MAX_REQUESTS),
            follow_redirects=False,
        ) as client:
            # websockets retries transient handshake failures with backoff; auth
            # failures and duplicate connections fail immediately.
            async for socket in connect(
                self.url,
                additional_headers=headers,
                max_size=MAX_REQUEST_BYTES * 2,  # Base64 plus envelope.
                open_timeout=30,
            ):
                try:
                    self.ready.set()
                    logger.debug("Relay connector connected")
                    await self._serve(socket, client)
                finally:
                    self.ready.clear()
                    await socket.close()
                if socket.close_code in {1002, 1003, 1007, 1008, 1009}:
                    raise ValueError("Relay closed the connection due to a protocol error.")
                await asyncio.sleep(1)

    async def _serve(self, socket: ClientConnection, client: httpx.AsyncClient) -> None:
        requests: dict[str, asyncio.Task[None]] = {}
        async with asyncio.TaskGroup() as group:
            try:
                async for raw in socket:
                    try:
                        if not isinstance(raw, str):
                            raise TypeError("Expected a text envelope.")
                        message = _MESSAGE.validate_json(raw)
                    except (TypeError, ValidationError):
                        await socket.close(code=1008, reason="invalid_message")
                        return
                    if isinstance(message, Cancel):
                        if task := requests.get(message.request_id):
                            task.cancel()
                        continue
                    if message.request_id in requests:
                        await socket.close(code=1008, reason="duplicate_request_id")
                        return
                    if len(requests) >= MAX_REQUESTS:
                        await self._send(
                            socket,
                            message.request_id,
                            type="error",
                            code="overloaded",
                            message="Connector concurrency limit exceeded.",
                        )
                        continue
                    task = group.create_task(self._forward(socket, client, message))
                    requests[message.request_id] = task
                    task.add_done_callback(
                        lambda _, key=message.request_id: requests.pop(key, None)
                    )
            except ConnectionClosed:
                pass
            finally:
                self.ready.clear()
                for task in requests.values():
                    task.cancel()

    @staticmethod
    async def _send(socket: ClientConnection, request_id: str, **message: Any) -> None:
        await socket.send(json.dumps({"request_id": request_id, **message}))

    async def _send_chunk(self, socket: ClientConnection, request_id: str, data: bytes) -> None:
        for offset in range(0, len(data), 64 * 1024):
            await self._send(
                socket,
                request_id,
                type="response_chunk",
                body_b64=base64.b64encode(data[offset : offset + 64 * 1024]).decode("ascii"),
            )

    async def _send_heartbeats(
        self,
        socket: ClientConnection,
        request_id: str,
        pending: asyncio.Task[httpx.Response],
    ) -> bool:
        """Return whether we opened the agent's SSE stream before the proxy sent headers."""
        started = False
        while not pending.done():
            done, _ = await asyncio.wait({pending}, timeout=HEARTBEAT_INTERVAL)
            if done:
                break
            if not started:
                await self._send(
                    socket,
                    request_id,
                    type="response_start",
                    status=200,
                    headers=[
                        ("content-type", "text/event-stream"),
                        ("cache-control", "no-cache"),
                        ("x-accel-buffering", "no"),
                    ],
                )
                started = True
            await self._send_chunk(socket, request_id, b": keep-alive\n\n")
        return started

    async def _proxy_stream(
        self,
        socket: ClientConnection,
        request_id: str,
        client: httpx.AsyncClient,
        request: httpx.Request,
        path: str,
    ) -> None:
        async with _pending_response(client, request) as pending:
            if _requests_sse(request, path) and await self._send_heartbeats(
                socket, request_id, pending
            ):
                chunks = _committed_sse(pending, path)
            else:
                response = await pending
                headers = [
                    (k.decode("latin-1"), v.decode("latin-1")) for k, v in response.headers.raw
                ]
                await self._send(
                    socket,
                    request_id,
                    type="response_start",
                    status=response.status_code,
                    headers=forwarded_headers(headers),
                )
                chunks = response.aiter_raw()
            async for data in chunks:
                await self._send_chunk(socket, request_id, data)
            await self._send(socket, request_id, type="response_end")

    async def _forward(
        self,
        socket: ClientConnection,
        client: httpx.AsyncClient,
        message: InferenceRequest,
    ) -> None:
        try:
            try:
                target = urlsplit(message.target)
                if (
                    not message.target.startswith("/")
                    or message.target.startswith("//")
                    or target.scheme
                    or target.netloc
                    or target.fragment
                    or "\\" in unquote(target.path)
                    or any(part in {".", ".."} for part in unquote(target.path).split("/"))
                ):
                    raise ValueError("Expected a relative inference target.")
                body = base64.b64decode(message.body_b64, validate=True)
                if len(body) > MAX_REQUEST_BYTES:
                    raise ValueError("Inference request body is too large.")
                headers = forwarded_headers(message.headers)
                url = (
                    f"{self.proxy_url}/rollout_id/{message.rollout_id}"
                    f"/request_id/{message.request_id}{message.target}"
                )
                # Construct directly so AsyncClient's cookie jar isn't shared
                # between rollouts. Neither Harbor nor agent credentials go here.
                request = httpx.Request(
                    message.method,
                    url,
                    headers=[(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers],
                    content=body,
                )
                await self._proxy_stream(socket, message.request_id, client, request, target.path)
            except (httpx.HTTPError, ValueError) as exc:
                await self._send(
                    socket,
                    message.request_id,
                    type="error",
                    code="upstream_timeout"
                    if isinstance(exc, httpx.TimeoutException)
                    else "upstream_error",
                    message=f"Proxy forwarding failed ({type(exc).__name__}).",
                )
        except ConnectionClosed:
            pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relay-url", required=True, help="Harbor relay HTTP(S) root URL.")
    parser.add_argument(
        "--proxy-url", default="http://127.0.0.1:7462", help="Tinker capture proxy URL."
    )
    args = parser.parse_args(argv)
    try:
        connector = Connector(
            args.relay_url,
            args.proxy_url,
            api_key=os.environ.get("HARBOR_API_KEY"),
        )
        print(json.dumps({"session_id": connector.session_id}), flush=True)
        asyncio.run(connector.run())
    except ValueError as exc:
        parser.error(str(exc))
    except InvalidStatus as exc:
        parser.error(f"Relay rejected the connection (HTTP {exc.response.status_code}).")


if __name__ == "__main__":
    main()
