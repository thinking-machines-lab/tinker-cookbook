"""Interactive chat with model checkpoints -- persistent sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from tinker_cookbook.chef.routes._helpers import require_run
from tinker_cookbook.stores import RunRegistry
from tinker_cookbook.stores.chat_session_store import ChatSession, ChatSessionStore

logger = logging.getLogger(__name__)

_MAX_CACHED_CLIENTS = 3
_client_cache: OrderedDict[str, Any] = OrderedDict()
_client_cache_lock = asyncio.Lock()


def has_api_key() -> bool:
    return bool(os.environ.get("TINKER_API_KEY"))


def _normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Ensure a message dict has structured content (list of ContentPart dicts).

    Accepts both ``{"role": "user", "content": "hello"}`` (plain string)
    and ``{"role": "user", "content": [{"type": "text", "text": "hello"}]}``
    (already structured). Normalizes the former to the latter.
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        return {**msg, "content": [{"type": "text", "text": content}]}
    return msg


def _extract_text(msg: dict[str, Any]) -> str:
    """Extract plain text from a message's content (structured or string)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(
        p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
    )


_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]
    checkpoint_name: str
    session_id: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str | None) -> str | None:
        if v is not None and not _SESSION_ID_RE.match(v):
            raise ValueError("session_id must be alphanumeric, hyphens, or underscores")
        return v


def _get_session_store(registry: RunRegistry, run_id: str) -> ChatSessionStore:
    """Create a ChatSessionStore for a run."""
    storage = registry.storage_for(run_id)
    run = registry.get_run(run_id)
    prefix = run.prefix if run else ""
    return ChatSessionStore(storage, prefix)


def create_router(resolve_registry: Callable[..., RunRegistry]) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["chat"])

    @router.get("/capabilities")
    def get_capabilities() -> dict[str, bool]:
        return {"chat": has_api_key()}

    @router.get("/runs/{run_id}/chat-sessions")
    def list_chat_sessions(
        run_id: str, source: list[str] = Query(default=[])
    ) -> list[dict[str, Any]]:
        registry = resolve_registry(source)
        require_run(registry, run_id)
        return _get_session_store(registry, run_id).list_summaries()

    @router.get("/runs/{run_id}/chat-sessions/{session_id}")
    def get_chat_session(
        run_id: str, session_id: str, source: list[str] = Query(default=[])
    ) -> dict[str, Any]:
        registry = resolve_registry(source)
        session = _get_session_store(registry, run_id).load(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.model_dump()

    @router.post("/runs/{run_id}/chat")
    async def chat_completion(
        run_id: str, body: ChatRequest, source: list[str] = Query(default=[])
    ) -> StreamingResponse:
        if not has_api_key():
            raise HTTPException(status_code=503, detail="Interactive chat requires TINKER_API_KEY")

        registry = resolve_registry(source)
        require_run(registry, run_id)

        store = registry.get_training_store(run_id)
        config = store.read_config()
        checkpoints = store.read_checkpoints()

        ckpt = next((c for c in checkpoints if c.get("name") == body.checkpoint_name), None)
        if ckpt is None:
            raise HTTPException(
                status_code=404, detail=f"Checkpoint '{body.checkpoint_name}' not found"
            )

        sampler_path = ckpt.get("sampler_path")
        if not sampler_path:
            raise HTTPException(status_code=400, detail="Checkpoint has no sampler_path")

        model_name = config.get("model_name", "") if config else ""
        if not model_name:
            raise HTTPException(status_code=400, detail="No model_name in run config")

        session_id = body.session_id or uuid.uuid4().hex[:12]
        session_store = _get_session_store(registry, run_id)
        existing = session_store.load(session_id) if body.session_id else None

        async def event_stream():
            try:
                import tinker

                from tinker_cookbook.model_info import get_recommended_renderer_name
                from tinker_cookbook.renderers import get_renderer
                from tinker_cookbook.renderers.base import (
                    Message,
                    get_text_content,
                    message_to_jsonable,
                )
                from tinker_cookbook.tokenizer_utils import get_tokenizer

                cache_key = f"{run_id}:{body.checkpoint_name}"
                async with _client_cache_lock:
                    if cache_key in _client_cache:
                        _client_cache.move_to_end(cache_key)
                    need_load = cache_key not in _client_cache

                if need_load:
                    yield f"data: {json.dumps({'status': 'Loading model and checkpoint...'})}\n\n"
                    tokenizer = get_tokenizer(model_name)
                    renderer_name = config.get("renderer_name") or get_recommended_renderer_name(
                        model_name
                    )
                    renderer = get_renderer(renderer_name, tokenizer, model_name=model_name)
                    service = tinker.ServiceClient()
                    tc = await service.create_lora_training_client_async(
                        base_model=model_name, rank=config.get("lora_rank", 32)
                    )
                    load_future = tc.load_state_async(sampler_path)
                    await load_future
                    sc = await tc.create_sampling_client_async(sampler_path)
                    async with _client_cache_lock:
                        _client_cache[cache_key] = (renderer, sc)
                        while len(_client_cache) > _MAX_CACHED_CLIENTS:
                            _client_cache.popitem(last=False)

                async with _client_cache_lock:
                    renderer, sc = _client_cache[cache_key]

                messages = [Message(role=m["role"], content=m["content"]) for m in body.messages]
                model_input = renderer.build_generation_prompt(messages)
                sampling_params = tinker.SamplingParams(
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                )
                result = await sc.sample_async(
                    model_input, num_samples=1, sampling_params=sampling_params
                )

                tokens = result.sequences[0].tokens if result.sequences else []
                response_msg, _ = renderer.parse_response(tokens)
                response_jsonable = message_to_jsonable(response_msg)
                response_text = get_text_content(response_msg)

                # Normalize all messages to structured format before saving
                all_messages = [_normalize_message(m) for m in body.messages] + [response_jsonable]
                first_user = next((m for m in all_messages if m.get("role") == "user"), None)
                first_user_content = _extract_text(first_user) if first_user else ""
                session = ChatSession(
                    session_id=session_id,
                    checkpoint_name=body.checkpoint_name,
                    run_id=run_id,
                    created_at=existing.created_at if existing else datetime.now().isoformat(),
                    messages=all_messages,
                    title=(first_user_content or "Chat")[:60],
                )
                session_store.save(session)

                yield f"data: {json.dumps({'content': response_text, 'done': True, 'session_id': session_id, 'message': response_jsonable})}\n\n"

            except ImportError as e:
                yield f"data: {json.dumps({'error': f'Missing dependency: {e}'})}\n\n"
            except Exception as e:
                logger.exception("Chat completion error")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return router
