"""Interactive chat with model checkpoints — persistent sessions."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tinker_cookbook.stores import RunRegistry

logger = logging.getLogger(__name__)

_client_cache: dict[str, Any] = {}


def has_api_key() -> bool:
    return bool(os.environ.get("TINKER_API_KEY"))


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    checkpoint_name: str
    session_id: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024


class ChatSession(BaseModel):
    session_id: str
    checkpoint_name: str
    run_id: str
    created_at: str
    messages: list[dict[str, str]]
    title: str  # first user message, truncated


def create_router(registry: RunRegistry) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["chat"])

    def _sessions_path(run_id: str) -> str:
        """Path prefix for chat sessions within a run's storage."""
        run = registry.get_run(run_id)
        prefix = run.prefix if run else ""
        return f"{prefix}/chat_sessions" if prefix else "chat_sessions"

    def _save_session(run_id: str, session: ChatSession) -> None:
        storage = registry.storage_for(run_id)
        path = f"{_sessions_path(run_id)}/{session.session_id}.json"
        storage.write(path, json.dumps(session.model_dump(), indent=2).encode())

    def _load_session(run_id: str, session_id: str) -> ChatSession | None:
        storage = registry.storage_for(run_id)
        path = f"{_sessions_path(run_id)}/{session_id}.json"
        try:
            data = storage.read(path)
            return ChatSession(**json.loads(data))
        except FileNotFoundError:
            return None

    def _list_sessions(run_id: str) -> list[dict[str, Any]]:
        storage = registry.storage_for(run_id)
        path = _sessions_path(run_id)
        try:
            files = storage.list_dir(path)
        except Exception:
            return []
        sessions = []
        for f in sorted(files, reverse=True):
            if not f.endswith(".json"):
                continue
            try:
                data = storage.read(f"{path}/{f}")
                s = json.loads(data)
                sessions.append({
                    "session_id": s["session_id"],
                    "checkpoint_name": s["checkpoint_name"],
                    "title": s.get("title", "Untitled"),
                    "created_at": s.get("created_at", ""),
                    "message_count": len(s.get("messages", [])),
                })
            except Exception:
                continue
        return sessions

    @router.get("/capabilities")
    async def get_capabilities() -> dict[str, bool]:
        return {"chat": has_api_key()}

    @router.get("/runs/{run_id}/chat-sessions")
    async def list_chat_sessions(run_id: str) -> list[dict[str, Any]]:
        if registry.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return _list_sessions(run_id)

    @router.get("/runs/{run_id}/chat-sessions/{session_id}")
    async def get_chat_session(run_id: str, session_id: str) -> dict[str, Any]:
        session = _load_session(run_id, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.model_dump()

    @router.post("/runs/{run_id}/chat")
    async def chat_completion(run_id: str, body: ChatRequest) -> StreamingResponse:
        if not has_api_key():
            raise HTTPException(status_code=503, detail="Interactive chat requires TINKER_API_KEY")

        run = registry.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        store = registry.get_training_store(run_id)
        config = store.read_config()
        checkpoints = store.read_checkpoints()

        ckpt = next((c for c in checkpoints if c.get("name") == body.checkpoint_name), None)
        if ckpt is None:
            raise HTTPException(status_code=404, detail=f"Checkpoint '{body.checkpoint_name}' not found")

        sampler_path = ckpt.get("sampler_path")
        if not sampler_path:
            raise HTTPException(status_code=400, detail="Checkpoint has no sampler_path")

        model_name = config.get("model_name", "") if config else ""
        if not model_name:
            raise HTTPException(status_code=400, detail="No model_name in run config")

        # Create or load session
        session_id = body.session_id or uuid.uuid4().hex[:12]
        existing = _load_session(run_id, session_id) if body.session_id else None

        async def event_stream():
            try:
                import tinker
                from tinker_cookbook.model_info import get_recommended_renderer_name
                from tinker_cookbook.renderers import get_renderer
                from tinker_cookbook.renderers.base import Message
                from tinker_cookbook.tokenizer_utils import get_tokenizer

                cache_key = f"{run_id}:{body.checkpoint_name}"
                if cache_key not in _client_cache:
                    yield f"data: {json.dumps({'status': 'Loading model and checkpoint...'})}\n\n"
                    tokenizer = get_tokenizer(model_name)
                    renderer_name = config.get("renderer_name") or get_recommended_renderer_name(model_name)
                    renderer = get_renderer(renderer_name, tokenizer, model_name=model_name)
                    service = tinker.ServiceClient()
                    tc = await service.create_lora_training_client_async(
                        base_model=model_name, rank=config.get("lora_rank", 32)
                    )
                    load_future = tc.load_state_async(sampler_path)
                    await load_future
                    sc = await tc.create_sampling_client_async(sampler_path)
                    _client_cache[cache_key] = (renderer, sc)

                renderer, sc = _client_cache[cache_key]

                messages = [Message(role=m["role"], content=m["content"]) for m in body.messages]
                model_input = renderer.build_generation_prompt(messages)
                sampling_params = tinker.SamplingParams(
                    max_tokens=body.max_tokens, temperature=body.temperature,
                )
                result = await sc.sample_async(model_input, num_samples=1, sampling_params=sampling_params)

                tokens = result.sequences[0].tokens if result.sequences else []
                response_msg, _ = renderer.parse_response(tokens)
                # Message is a TypedDict — access via dict key, not attribute
                if isinstance(response_msg, dict):
                    content = response_msg.get('content', '')
                else:
                    content = getattr(response_msg, 'content', str(response_msg))
                # Content may be string or list of parts
                if isinstance(content, str):
                    response_text = content
                elif isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict):
                            parts.append(part.get('text', part.get('thinking', '')))
                        elif hasattr(part, 'text'):
                            parts.append(part.text)
                    response_text = ''.join(parts)
                else:
                    response_text = str(content)

                # Save session with the new assistant message
                all_messages = list(body.messages) + [{"role": "assistant", "content": response_text}]
                first_user = next((m["content"] for m in all_messages if m["role"] == "user"), "Chat")
                session = ChatSession(
                    session_id=session_id,
                    checkpoint_name=body.checkpoint_name,
                    run_id=run_id,
                    created_at=existing.created_at if existing else datetime.now().isoformat(),
                    messages=all_messages,
                    title=first_user[:60],
                )
                _save_session(run_id, session)

                yield f"data: {json.dumps({'content': response_text, 'done': True, 'session_id': session_id})}\n\n"

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
