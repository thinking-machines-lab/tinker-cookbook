"""Chat session persistence — stores conversations with model checkpoints."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from tinker_cookbook.stores.storage import Storage

logger = logging.getLogger(__name__)


class ChatSession(BaseModel):
    """A persisted chat session with a model checkpoint."""

    session_id: str
    checkpoint_name: str
    run_id: str
    created_at: str
    messages: list[dict[str, Any]]
    title: str


class ChatSessionStore:
    """Reads and writes chat sessions for a run within its storage."""

    def __init__(self, storage: Storage, prefix: str = "") -> None:
        self._storage = storage
        self._base = f"{prefix}/chat_sessions" if prefix else "chat_sessions"

    def save(self, session: ChatSession) -> None:
        path = f"{self._base}/{session.session_id}.json"
        self._storage.write(path, json.dumps(session.model_dump(), indent=2).encode())

    def load(self, session_id: str) -> ChatSession | None:
        path = f"{self._base}/{session_id}.json"
        try:
            data = self._storage.read(path)
            return ChatSession(**json.loads(data))
        except FileNotFoundError:
            return None

    def list_summaries(self) -> list[dict[str, Any]]:
        """Return lightweight summaries of all sessions, newest first."""
        try:
            files = self._storage.list_dir(self._base)
        except FileNotFoundError:
            return []
        except Exception:
            logger.debug("Failed to list chat sessions", exc_info=True)
            return []

        sessions: list[dict[str, Any]] = []
        for f in sorted(files, reverse=True):
            if not f.endswith(".json"):
                continue
            try:
                data = self._storage.read(f"{self._base}/{f}")
                s = json.loads(data)
                sessions.append({
                    "session_id": s["session_id"],
                    "checkpoint_name": s["checkpoint_name"],
                    "title": s.get("title", "Untitled"),
                    "created_at": s.get("created_at", ""),
                    "message_count": len(s.get("messages", [])),
                })
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug("Skipping malformed session file %s: %s", f, e)
                continue
        return sessions
