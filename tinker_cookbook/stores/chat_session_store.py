"""Chat session persistence — stores conversations with model checkpoints."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from tinker_cookbook.stores._base import BaseStore
from tinker_cookbook.stores.storage import Storage, storage_join

logger = logging.getLogger(__name__)


class ChatSession(BaseModel):
    """A persisted chat session with a model checkpoint."""

    session_id: str
    checkpoint_name: str
    run_id: str
    created_at: str
    messages: list[dict[str, Any]]
    title: str


class ChatSessionStore(BaseStore):
    """Reads and writes chat sessions for a run within its storage."""

    def __init__(self, storage: Storage, prefix: str = "") -> None:
        chat_prefix = storage_join(prefix, "chat_sessions") if prefix else "chat_sessions"
        super().__init__(storage, chat_prefix)

    def save(self, session: ChatSession) -> None:
        self._write_json(session.model_dump(), f"{session.session_id}.json")

    def load(self, session_id: str) -> ChatSession | None:
        data = self._read_json(f"{session_id}.json")
        return ChatSession(**data) if data is not None else None

    def list_summaries(self) -> list[dict[str, Any]]:
        """Return lightweight summaries of all sessions, newest first."""
        try:
            files = self.storage.list_dir(self._path())
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
                data = self.storage.read(self._path(f))
                s = json.loads(data)
                sessions.append(
                    {
                        "session_id": s["session_id"],
                        "checkpoint_name": s["checkpoint_name"],
                        "title": s.get("title", "Untitled"),
                        "created_at": s.get("created_at", ""),
                        "message_count": len(s.get("messages", [])),
                    }
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug("Skipping malformed session file %s: %s", f, e)
                continue
        return sessions
