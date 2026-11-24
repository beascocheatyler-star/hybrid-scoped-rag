"""Persistent chats, messages and notes, plus per-notebook chat configuration.

Chats are private to the user who created them, even inside a shared
notebook: every read goes through ``get_chat(chat_id, user_id)``, which
returns None for anyone else (roadmap M4: viewers cannot read other users'
chats). Notes belong to the notebook and are visible to its members.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from app.core.db import Database, utcnow
from app.core.persona import PRESETS, Persona

MAX_HISTORY_MESSAGES = 16  # 8 turns


def _j(text: str, default):
    try:
        v = json.loads(text) if text else default
    except ValueError:
        return default
    return default if v is None and default is not None else v


@dataclass
class Message:
    id: str
    chat_id: str
    role: str
    content: str
    decision: str = ""
    reason: str = ""
    retrieval: Dict[str, Any] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    feedback: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_row(cls, r) -> "Message":
        return cls(r["id"], r["chat_id"], r["role"], r["content"], r["decision"], r["reason"],
                   _j(r["retrieval_json"], {}), _j(r["citations_json"], []), _j(r["usage_json"], {}),
                   r["model"], r["feedback"], r["created_at"])


@dataclass
class Chat:
    id: str
    notebook_id: str
    user_id: str
    title: str
    source_ids: Optional[List[str]]
    scope: Optional[Dict[str, Any]]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_row(cls, r) -> "Chat":
        return cls(r["id"], r["notebook_id"], r["user_id"], r["title"],
                   _j(r["selected_source_ids_json"], None), _j(r["scope_json"], None),
                   r["created_at"], r["updated_at"])


@dataclass
class Note:
    id: str
    notebook_id: str
    author_id: str
    title: str
    body_md: str
    citations: List[Dict[str, Any]]
    from_message_id: Optional[str]
    is_source: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_row(cls, r) -> "Note":
        return cls(r["id"], r["notebook_id"], r["author_id"], r["title"], r["body_md"],
                   _j(r["citations_json"], []), r["from_message_id"], bool(r["is_source"]),
                   r["created_at"], r["updated_at"])


class ChatRepo:
    def __init__(self, db: Database):
        self.db = db

    # --- chats -----------------------------------------------------------

    def create_chat(self, notebook_id: str, user_id: str, title: str = "New chat",
                    source_ids: Optional[List[str]] = None,
                    scope: Optional[Dict[str, Any]] = None) -> Chat:
        cid = uuid.uuid4().hex[:16]
        now = utcnow()
        self.db.execute(
            "INSERT INTO chats(id, notebook_id, user_id, title, selected_source_ids_json, scope_json, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (cid, notebook_id, user_id, (title or "New chat")[:200], json.dumps(source_ids),
             json.dumps(scope), now, now))
        return self.get_chat(cid, user_id)  # type: ignore[return-value]

    def get_chat(self, chat_id: str, user_id: str) -> Optional[Chat]:
        r = self.db.one("SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
        return Chat.from_row(r) if r else None

    def list_chats(self, notebook_id: str, user_id: str) -> List[Chat]:
        return [Chat.from_row(r) for r in self.db.all(
            "SELECT * FROM chats WHERE notebook_id = ? AND user_id = ? ORDER BY updated_at DESC",
            (notebook_id, user_id))]

    def update_chat(self, chat_id: str, user_id: str, title: Optional[str] = None,
                    source_ids: Any = ..., scope: Any = ...) -> Optional[Chat]:
        chat = self.get_chat(chat_id, user_id)
        if chat is None:
            return None
        sets, params = ["updated_at = ?"], [utcnow()]
        if title is not None:
            sets.append("title = ?"); params.append(title[:200])
        if source_ids is not ...:
            sets.append("selected_source_ids_json = ?"); params.append(json.dumps(source_ids))
        if scope is not ...:
            sets.append("scope_json = ?"); params.append(json.dumps(scope))
        self.db.execute("UPDATE chats SET {} WHERE id = ?".format(", ".join(sets)), params + [chat_id])
        return self.get_chat(chat_id, user_id)

    def delete_chat(self, chat_id: str, user_id: str) -> bool:
        return self.db.execute("DELETE FROM chats WHERE id = ? AND user_id = ?",
                               (chat_id, user_id)).rowcount > 0

    # --- messages --------------------------------------------------------

    def add_message(self, chat_id: str, role: str, content: str, **fields) -> Message:
        mid = uuid.uuid4().hex[:16]
        now = utcnow()
        self.db.execute(
            "INSERT INTO messages(id, chat_id, role, content, decision, reason, retrieval_json, "
            "citations_json, usage_json, model, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, chat_id, role, content, fields.get("decision", ""), fields.get("reason", ""),
             json.dumps(fields.get("retrieval", {})), json.dumps(fields.get("citations", [])),
             json.dumps(fields.get("usage") or {}), fields.get("model") or "", now, now))
        self.db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
        return Message.from_row(self.db.one("SELECT * FROM messages WHERE id = ?", (mid,)))

    def messages(self, chat_id: str) -> List[Message]:
        return [Message.from_row(r) for r in self.db.all(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at, rowid", (chat_id,))]

    def history(self, chat_id: str) -> List[Dict[str, str]]:
        """Prose of recent turns for the engine (no context blocks)."""
        msgs = self.messages(chat_id)[-MAX_HISTORY_MESSAGES:]
        return [{"role": m.role, "content": m.content} for m in msgs]

    def get_message(self, message_id: str, user_id: str) -> Optional[Message]:
        r = self.db.one(
            "SELECT m.* FROM messages m JOIN chats c ON c.id = m.chat_id "
            "WHERE m.id = ? AND c.user_id = ?", (message_id, user_id))
        return Message.from_row(r) if r else None

    def message_notebook(self, message_id: str) -> Optional[str]:
        r = self.db.one("SELECT c.notebook_id FROM messages m JOIN chats c ON c.id = m.chat_id "
                        "WHERE m.id = ?", (message_id,))
        return r["notebook_id"] if r else None

    def set_feedback(self, message_id: str, user_id: str, feedback: str) -> Optional[Message]:
        if self.get_message(message_id, user_id) is None:
            return None
        self.db.execute("UPDATE messages SET feedback = ?, updated_at = ? WHERE id = ?",
                        (feedback, utcnow(), message_id))
        return self.get_message(message_id, user_id)

    # --- notes -----------------------------------------------------------

    def create_note(self, notebook_id: str, author_id: str, title: str, body_md: str,
                    citations: Optional[List[Dict]] = None,
                    from_message_id: Optional[str] = None) -> Note:
        nid = uuid.uuid4().hex[:16]
        now = utcnow()
        self.db.execute(
            "INSERT INTO notes(id, notebook_id, author_id, title, body_md, citations_json, "
            "from_message_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (nid, notebook_id, author_id, (title or "Untitled note")[:200], body_md,
             json.dumps(citations or []), from_message_id, now, now))
        return self.get_note(nid, notebook_id)  # type: ignore[return-value]

    def get_note(self, note_id: str, notebook_id: str) -> Optional[Note]:
        r = self.db.one("SELECT * FROM notes WHERE id = ? AND notebook_id = ?", (note_id, notebook_id))
        return Note.from_row(r) if r else None

    def note_notebook(self, note_id: str) -> Optional[str]:
        r = self.db.one("SELECT notebook_id FROM notes WHERE id = ?", (note_id,))
        return r["notebook_id"] if r else None

    def list_notes(self, notebook_id: str) -> List[Note]:
        return [Note.from_row(r) for r in self.db.all(
            "SELECT * FROM notes WHERE notebook_id = ? ORDER BY updated_at DESC", (notebook_id,))]

    def update_note(self, note_id: str, notebook_id: str, title: Optional[str] = None,
                    body_md: Optional[str] = None) -> Optional[Note]:
        if self.get_note(note_id, notebook_id) is None:
            return None
        sets, params = ["updated_at = ?"], [utcnow()]
        if title is not None:
            sets.append("title = ?"); params.append(title[:200])
        if body_md is not None:
            sets.append("body_md = ?"); params.append(body_md)
        self.db.execute("UPDATE notes SET {} WHERE id = ?".format(", ".join(sets)), params + [note_id])
        return self.get_note(note_id, notebook_id)

    def mark_note_source(self, note_id: str) -> None:
        self.db.execute("UPDATE notes SET is_source = 1, updated_at = ? WHERE id = ?", (utcnow(), note_id))

    def delete_note(self, note_id: str, notebook_id: str) -> bool:
        return self.db.execute("DELETE FROM notes WHERE id = ? AND notebook_id = ?",
                               (note_id, notebook_id)).rowcount > 0


# --------------------------------------------------------------------------
# Chat configuration (replaces the single global persona; presets stay as
# templates)
# --------------------------------------------------------------------------

STYLES = ("default", "learning_guide", "custom")
LENGTHS = ("short", "default", "long")

_STYLE_TEXT = {
    "default": "Be clear, direct and well organised.",
    "learning_guide": (
        "Act as a learning guide. Before giving a full answer, ask one or two short "
        "guiding questions that help the user reason toward it, and check their "
        "understanding. Give the complete answer when the user asks for it directly or "
        "after they have attempted the guiding questions."
    ),
}
_LENGTH_TEXT = {
    "short": "Keep answers to a few sentences.",
    "default": "Use as much length as the question needs, and no more.",
    "long": "Give thorough, detailed answers with structure.",
}


@dataclass
class ChatConfig:
    goal: str = ""
    style: str = "default"
    custom_style: str = ""
    length: str = "default"
    language: str = ""
    name: str = "Notebook assistant"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "ChatConfig":
        d = dict(d or {})
        legacy = d.pop("persona", None)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if legacy and not cfg.goal:  # imported v1 persona
            cfg.goal = legacy.get("topic", "")
            cfg.name = legacy.get("name", cfg.name) or cfg.name
            if legacy.get("style"):
                cfg.style, cfg.custom_style = "custom", legacy["style"]
        if cfg.style not in STYLES:
            cfg.style = "default"
        if cfg.length not in LENGTHS:
            cfg.length = "default"
        return cfg

    def to_persona(self, notebook_title: str, notebook_description: str = "") -> Persona:
        topic = self.goal or notebook_description or "the sources in the notebook '{}'".format(notebook_title)
        style = self.custom_style if self.style == "custom" and self.custom_style else _STYLE_TEXT.get(
            self.style, _STYLE_TEXT["default"])
        style += " " + _LENGTH_TEXT[self.length]
        if self.language:
            style += " Always answer in {}.".format(self.language)
        return Persona(name=self.name or "Notebook assistant", topic=topic, style=style)


def preset_config(key: str) -> ChatConfig:
    p = PRESETS[key]
    return ChatConfig(goal=p.topic, style="custom", custom_style=p.style, name=p.name)
