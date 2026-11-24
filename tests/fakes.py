"""Deterministic fakes shared by tests and the offline eval.

``FakeLLM`` mimics the slice of the Anthropic client the engine uses
(``client.messages.create``) and records every call, so a test can assert that
a refusal path spent zero tokens.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class _Block:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Response:
    stop_reason = "end_turn"
    model = "fake-model"

    def __init__(self, text: str, usage: _Usage):
        self.content = [_Block(text)]
        self.usage = usage


class _Messages:
    def __init__(self, owner: "FakeLLM"):
        self._owner = owner

    def create(self, **kwargs: Any) -> _Response:
        o = self._owner
        o.calls.append(kwargs)
        text = o.reply_fn(kwargs) if o.reply_fn else o.reply
        prompt_chars = len(str(kwargs.get("system", ""))) + len(
            str(kwargs.get("messages", ""))
        )
        # ~4 chars/token: good enough for a tokens-per-answer metric offline.
        return _Response(text, _Usage(prompt_chars // 4, len(text) // 4))


class FakeLLM:
    """Stand-in for ``anthropic.Anthropic``. Never touches the network."""

    def __init__(self, reply: str = "A grounded answer.", reply_fn=None):
        self.reply = reply
        self.reply_fn = reply_fn
        self.calls: List[Dict[str, Any]] = []
        self.messages = _Messages(self)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last(self) -> Optional[Dict[str, Any]]:
        return self.calls[-1] if self.calls else None
