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


class _Stream:
    def __init__(self, messages: "_Messages", kwargs):
        self._m, self._kw = messages, kwargs
        self._final = None

    def __enter__(self):
        self._final = self._m.create(**self._kw)
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        text = self._final.content[0].text
        words = text.split(" ")
        for i, w in enumerate(words):
            yield w if i == len(words) - 1 else w + " "

    def get_final_message(self):
        return self._final


def _stream(self, **kwargs):
    return _Stream(self, kwargs)


_Messages.stream = _stream


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


def extractive_answerer(max_passages: int = 2, hallucinate: bool = False):
    """A deterministic stand-in for a model that cites correctly: for the
    passages that best overlap the question, it quotes their best-matching
    sentence followed by the passage marker. Used by the offline eval to
    exercise the citation parser and verifier end to end (it measures the
    pipeline, not a real model; use --live for that)."""
    import re as _re

    from app.core.citations import split_sentences
    from app.core.retrieval import tokenize

    def reply(kwargs):
        turn = kwargs["messages"][-1]["content"]
        q = turn.split("QUESTION:", 1)[-1].split("\n", 1)[0]
        qt = set(tokenize(q))
        blocks = _re.findall(r"\[Source (\d+)\][^\n]*\n(.*?)(?=\n\n---\n\n\[Source |\n\n=== END OF CONTEXT)",
                             turn, _re.S)
        scored = []
        for n, body in blocks:
            best, bs = None, 0
            for s in split_sentences(body):
                ov = len(qt & set(tokenize(s)))
                if ov > bs and 4 <= len(s.split()) <= 60 and not s.lstrip().startswith(("#", "|")):
                    best, bs = s, ov
            if best:
                scored.append((bs, int(n), best))
        scored.sort(key=lambda x: (-x[0], x[1]))
        parts = []
        for _, n, s in scored[:max_passages]:
            s = s.rstrip()
            if not s.endswith((".", "!", "?")):
                s += "."
            parts.append("{} [{}]".format(s, n))
        if hallucinate:
            parts.append("This is invented [{}].".format(len(blocks) + 7))
        return " ".join(parts) or "The passages do not say."

    return reply
