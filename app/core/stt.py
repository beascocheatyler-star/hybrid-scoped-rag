"""Speech-to-text via an OpenAI-compatible transcription endpoint.

Used when STT_PROVIDER=openai. Works with OpenAI and with local servers that
expose the same API (for example faster-whisper-server). Stdlib only.

    STT_BASE_URL   default https://api.openai.com/v1
    STT_API_KEY    bearer token
    STT_MODEL      default whisper-1
"""

from __future__ import annotations

import json
import os
import urllib.request
import uuid
from typing import List, Tuple

from app.core.extract import ExtractionError


def openai_transcribe(data: bytes, filename: str) -> List[Tuple[int, int, str]]:
    base = os.getenv("STT_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.getenv("STT_API_KEY", "")
    model = os.getenv("STT_MODEL", "whisper-1")
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("model", model), ("response_format", "verbose_json")):
        parts.append('--{}\r\nContent-Disposition: form-data; name="{}"\r\n\r\n{}\r\n'
                     .format(boundary, name, value).encode())
    parts.append('--{}\r\nContent-Disposition: form-data; name="file"; filename="{}"\r\n'
                 'Content-Type: application/octet-stream\r\n\r\n'.format(boundary, filename).encode()
                 + data + b"\r\n")
    parts.append("--{}--\r\n".format(boundary).encode())
    req = urllib.request.Request(
        base + "/audio/transcriptions", data=b"".join(parts), method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary,
                 "Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise ExtractionError("Transcription request failed: {}".format(exc)) from exc
    segs = body.get("segments") or []
    if segs:
        return [(int(s["start"] * 1000), int(s["end"] * 1000), s.get("text", "")) for s in segs]
    return [(0, 0, body.get("text", ""))]
