"""Split extracted document text into overlapping, retrieval-sized chunks.

Strategy: split on the strongest available boundary first (blank lines, then
sentences, then whitespace) so a chunk rarely cuts a sentence in half. Overlap
carries context across the seam, which keeps answers coherent when the relevant
passage straddles a boundary.
"""

from __future__ import annotations

import re
from typing import List

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _hard_split(text: str, size: int) -> List[str]:
    """Last-resort split for a run of text with no usable boundary."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def _split_long_unit(unit: str, size: int) -> List[str]:
    """Break an oversized paragraph down along sentence, then word, boundaries."""
    pieces: List[str] = []
    buffer = ""

    for sentence in _SENTENCE_SPLIT.split(unit):
        if not sentence:
            continue
        candidate = (buffer + " " + sentence).strip() if buffer else sentence
        if len(candidate) <= size:
            buffer = candidate
            continue

        if buffer:
            pieces.append(buffer)
            buffer = ""

        if len(sentence) <= size:
            buffer = sentence
        else:
            # A single sentence longer than the chunk size: split on words.
            words = sentence.split()
            line = ""
            for word in words:
                trial = (line + " " + word).strip() if line else word
                if len(trial) <= size:
                    line = trial
                else:
                    if line:
                        pieces.append(line)
                    # A single word longer than the chunk size (rare).
                    line = word if len(word) <= size else ""
                    if not line:
                        pieces.extend(_hard_split(word, size))
            if line:
                buffer = line

    if buffer:
        pieces.append(buffer)
    return pieces


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Return ``text`` split into chunks of at most ``chunk_size`` characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= chunk_size:
        # Guard against the caller configuring an infinite loop.
        overlap = chunk_size // 4

    text = text.strip()
    if not text:
        return []

    # 1. Break into units that are each no larger than chunk_size.
    units: List[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(_split_long_unit(paragraph, chunk_size))

    # 2. Greedily pack units into chunks.
    chunks: List[str] = []
    current = ""
    for unit in units:
        candidate = (current + "\n\n" + unit) if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = unit
    if current:
        chunks.append(current)

    if overlap == 0 or len(chunks) < 2:
        return chunks

    # 3. Prepend the tail of the previous chunk to each subsequent chunk.
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:].lstrip()
        overlapped.append("{}\n\n{}".format(tail, chunks[i]) if tail else chunks[i])
    return overlapped
