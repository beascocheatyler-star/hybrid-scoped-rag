"""Shared fixtures. Every test runs against a throwaway data directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.core.store import DocumentStore

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.data_dir = tmp_path
    s.ensure_dirs()
    return s


@pytest.fixture
def empty_store(tmp_path) -> DocumentStore:
    return DocumentStore(data_dir=tmp_path, chunk_size=1200, overlap=200)


@pytest.fixture
def syllabus_text() -> str:
    return (EXAMPLES / "course_ta" / "cs1428_syllabus.md").read_text("utf-8")


@pytest.fixture
def support_text() -> str:
    return (EXAMPLES / "customer_care" / "returns_and_shipping.md").read_text("utf-8")


@pytest.fixture
def syllabus_store(tmp_path, syllabus_text) -> DocumentStore:
    """A store loaded with the CS 1428 syllabus."""
    store = DocumentStore(data_dir=tmp_path, chunk_size=1200, overlap=200)
    store.add("cs1428_syllabus.md", syllabus_text.encode("utf-8"))
    return store


@pytest.fixture
def support_store(tmp_path, support_text) -> DocumentStore:
    """A store loaded with the Northwind support policy."""
    store = DocumentStore(data_dir=tmp_path, chunk_size=1200, overlap=200)
    store.add("returns_and_shipping.md", support_text.encode("utf-8"))
    return store
