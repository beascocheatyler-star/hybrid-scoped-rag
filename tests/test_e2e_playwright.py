"""Browser smoke test for the notebook UI (optional).

Skipped unless the `playwright` package is installed and a Chrome/Chromium is
available (uses the system Chrome via channel="chrome", or Playwright's own
Chromium if installed). Runs the real FastAPI app in-process with the
deterministic extractive answerer standing in for the LLM.

Flow: create notebook -> upload -> ask -> citation chip hover card -> click
chip -> source viewer opens with the cited span highlighted.
"""

from __future__ import annotations

import importlib
import socket
import threading
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

EX = Path(__file__).resolve().parent.parent / "examples"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def server(tmp_path, monkeypatch):
    import uvicorn

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app.config as config
    import app.core.embedding as embedding

    config.reset_settings()
    embedding.reset_embedder()
    import app.main as main
    from app.core.llm import AnthropicLLM, set_llm_override
    from tests.fakes import FakeLLM, extractive_answerer

    importlib.reload(main)
    set_llm_override(AnthropicLLM("claude-opus-5-5", client=FakeLLM(reply_fn=extractive_answerer()),
                                  refusal_fallback=False))
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield "http://127.0.0.1:{}".format(port)
    srv.should_exit = True
    t.join(timeout=5)
    set_llm_override(None)
    main.services.close()
    config.reset_settings()


def _browser(p):
    try:
        return p.chromium.launch(channel="chrome", headless=True)
    except Exception:
        try:
            return p.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover
            pytest.skip("no Chrome/Chromium available: {}".format(exc))


def test_upload_ask_cite_view(server, tmp_path):
    with playwright.sync_playwright() as p:
        browser = _browser(p)
        page = browser.new_page(viewport={"width": 1280, "height": 860})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append("console:" + m.text) if m.type == "error" else None)
        page.goto(server + "/")
        page.wait_for_selector("#nb-grid li")
        page.once("dialog", lambda d: d.accept("Biology 101"))
        page.click("#nb-create")
        page.wait_for_selector("#nb-title:has-text('Biology 101')")

        page.click("#src-add")
        page.set_input_files("#add-files", str(EX / "course_ta" / "cs1428_syllabus.md"))
        page.click("#add-submit")
        page.wait_for_selector("#add-dialog:not([open])", state="attached")
        page.wait_for_selector(".src-row.status-ready", timeout=20000)

        page.fill("#input", "What is the late work penalty?")
        page.press("#input", "Enter")
        try:
            chip = page.wait_for_selector(".msg.assistant button.cite", timeout=20000)
        except Exception:
            print("MESSAGES:", page.inner_text("#messages"))
            print("ERRORS:", errors)
            page.screenshot(path="/tmp/hsr-e2e-debug.png")
            raise
        chip.hover()
        card = page.wait_for_selector("#cite-card:not([hidden])")
        assert "cs1428_syllabus.md" in card.inner_text()
        chip.click()
        page.wait_for_selector("#viewer:not([hidden]) mark#hl")
        assert page.inner_text("#viewer-title") == "cs1428_syllabus.md"
        assert len(page.inner_text("mark#hl")) > 20

        page.click("summary:has-text('Why this answer?')")
        assert "Gate decision" in page.inner_text(".why")

        # Off-topic question: refused, no citations.
        page.click("#viewer-close")
        page.fill("#input", "Who won the 1998 World Cup final?")
        page.press("#input", "Enter")
        page.wait_for_function("document.querySelectorAll('.msg.assistant .msg-body.refused').length >= 1",
                               timeout=20000)
        page.screenshot(path=str(tmp_path / "notebook.png"))
        assert not errors, errors
        browser.close()


def test_mobile_layout_uses_tabs(server):
    with playwright.sync_playwright() as p:
        browser = _browser(p)
        page = browser.new_page(viewport={"width": 390, "height": 800})
        page.goto(server + "/#/nb/default")
        page.wait_for_selector(".tabs [role=tab]")
        assert page.is_visible("#panel-chat") and not page.is_visible("#panel-sources")
        page.click("#tab-sources")
        assert page.is_visible("#panel-sources") and not page.is_visible("#panel-chat")
        browser.close()
