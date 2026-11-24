# Manual QA checklist — notebook UI

Run `python run.py` and open http://localhost:8000. Automated coverage is
noted per item: [h] = renderer harness (`tests/notebook_harness.js`),
[e2e] = Playwright smoke test (`tests/test_e2e_playwright.py`), [api] = API
tests.

## Home
- [ ] Notebook grid lists every notebook with emoji, title, source count. [e2e]
- [ ] "+ New notebook" creates one and opens it. [e2e]
- [ ] Rename, Copy (sources copied, chats/notes not) and Delete work; the
      Default notebook has no Delete button. [api]

## Sources
- [ ] Add dialog: Files, Website, YouTube, Paste text tabs; optional tags.
      Files land as `queued`/`processing`, then `ready`. [e2e for files]
- [ ] A failed source shows its error under the row (try a `.pdf` that is not
      a PDF). [api]
- [ ] Per-source checkbox and "Select all"; the line under the composer
      reports what is selected; questions only use selected sources. [api]
- [ ] Metadata chips filter (AND across keys, OR within a key). Label groups
      appear once auto-labels exist (5+ sources).
- [ ] Clicking a title opens the viewer drawer with the Source Guide.
- [ ] Removing a source updates the list and the overview (marked updating).

## Chat
- [ ] Answers stream token by token; Stop aborts. [e2e for streaming]
- [ ] Citation chips `[n]` appear; hover/focus shows source, location and
      quote; weak citations are amber. [h][e2e]
- [ ] Clicking a chip opens the viewer at the cited span, highlighted and
      scrolled into view; PDFs show the page, YouTube links to the timestamp.
      [e2e for highlight]
- [ ] Unsupported sentences are dotted-underlined; block answers instead show
      "N sentences could not be verified". [h]
- [ ] Off-topic question: refused immediately, no tokens, no citations.
      [e2e][api]
- [ ] Follow-up ("When is it?") after an in-scope question is answered; "Why
      this answer?" shows the rewrite. [api]
- [ ] "Why this answer?" shows the gate decision, relevance, dense score,
      rescue flag and per-retriever ranks. [e2e]
- [ ] Copy, Save to note, Regenerate, 👍/👎 work; feedback persists on reload.
- [ ] Chats persist across reloads; New chat, switch, Delete. [api]
- [ ] Overview shows a summary and suggested-question chips that ask.

## Studio and notes
- [ ] Jobs list shows ingest/guide/overview jobs with progress; queued and
      running jobs can be cancelled.
- [ ] Notes: create, edit, delete; saved answers keep clickable citation
      chips; "Use as source" turns a note into a source. [h][api]

## Accessibility and layout
- [ ] Under 900 px wide the three panels become tabs (arrow keys move between
      tabs). [e2e]
- [ ] Everything is reachable by keyboard; focus is visible; Esc closes the
      viewer; citation cards appear on focus.
- [ ] Lighthouse accessibility ≥ 90 on the notebook page. Measured
      2026-11-23: 100 desktop, 100 mobile, home page 100; best practices 100.
- [ ] Dark mode (OS setting) keeps contrast.

## Classic UI
- [ ] `/classic` still serves the original single-corpus UI on the Default
      notebook.
