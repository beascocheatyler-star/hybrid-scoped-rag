// Notebook UI. Vanilla ES modules, no build step. Every string that reaches
// innerHTML goes through an escape-first renderer in ./markdown.js (pinned by
// tests/notebook_harness.js); everything else uses textContent.

import { api, streamSSE, waitForJob, ApiError } from "./api.js";
import {
  escapeHtml, renderMarkdown, renderAnswer, renderCitationCard, renderNote,
  renderHighlighted, formatMs, locationLabel,
} from "./markdown.js";

const $ = (id) => document.getElementById(id);
const state = {
  nb: null, sources: [], selected: new Set(), scope: new Map(),
  chats: [], chatId: null, messages: [], abort: null, citeIndex: new Map(),
};

// ---------------------------------------------------------------- helpers

function el(tag, attrs = {}, text) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  if (text !== undefined) n.textContent = text;
  return n;
}

let toastTimer = null;
function toast(msg, kind = "info") {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast " + kind;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 4500);
}

function fail(e) {
  console.error(e);
  toast(e instanceof ApiError ? e.message : "Something went wrong: " + e, "error");
}

function parseTags(raw) {
  const out = {};
  (raw || "").split(",").forEach((pair) => {
    const i = pair.indexOf("=");
    if (i > 0) {
      const k = pair.slice(0, i).trim(), v = pair.slice(i + 1).trim();
      if (k && v) out[k] = v;
    }
  });
  return out;
}

// ---------------------------------------------------------------- routing

function route() {
  const m = location.hash.match(/^#\/nb\/([A-Za-z0-9_-]+)/);
  $("home").hidden = !!m;
  $("notebook").hidden = !m;
  if (m) openNotebook(m[1]).catch(fail);
  else showHome().catch(fail);
}
window.addEventListener("hashchange", route);

// ---------------------------------------------------------------- home

async function showHome() {
  state.nb = null;
  $("nb-title").textContent = "";
  document.title = "Hybrid Scoped RAG — Notebooks";
  const grid = $("nb-grid");
  grid.replaceChildren();
  const nbs = await api.get("/notebooks");
  for (const nb of nbs) {
    const li = el("li", { class: "nb-card" });
    const link = el("a", { href: "#/nb/" + nb.id, class: "nb-link" });
    link.append(el("span", { class: "nb-emoji", "aria-hidden": "true" }, nb.emoji || "📓"),
      el("span", { class: "nb-name" }, nb.title),
      el("span", { class: "nb-meta" }, nb.source_count + (nb.source_count === 1 ? " source" : " sources")));
    const actions = el("div", { class: "nb-actions" });
    actions.append(
      el("button", { class: "btn ghost", type: "button", "aria-label": "Rename " + nb.title,
        onclick: async () => {
          const t = prompt("Rename notebook", nb.title);
          if (t && t.trim()) { await api.patch("/notebooks/" + nb.id, { title: t.trim() }).catch(fail); showHome(); }
        } }, "Rename"),
      el("button", { class: "btn ghost", type: "button", "aria-label": "Copy " + nb.title,
        onclick: async () => { await api.post("/notebooks/" + nb.id + "/copy", {}).catch(fail); showHome(); } }, "Copy"),
    );
    if (!nb.is_default) {
      actions.append(el("button", { class: "btn ghost danger", type: "button", "aria-label": "Delete " + nb.title,
        onclick: async () => {
          if (confirm("Delete “" + nb.title + "” and all its sources?")) {
            await api.del("/notebooks/" + nb.id).catch(fail); showHome();
          }
        } }, "Delete"));
    }
    li.append(link, actions);
    grid.append(li);
  }
}

$("nb-create").addEventListener("click", async () => {
  const title = prompt("Notebook title", "Untitled notebook");
  if (title === null) return;
  try {
    const nb = await api.post("/notebooks", { title: title.trim() || "Untitled notebook" });
    location.hash = "#/nb/" + nb.id;
  } catch (e) { fail(e); }
});

// ---------------------------------------------------------------- notebook

async function openNotebook(id) {
  const nb = await api.get("/notebooks/" + id);
  state.nb = nb;
  state.scope.clear();
  state.selected.clear();
  $("nb-title").textContent = (nb.emoji ? nb.emoji + " " : "") + nb.title;
  document.title = nb.title + " — Hybrid Scoped RAG";
  await Promise.all([loadSources(true), loadChats(), loadNotes(), loadOverview(), loadJobs(), loadArtifacts()]);
  selectTab(matchMedia("(max-width: 900px)").matches ? "chat" : null);
}

// ---- sources

async function loadSources(selectAll) {
  const nb = state.nb;
  state.sources = await api.get("/notebooks/" + nb.id + "/sources");
  if (selectAll) state.sources.forEach((s) => state.selected.add(s.id));
  for (const id of [...state.selected]) if (!state.sources.find((s) => s.id === id)) state.selected.delete(id);
  renderSources();
  renderFacets(await api.get("/notebooks/" + nb.id + "/facets"));
  updateScopeLine();
  if (state.sources.some((s) => s.status === "queued" || s.status === "processing")) {
    setTimeout(() => { if (state.nb === nb) loadSources(false).catch(fail); }, 1500);
  }
}

function renderSources() {
  const list = $("src-list");
  list.replaceChildren();
  if (!state.sources.length) {
    list.append(el("li", { class: "empty" }, "No sources yet. Add files, a web page, a YouTube video or pasted text."));
  }
  const groups = new Map();
  for (const s of state.sources) {
    const label = (s.metadata && s.metadata.label) || "";
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(s);
  }
  const labels = [...groups.keys()].sort((a, b) => (a === "") - (b === "") || a.localeCompare(b));
  for (const label of labels) {
    if (labels.length > 1 || label) list.append(el("li", { class: "src-group" }, label || "Unlabelled"));
    for (const s of groups.get(label)) list.append(sourceRow(s));
  }
  $("src-all").checked = state.sources.length > 0 && state.selected.size === state.sources.length;
}

function sourceRow(s) {
  const li = el("li", { class: "src-row status-" + s.status });
  const cb = el("input", { type: "checkbox", id: "src-" + s.id, "aria-label": "Use " + s.title + " in chat" });
  cb.checked = state.selected.has(s.id);
  cb.disabled = s.status !== "ready";
  cb.addEventListener("change", () => {
    cb.checked ? state.selected.add(s.id) : state.selected.delete(s.id);
    $("src-all").checked = state.selected.size === state.sources.length;
    updateScopeLine();
  });
  const title = el("button", { type: "button", class: "src-title linklike",
    "aria-label": "Open " + s.title, onclick: () => openViewer(s.id) }, s.title);
  title.disabled = s.status !== "ready";
  const kind = el("span", { class: "badge kind" }, s.kind);
  const st = s.status === "ready" ? null : el("span", { class: "badge " + s.status }, s.status);
  const del = el("button", { type: "button", class: "btn ghost icon", "aria-label": "Remove " + s.title,
    onclick: async () => {
      if (!confirm("Remove “" + s.title + "” from this notebook?")) return;
      await api.del("/notebooks/" + state.nb.id + "/sources/" + s.id).catch(fail);
      loadSources(false);
    } }, "✕");
  li.append(cb, title, kind);
  if (st) li.append(st);
  li.append(del);
  if (s.status === "failed" && s.error) li.append(el("div", { class: "src-error" }, s.error));
  return li;
}

$("src-all").addEventListener("change", (e) => {
  state.selected.clear();
  if (e.target.checked) state.sources.filter((s) => s.status === "ready").forEach((s) => state.selected.add(s.id));
  renderSources();
  updateScopeLine();
});

function renderFacets(facets) {
  const box = $("facets");
  box.replaceChildren();
  for (const [key, values] of Object.entries(facets || {})) {
    const group = el("div", { class: "facet-group", role: "group", "aria-label": "Filter by " + key });
    group.append(el("span", { class: "facet-key" }, key));
    for (const [value, count] of Object.entries(values)) {
      const on = state.scope.has(key) && state.scope.get(key).has(value);
      group.append(el("button", { type: "button", class: "chip" + (on ? " on" : ""), "aria-pressed": on ? "true" : "false",
        onclick: () => {
          if (!state.scope.has(key)) state.scope.set(key, new Set());
          const set = state.scope.get(key);
          set.has(value) ? set.delete(value) : set.add(value);
          if (!set.size) state.scope.delete(key);
          renderFacets(facets);
          updateScopeLine();
        } }, value + " (" + count + ")"));
    }
    box.append(group);
  }
}

function selection() {
  const ready = state.sources.filter((s) => s.status === "ready");
  const ids = ready.length && state.selected.size === ready.length ? null : [...state.selected];
  const scope = {};
  for (const [k, set] of state.scope) scope[k] = [...set];
  return { source_ids: ids, scope: Object.keys(scope).length ? scope : null };
}

function updateScopeLine() {
  const sel = selection();
  const n = sel.source_ids ? sel.source_ids.length : state.sources.filter((s) => s.status === "ready").length;
  const parts = [n + (n === 1 ? " source" : " sources") + " selected"];
  if (sel.scope) parts.push("filter: " + Object.entries(sel.scope).map(([k, v]) => k + "=" + v.join("|")).join(", "));
  $("scope-line").textContent = parts.join(" · ");
}

// ---- add-sources dialog

let addKind = "file";
$("src-add").addEventListener("click", () => { $("add-status").textContent = ""; $("add-dialog").showModal(); });
$("add-cancel").addEventListener("click", () => $("add-dialog").close());
document.querySelectorAll(".add-tabs [role=tab]").forEach((b) => b.addEventListener("click", () => {
  addKind = b.dataset.kind;
  document.querySelectorAll(".add-tabs [role=tab]").forEach((x) => x.setAttribute("aria-selected", x === b ? "true" : "false"));
  document.querySelectorAll("[data-pane]").forEach((p) => { p.hidden = p.dataset.pane !== addKind; });
}));
$("add-submit").addEventListener("click", async () => {
  const nb = state.nb, status = $("add-status");
  const tags = parseTags($("add-tags").value);
  status.textContent = "Adding…";
  try {
    let res;
    if (addKind === "file") {
      const files = $("add-files").files;
      if (!files.length) { status.textContent = "Choose at least one file."; return; }
      const form = new FormData();
      for (const f of files) form.append("files", f);
      if (Object.keys(tags).length) form.append("metadata", JSON.stringify(tags));
      res = await api.upload("/notebooks/" + nb.id + "/sources", form);
    } else if (addKind === "text") {
      res = await api.post("/notebooks/" + nb.id + "/sources",
        { kind: "text", title: $("add-text-title").value || "Pasted text", text: $("add-text").value, metadata: tags });
    } else {
      const url = (addKind === "url" ? $("add-url") : $("add-yt")).value;
      res = await api.post("/notebooks/" + nb.id + "/sources", { kind: addKind, url, metadata: tags });
    }
    const skipped = (res.skipped || []).map((s) => s.name + ": " + s.reason);
    status.textContent = res.queued.length + " added" + (skipped.length ? "; skipped " + skipped.join("; ") : "") + ".";
    res.queued.forEach((q) => state.selected.add(q.source_id));
    $("add-form").reset();
    if (skipped.length) setTimeout(() => $("add-dialog").close(), 3000);
    else { $("add-dialog").close(); toast(res.queued.length + " source(s) added"); }
    loadSources(false);
    loadJobs();
  } catch (e) { status.textContent = e.message; }
});

// ---- source viewer

async function openViewer(sourceId, chunkId) {
  const nb = state.nb;
  try {
    const q = chunkId ? "?highlight=" + encodeURIComponent(chunkId) : "";
    const [content, src] = await Promise.all([
      api.get("/notebooks/" + nb.id + "/sources/" + sourceId + "/content" + q),
      api.get("/notebooks/" + nb.id + "/sources/" + sourceId),
    ]);
    $("viewer-title").textContent = content.title;
    const meta = $("viewer-meta");
    meta.replaceChildren(el("span", { class: "badge kind" }, content.kind));
    const h = content.highlight;
    if (h) {
      const loc = locationLabel(h);
      if (loc) meta.append(el("span", { class: "viewer-loc" }, loc.replace(/&[a-z#0-9]+;/g, "")));
      if (content.kind === "youtube" && h.t_start_ms != null && content.uri) {
        const u = new URL(content.uri);
        u.searchParams.set("t", Math.floor(h.t_start_ms / 1000) + "s");
        meta.append(el("a", { href: u.toString(), target: "_blank", rel: "noopener noreferrer" },
          "Play at " + formatMs(h.t_start_ms)));
      }
    }
    if (src.has_original) {
      meta.append(el("a", { href: "/api/v2/notebooks/" + nb.id + "/sources/" + sourceId + "/file",
        target: "_blank", rel: "noopener" }, "Original file"));
    }
    const g = src.guide || {};
    const guideBox = $("viewer-guide");
    guideBox.hidden = !g.summary;
    if (g.summary) {
      guideBox.querySelector("div").innerHTML =
        renderMarkdown(g.summary.replace(/\s*\[\d{1,3}\]/g, "")) +
        (g.topics && g.topics.length ? "<p><strong>Topics:</strong> " + g.topics.map(escapeHtml).join(" · ") + "</p>" : "");
      const qs = el("div", { class: "suggest" });
      (g.questions || []).forEach((qq) => qs.append(el("button", { type: "button", class: "chip",
        onclick: () => { closeViewer(); ask(qq); } }, qq)));
      guideBox.querySelector("div").append(qs);
    }
    const pre = $("viewer-text");
    pre.innerHTML = renderHighlighted(content.text, h ? h.start_char : null, h ? h.end_char : null);
    $("viewer").hidden = false;
    const mark = $("hl");
    if (mark) { mark.scrollIntoView({ block: "center" }); mark.focus({ preventScroll: true }); }
    else { pre.scrollTop = 0; $("viewer-close").focus(); }
  } catch (e) { fail(e); }
}
function closeViewer() { $("viewer").hidden = true; }
$("viewer-close").addEventListener("click", closeViewer);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeViewer(); hideCard(); }
});

// ---------------------------------------------------------------- chat

async function loadChats() {
  const nb = state.nb;
  state.chats = await api.get("/notebooks/" + nb.id + "/chats");
  if (!state.chats.length) state.chats = [await api.post("/notebooks/" + nb.id + "/chats", {})];
  if (!state.chats.find((c) => c.id === state.chatId)) state.chatId = state.chats[0].id;
  renderChatSelect();
  await loadMessages();
}

function renderChatSelect() {
  const sel = $("chat-select");
  sel.replaceChildren(...state.chats.map((c) => {
    const o = el("option", { value: c.id }, c.title || "New chat");
    if (c.id === state.chatId) o.selected = true;
    return o;
  }));
}
$("chat-select").addEventListener("change", (e) => { state.chatId = e.target.value; loadMessages().catch(fail); });
$("chat-new").addEventListener("click", async () => {
  try {
    const c = await api.post("/notebooks/" + state.nb.id + "/chats", {});
    state.chats.unshift(c); state.chatId = c.id; renderChatSelect(); await loadMessages(); $("input").focus();
  } catch (e) { fail(e); }
});
$("chat-delete").addEventListener("click", async () => {
  if (!confirm("Delete this chat?")) return;
  await api.del("/chats/" + state.chatId).catch(fail);
  state.chatId = null;
  loadChats().catch(fail);
});

async function loadMessages() {
  const chat = await api.get("/chats/" + state.chatId);
  state.messages = chat.messages;
  const box = $("messages");
  box.replaceChildren();
  chat.messages.forEach((m) => box.append(messageNode(m)));
  box.scrollTop = box.scrollHeight;
}

function messageNode(m) {
  const wrap = el("article", { class: "msg " + m.role, "aria-label": m.role === "user" ? "You" : "Assistant" });
  const body = el("div", { class: "msg-body" });
  if (m.role === "user") body.textContent = m.content;
  else fillAnswer(body, m);
  wrap.append(body);
  if (m.role === "assistant" && m.id) wrap.append(answerActions(m), whyPanel(m));
  return wrap;
}

function fillAnswer(body, m) {
  const r = m.retrieval || {};
  body.innerHTML = renderAnswer(m.content, m.citations || [], r.sentences || []);
  const refused = m.decision && !["in_scope", "borderline", "small_talk", "meta"].includes(m.decision);
  body.classList.toggle("refused", !!refused);
  state.citeIndex.set(body, m.citations || []);
}

function answerActions(m) {
  const bar = el("div", { class: "msg-actions", role: "group", "aria-label": "Answer actions" });
  const fb = (v, label, glyph) => el("button", { type: "button", class: "btn ghost icon" + (m.feedback === v ? " on" : ""),
    "aria-label": label, "aria-pressed": m.feedback === v ? "true" : "false",
    onclick: async (e) => {
      const next = m.feedback === v ? "" : v;
      try {
        const res = await api.post("/messages/" + m.id + "/feedback", { feedback: next });
        m.feedback = res.feedback;
        bar.querySelectorAll("[aria-pressed]").forEach((b) => { b.classList.remove("on"); b.setAttribute("aria-pressed", "false"); });
        if (next) { e.currentTarget.classList.add("on"); e.currentTarget.setAttribute("aria-pressed", "true"); }
      } catch (err) { fail(err); }
    } }, glyph);
  bar.append(
    el("button", { type: "button", class: "btn ghost", onclick: () => {
      navigator.clipboard.writeText(m.content).then(() => toast("Copied"), fail);
    } }, "Copy"),
    el("button", { type: "button", class: "btn ghost", onclick: async () => {
      try { await api.post("/messages/" + m.id + "/save-to-note"); toast("Saved to notes"); loadNotes(); } catch (e) { fail(e); }
    } }, "Save to note"),
    el("button", { type: "button", class: "btn ghost", "aria-label": "Regenerate this answer", onclick: () => {
      const i = state.messages.findIndex((x) => x.id === m.id);
      const prev = state.messages.slice(0, i).reverse().find((x) => x.role === "user");
      if (prev) ask(prev.content);
    } }, "Regenerate"),
    fb("up", "Good answer", "👍"),
    fb("down", "Bad answer", "👎"),
  );
  return bar;
}

function whyPanel(m) {
  const r = m.retrieval || {};
  const d = el("details", { class: "why" });
  d.append(el("summary", {}, "Why this answer?"));
  const dl = el("dl", { class: "why-grid" });
  const row = (k, v) => { dl.append(el("dt", {}, k), el("dd", {}, String(v))); };
  row("Gate decision", m.decision || "—");
  if (m.reason) row("Reason", m.reason);
  if (r.relevance !== undefined) row("Lexical relevance", r.relevance);
  if (r.dense_score !== undefined) row("Best dense score", r.dense_score);
  if (r.rescued_by_dense) row("Rescued by dense", "yes (model told to verify scope)");
  if (r.rewrite && r.rewrite !== "none") row("Follow-up rewritten (" + r.rewrite + ")", r.query);
  if (r.documents_in_scope !== undefined) row("Sources in scope", r.documents_in_scope);
  d.append(dl);
  const passages = r.passages || [];
  if (passages.length) {
    const t = el("table", { class: "why-table" });
    const head = el("tr");
    ["#", "Source", "Retrievers", "Lexical rank", "Dense rank"].forEach((h) => head.append(el("th", { scope: "col" }, h)));
    t.append(el("thead"), el("tbody"));
    t.tHead.append(head);
    passages.forEach((p) => {
      const tr = el("tr");
      [p.n, p.source_title, (p.retrievers || []).join(" + "), p.lexical_rank ?? "—", p.dense_rank ?? "—"]
        .forEach((v) => tr.append(el("td", {}, String(v))));
      t.tBodies[0].append(tr);
    });
    d.append(t);
  }
  return d;
}

// ---- citation chips: hover card + click to open source

function citationFor(btn) {
  const body = btn.closest(".msg-body, .note-body");
  const list = state.citeIndex.get(body) || [];
  return list.find((c) => c.marker === Number(btn.dataset.marker));
}
function showCard(btn) {
  const c = citationFor(btn);
  if (!c) return;
  const card = $("cite-card");
  card.innerHTML = renderCitationCard(c);
  card.hidden = false;
  const r = btn.getBoundingClientRect();
  const top = Math.min(window.innerHeight - card.offsetHeight - 8, r.bottom + 6);
  const left = Math.min(window.innerWidth - card.offsetWidth - 8, Math.max(8, r.left - 20));
  card.style.top = Math.max(8, top) + "px";
  card.style.left = left + "px";
  btn.setAttribute("aria-describedby", "cite-card");
}
function hideCard() { $("cite-card").hidden = true; }
document.addEventListener("mouseover", (e) => { const b = e.target.closest("button.cite"); if (b) showCard(b); });
document.addEventListener("mouseout", (e) => { if (e.target.closest("button.cite")) hideCard(); });
document.addEventListener("focusin", (e) => { const b = e.target.closest("button.cite"); b ? showCard(b) : hideCard(); });
document.addEventListener("click", (e) => {
  const b = e.target.closest("button.cite");
  if (!b) return;
  const c = citationFor(b);
  if (c) { hideCard(); openViewer(c.source_id, c.chunk_id); }
});

// ---- asking (streaming)

function setBusy(busy) {
  $("send").hidden = busy;
  $("stop").hidden = !busy;
  $("input").disabled = busy;
}
$("stop").addEventListener("click", () => { if (state.abort) state.abort.abort(); });

async function ask(text) {
  text = (text || "").trim();
  if (!text || state.abort) return;
  selectTab("chat");
  const box = $("messages");
  const userNode = messageNode({ role: "user", content: text });
  const pending = el("article", { class: "msg assistant pending", "aria-label": "Assistant", "aria-busy": "true" });
  const body = el("div", { class: "msg-body" });
  const status = el("div", { class: "msg-status" }, "Checking scope…");
  pending.append(status, body);
  box.append(userNode, pending);
  box.scrollTop = box.scrollHeight;
  setBusy(true);
  state.abort = new AbortController();
  let raw = "";
  try {
    await streamSSE("/chats/" + state.chatId + "/messages/stream", { content: text, ...selection() }, (name, data) => {
      if (name === "decision") {
        status.textContent = data.allowed ? "Answering from " + data.documents_in_scope + " source(s)…" : "";
        if (!data.allowed && data.refusal) body.textContent = data.refusal;
      } else if (name === "rewrite") {
        status.textContent = "Follow-up understood as: " + data.query;
      } else if (name === "token") {
        raw += data.text;
        body.textContent = raw;  // plain text while streaming; rendered on done
        box.scrollTop = box.scrollHeight;
      } else if (name === "done") {
        const m = data.message;
        state.messages.push({ role: "user", content: text }, m);
        pending.replaceWith(messageNode(m));
      } else if (name === "error") {
        status.textContent = "";
        body.textContent = data.detail;
        pending.classList.add("error");
      }
    }, state.abort.signal);
  } catch (e) {
    if (e.name === "AbortError") { status.textContent = "Stopped."; pending.removeAttribute("aria-busy"); }
    else { fail(e); pending.remove(); }
  } finally {
    state.abort = null;
    setBusy(false);
    $("input").value = "";
    $("input").focus();
    box.scrollTop = box.scrollHeight;
    refreshChatTitle();
  }
}

async function refreshChatTitle() {
  try {
    state.chats = await api.get("/notebooks/" + state.nb.id + "/chats");
    renderChatSelect();
  } catch (_) { /* non-critical */ }
}

$("composer").addEventListener("submit", (e) => { e.preventDefault(); ask($("input").value); });
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask($("input").value); }
});
$("input").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(200, e.target.scrollHeight) + "px";
});

// ---- overview + suggested questions

async function loadOverview() {
  const box = $("overview");
  box.replaceChildren();
  const ov = await api.get("/notebooks/" + state.nb.id + "/overview").catch(() => ({}));
  if (!ov || !ov.summary) return;
  const d = el("details", { class: "overview-box", open: state.messages.length ? null : true });
  d.append(el("summary", {}, "Notebook overview" + (ov.stale ? " (updating…)" : "")));
  const p = el("div");
  p.innerHTML = renderMarkdown(ov.summary.replace(/\s*\[\d{1,3}\]/g, ""));
  d.append(p);
  const qs = el("div", { class: "suggest", role: "group", "aria-label": "Suggested questions" });
  (ov.questions || []).forEach((q) => qs.append(el("button", { type: "button", class: "chip", onclick: () => ask(q) }, q)));
  d.append(qs);
  box.append(d);
}

// ---------------------------------------------------------------- studio

async function loadJobs() {
  const ul = $("jobs");
  const jobs = await api.get("/notebooks/" + state.nb.id + "/jobs").catch(() => []);
  ul.replaceChildren(...jobs.slice(0, 8).map((j) => {
    const li = el("li", { class: "job " + j.status });
    li.append(el("span", {}, j.type), el("span", { class: "badge " + j.status }, j.status));
    if (j.status === "running" || j.status === "queued") {
      li.append(el("progress", { max: "1", value: String(j.progress || 0), "aria-label": j.type + " progress" }));
      li.append(el("button", { type: "button", class: "btn ghost icon", "aria-label": "Cancel " + j.type,
        onclick: async () => { await api.post("/jobs/" + j.id + "/cancel").catch(fail); loadJobs(); } }, "✕"));
    }
    if (j.error) li.append(el("div", { class: "src-error" }, j.error));
    return li;
  }));
  if (jobs.some((j) => j.status === "running" || j.status === "queued")) {
    const nb = state.nb;
    setTimeout(() => {
      if (state.nb === nb) { loadJobs(); loadOverview(); loadArtifacts(); }
    }, 2000);
  }
}

async function loadNotes() {
  const ul = $("notes");
  const notes = await api.get("/notebooks/" + state.nb.id + "/notes");
  ul.replaceChildren(...notes.map((n) => {
    const li = el("li", { class: "note" });
    const body = el("div", { class: "note-card" });
    body.innerHTML = renderNote(n);
    state.citeIndex.set(body.querySelector(".note-body"), n.citations || []);
    const bar = el("div", { class: "msg-actions" });
    bar.append(
      el("button", { type: "button", class: "btn ghost", onclick: async () => {
        const b = prompt("Edit note", n.body_md);
        if (b !== null) { await api.patch("/notebooks/" + state.nb.id + "/notes/" + n.id, { body_md: b }).catch(fail); loadNotes(); }
      } }, "Edit"),
      n.is_source ? el("span", { class: "badge" }, "is a source") :
        el("button", { type: "button", class: "btn ghost", onclick: async () => {
          try { await api.post("/notes/" + n.id + "/convert-to-source"); toast("Added as a source"); loadNotes(); loadSources(false); loadJobs(); }
          catch (e) { fail(e); }
        } }, "Use as source"),
      el("button", { type: "button", class: "btn ghost danger", "aria-label": "Delete note " + n.title, onclick: async () => {
        if (confirm("Delete this note?")) { await api.del("/notebooks/" + state.nb.id + "/notes/" + n.id).catch(fail); loadNotes(); }
      } }, "Delete"),
    );
    li.append(body, bar);
    return li;
  }));
  if (!notes.length) ul.append(el("li", { class: "empty" }, "Save answers or write notes here."));
}
$("note-new").addEventListener("click", async () => {
  const title = prompt("Note title", "Untitled note");
  if (title === null) return;
  const body = prompt("Note text (markdown)", "") || "";
  await api.post("/notebooks/" + state.nb.id + "/notes", { title, body_md: body }).catch(fail);
  loadNotes();
});

// Studio outputs are wired up in studio.js (M6); this keeps the panel usable
// when that module is absent.
async function loadArtifacts() {
  try {
    const mod = await import("./studio.js");
    await mod.render(state, { api, toast, fail, ask, openViewer, waitForJob, loadJobs });
  } catch (e) {
    if (!String(e).includes("studio.js")) console.error(e);
  }
}

// ---------------------------------------------------------------- tabs

function selectTab(name) {
  const tabs = ["sources", "chat", "studio"];
  document.querySelectorAll(".tabs [role=tab]").forEach((t) => {
    const on = t.id === "tab-" + name;
    t.setAttribute("aria-selected", on ? "true" : "false");
    t.tabIndex = on ? 0 : -1;
  });
  if (name) $("notebook").dataset.tab = name;
  else delete $("notebook").dataset.tab;
  return tabs;
}
document.querySelectorAll(".tabs [role=tab]").forEach((t, i, all) => {
  t.addEventListener("click", () => selectTab(t.id.slice(4)));
  t.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      const next = all[(i + (e.key === "ArrowRight" ? 1 : all.length - 1)) % all.length];
      next.focus(); selectTab(next.id.slice(4));
    }
  });
});

route();
