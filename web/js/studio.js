// Studio panel: generate, list and view artifacts. All innerHTML goes through
// escape-first renderers in ./markdown.js; the mind map is drawn with SVG DOM
// nodes and textContent only.

import {
  citationChips, renderMarkdown, renderDataTable, renderStudyText, mindMapLabel,
} from "./markdown.js";

const $ = (id) => document.getElementById(id);
let ctx = null;   // { api, toast, fail, ask, openViewer, waitForJob, loadJobs }
let st = null;    // app state
let bound = false;

const TYPES = [
  { type: "report", label: "Report", hint: "Briefing, study guide, FAQ, timeline…" },
  { type: "flashcards", label: "Flashcards", hint: "Flip cards with progress" },
  { type: "quiz", label: "Quiz", hint: "Graded questions" },
  { type: "mind_map", label: "Mind map", hint: "Collapsible topic tree" },
  { type: "data_table", label: "Data table", hint: "Rows × columns you describe" },
];

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

function selectedSourceIds() {
  const ready = st.sources.filter((s) => s.status === "ready");
  return ready.length && st.selected.size === ready.length ? null : [...st.selected];
}

export async function render(state, helpers) {
  st = state;
  ctx = helpers;
  renderCards();
  if (!bound) bindDialogs();
  await renderList();
}

// ---------------------------------------------------------------- generate

function renderCards() {
  const box = $("studio-cards");
  if (box.dataset.ready) return;
  box.dataset.ready = "1";
  box.replaceChildren(...TYPES.map((t) => el("button", { type: "button", class: "studio-card",
    "aria-label": "Generate " + t.label, onclick: () => openCustomise(t.type) }, t.label + " — " + t.hint)));
}

function field(label, control) {
  const id = "f-" + Math.random().toString(36).slice(2, 8);
  control.id = id;
  const wrap = el("div", { class: "field" });
  wrap.append(el("label", { for: id }, label), control);
  return wrap;
}
function select(name, opts, value) {
  const s = el("select", { name });
  opts.forEach(([v, l]) => { const o = el("option", { value: v }, l); if (v === value) o.selected = true; s.append(o); });
  return s;
}

let pendingType = null;
function openCustomise(type) {
  pendingType = type;
  const f = $("studio-fields");
  f.replaceChildren();
  $("studio-title").textContent = "Generate " + TYPES.find((t) => t.type === type).label.toLowerCase();
  $("studio-status").textContent = "";
  if (type === "report") {
    f.append(field("Format", select("format", [["briefing", "Briefing doc"], ["study_guide", "Study guide"],
      ["faq", "FAQ"], ["timeline", "Timeline"], ["blog_post", "Blog post"], ["custom", "Custom"]], "briefing")));
    f.append(field("Custom instructions (optional)", el("input", { type: "text", name: "custom_prompt" })));
    f.append(field("Length", select("length", [["short", "Short"], ["", "Default"], ["long", "Long"]], "")));
  }
  if (type === "flashcards" || type === "quiz") {
    f.append(field("Number", select("quantity", [["fewer", "Fewer"], ["standard", "Standard"], ["more", "More"]], "standard")));
    f.append(field("Difficulty", select("difficulty", [["easy", "Easy"], ["medium", "Medium"], ["hard", "Hard"]], "medium")));
  }
  if (type === "quiz") {
    const fs = el("fieldset");
    fs.append(el("legend", {}, "Question types"));
    [["multiple_choice", "Multiple choice"], ["multi_select", "Select all that apply"],
      ["fill_blank", "Fill in the blank"], ["short_answer", "Short answer"]].forEach(([v, l], i) => {
      const lab = el("label");
      const cb = el("input", { type: "checkbox", name: "types", value: v });
      cb.checked = i === 0;
      lab.append(cb, document.createTextNode(" " + l));
      fs.append(lab);
    });
    f.append(el("div", { class: "field" }), fs);
  }
  if (type === "data_table") {
    f.append(field("Columns (comma-separated)", el("input", { type: "text", name: "columns", value: "Main idea, Key numbers" })));
  }
  if (type !== "report") f.append(field("Focus (optional)", el("input", { type: "text", name: "focus" })));
  f.append(field("Output language (optional)", el("input", { type: "text", name: "language" })));
  const n = selectedSourceIds();
  f.append(el("p", { class: "status" }, "Uses " + (n ? n.length + " selected source(s)" : "all sources") + "."));
  $("studio-dialog").showModal();
}

function readParams() {
  const form = $("studio-form");
  const p = {};
  form.querySelectorAll("select, input[type=text]").forEach((x) => { if (x.name && x.value.trim()) p[x.name] = x.value.trim(); });
  const types = [...form.querySelectorAll("input[name=types]:checked")].map((x) => x.value);
  if (types.length) p.types = types;
  if (p.columns) p.columns = p.columns.split(",").map((c) => c.trim()).filter(Boolean);
  return p;
}

function bindDialogs() {
  bound = true;
  $("studio-cancel").addEventListener("click", () => $("studio-dialog").close());
  $("studio-go").addEventListener("click", async () => {
    const status = $("studio-status");
    try {
      status.textContent = "Starting…";
      const r = await ctx.api.post("/notebooks/" + st.nb.id + "/artifacts",
        { type: pendingType, params: readParams(), source_ids: selectedSourceIds() });
      $("studio-dialog").close();
      ctx.toast("Generating " + r.artifact.title.toLowerCase() + "…");
      ctx.loadJobs();
      await renderList();
      const job = await ctx.waitForJob(r.job_id);
      await renderList();
      if (job.status === "succeeded") openArtifact(r.artifact.id);
      else ctx.toast(job.error || "Generation " + job.status, "error");
    } catch (e) { status.textContent = e.message; }
  });
  $("artifact-close").addEventListener("click", () => $("artifact-dialog").close());
  $("artifact-body").addEventListener("click", (e) => {
    const b = e.target.closest("button.cite");
    if (!b || !currentCites) return;
    // Markers are numbered per artifact; an item's own list (data-cites) wins.
    const scoped = b.closest("[data-cites]");
    const list = scoped ? JSON.parse(scoped.dataset.cites) : currentCites;
    const hit = list.find((x) => x.marker === Number(b.dataset.marker));
    if (hit) { $("artifact-dialog").close(); ctx.openViewer(hit.source_id, hit.chunk_id); }
  });
}

// ---------------------------------------------------------------- list

async function renderList() {
  const ul = $("artifacts");
  const list = await ctx.api.get("/notebooks/" + st.nb.id + "/artifacts").catch(() => []);
  ul.replaceChildren(...list.map((a) => {
    const li = el("li", { class: "artifact-row" });
    const open = el("button", { type: "button", class: "linklike", onclick: () => openArtifact(a.id) },
      a.title + (a.item_count ? " (" + a.item_count + ")" : ""));
    open.disabled = a.status !== "ready";
    li.append(open);
    if (a.status !== "ready") li.append(el("span", { class: "badge " + a.status }, a.status));
    if (a.stale) li.append(el("span", { class: "badge stale", title: "The sources changed since this was generated" }, "stale"));
    li.append(el("button", { type: "button", class: "btn ghost icon", "aria-label": "Delete " + a.title, onclick: async () => {
      if (confirm("Delete this output?")) { await ctx.api.del("/artifacts/" + a.id).catch(ctx.fail); renderList(); }
    } }, "✕"));
    if (a.status === "failed" && a.error) li.append(el("div", { class: "src-error" }, a.error));
    return li;
  }));
  if (!list.length) ul.append(el("li", { class: "empty" }, "Nothing generated yet."));
}

// ---------------------------------------------------------------- viewers

let currentCites = null;

function allCites(content) {
  const out = [];
  const seen = new Set();
  const add = (cs) => (cs || []).forEach((c) => { const k = c.marker + ":" + c.chunk_id; if (!seen.has(k)) { seen.add(k); out.push(c); } });
  (content.sections || []).forEach((s) => add(s.citations));
  (content.rows || []).forEach((r) => add(r.citations));
  (content.cards || []).forEach((c) => add(c.citations));
  (content.items || []).forEach((c) => add(c.citations));
  return out;
}

async function openArtifact(id) {
  try {
    const a = await ctx.api.get("/artifacts/" + id);
    $("artifact-title").textContent = a.title + (a.stale ? " (sources changed since generation)" : "");
    const tools = $("artifact-tools");
    tools.replaceChildren();
    const formats = { report: ["md", "docx", "pdf"], data_table: ["csv", "citations_csv", "md"], mind_map: ["svg", "md", "json"],
      flashcards: ["md", "json"], quiz: ["md", "json"] }[a.type];
    formats.forEach((f) => tools.append(el("a", { class: "btn ghost", href: "/api/v2/artifacts/" + a.id + "/download?format=" + f,
      download: "" }, f.replace("_", " ").toUpperCase())));
    currentCites = allCites(a.content);
    const body = $("artifact-body");
    body.replaceChildren();
    ({ report: viewReport, data_table: viewTable, mind_map: viewMindMap, flashcards: viewCards, quiz: viewQuiz })[a.type](a, body, tools);
    if (!$("artifact-dialog").open) $("artifact-dialog").showModal();
  } catch (e) { ctx.fail(e); }
}

function viewReport(a, body) {
  a.content.sections.forEach((s) => {
    const sec = el("section", { "data-cites": JSON.stringify(s.citations) });
    sec.append(el("h3", {}, s.heading));
    const div = el("div");
    div.innerHTML = citationChips(renderMarkdown(s.body_md), s.citations);
    sec.append(div);
    body.append(sec);
  });
}

function viewTable(a, body) {
  const div = el("div");
  div.innerHTML = renderDataTable(a.content);
  body.append(div);
}

// ---- flashcards

function viewCards(a, body, tools) {
  const cards = a.content.cards;
  let progress = a.progress || {};
  let deck = cards.slice();
  let i = 0, flipped = false;
  const face = el("div", { class: "flashcard", role: "button", tabindex: "0", "aria-live": "polite" });
  const counter = el("span", { class: "status" });
  const bar = el("div", { class: "study-bar" });
  const show = () => {
    if (!deck.length) { face.textContent = "Done! Retake all or just the missed cards."; counter.textContent = ""; return; }
    const c = deck[i];
    face.classList.toggle("back", flipped);
    face.innerHTML = renderStudyText(flipped ? c.back : c.front, flipped ? c.citations : []);
    face.dataset.cites = JSON.stringify(c.citations);
    const got = cards.filter((x) => (progress[x.id] || {}).state === "got_it").length;
    const missed = cards.filter((x) => (progress[x.id] || {}).state === "missed").length;
    counter.textContent = "Card " + (i + 1) + " of " + deck.length + " · got " + got + " · missed " + missed;
  };
  const mark = async (state) => {
    const c = deck[i];
    const r = await ctx.api.patch("/artifacts/" + a.id + "/progress", { item_id: c.id, state }).catch(ctx.fail);
    if (r) progress = r.progress;
    i += 1; flipped = false;
    if (i >= deck.length) { deck = []; i = 0; }
    show();
  };
  const flip = () => { flipped = !flipped; show(); };
  face.addEventListener("click", (e) => { if (!e.target.closest("button.cite")) flip(); });
  face.addEventListener("keydown", (e) => { if (e.key === " " || e.key === "Enter") { e.preventDefault(); flip(); } });
  bar.append(
    el("button", { type: "button", class: "btn", onclick: flip }, "Flip"),
    el("button", { type: "button", class: "btn", onclick: () => mark("missed") }, "Missed it"),
    el("button", { type: "button", class: "btn primary", onclick: () => mark("got_it") }, "Got it"),
    el("button", { type: "button", class: "btn ghost", onclick: () => {
      if (!deck.length) return;
      $("artifact-dialog").close();
      ctx.ask("Explain this: " + deck[i].front.replace(/_{3,}/g, "…"));
    } }, "Explain"),
  );
  tools.append(
    el("button", { type: "button", class: "btn ghost", onclick: async () => {
      await ctx.api.patch("/artifacts/" + a.id + "/progress", { item_id: "", state: "reset" }); progress = {}; deck = cards.slice(); i = 0; show();
    } }, "Retake all"),
    el("button", { type: "button", class: "btn ghost", onclick: async () => {
      const missed = cards.filter((x) => (progress[x.id] || {}).state === "missed");
      await ctx.api.patch("/artifacts/" + a.id + "/progress", { item_id: "", state: "reset", only: "missed" });
      const r = await ctx.api.get("/artifacts/" + a.id); progress = r.progress;
      deck = missed; i = 0; flipped = false; show();
    } }, "Retake missed"),
  );
  body.append(face, bar, el("div", { class: "study-bar" }, ""), counter);
  show();
}

// ---- quiz

function viewQuiz(a, body) {
  const items = a.content.items;
  const results = {};
  const summary = el("div", { class: "status", "aria-live": "polite" });
  items.forEach((it, n) => {
    const box = el("div", { class: "quiz-item", "data-cites": JSON.stringify(it.citations) });
    const q = el("p");
    q.innerHTML = "<strong>" + (n + 1) + ".</strong> " + renderStudyText(it.question, []);
    box.append(q);
    let read;
    if (it.type === "multiple_choice" || it.type === "multi_select") {
      const fs = el("fieldset");
      fs.append(el("legend", { class: "sr-only" }, "Question " + (n + 1) + " options"));
      it.options.forEach((o, j) => {
        const lab = el("label", { style: "display:block" });
        lab.append(el("input", { type: it.type === "multiple_choice" ? "radio" : "checkbox", name: "q-" + it.id, value: String(j) }),
          document.createTextNode(" " + o));
        fs.append(lab);
      });
      box.append(fs);
      read = () => {
        const sel = [...fs.querySelectorAll("input:checked")].map((x) => Number(x.value));
        return it.type === "multiple_choice" ? sel[0] : sel;
      };
    } else {
      const inp = el(it.type === "short_answer" ? "textarea" : "input",
        { type: "text", "aria-label": "Answer to question " + (n + 1), rows: "2", style: "width:100%" });
      box.append(inp);
      read = () => inp.value;
    }
    const fb = el("div", { class: "quiz-feedback", "aria-live": "polite" });
    box.append(el("button", { type: "button", class: "btn", onclick: async () => {
      try {
        const r = await ctx.api.post("/artifacts/" + a.id + "/grade", { item_id: it.id, answer: read() });
        results[it.id] = r.correct;
        box.classList.toggle("correct", r.correct);
        box.classList.toggle("incorrect", !r.correct);
        fb.innerHTML = (r.correct ? "✔ Correct. " : "✘ Not quite. ") + renderStudyText(it.rationale, it.citations);
        const done = Object.keys(results).length;
        const right = Object.values(results).filter(Boolean).length;
        summary.textContent = right + " of " + done + " answered correctly (" + items.length + " questions).";
      } catch (e) { ctx.fail(e); }
    } }, "Check"), fb);
    body.append(box);
  });
  body.append(summary, el("button", { type: "button", class: "btn", onclick: () => {
    const weak = items.filter((it) => results[it.id] === false).map((it) => it.question.replace(/_{3,}/g, "…"));
    $("artifact-dialog").close();
    ctx.ask(weak.length ? "Help me understand these topics I got wrong: " + weak.slice(0, 5).join(" | ")
      : "What are the most important points I should review?");
  } }, "Ask the chat about my weak areas"));
}

// ---- mind map (SVG, collapsible, zoomable; textContent only)

const SVGNS = "http://www.w3.org/2000/svg";
function viewMindMap(a, body, tools) {
  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("class", "mindmap");
  svg.setAttribute("role", "tree");
  svg.setAttribute("aria-label", "Mind map: " + a.content.title);
  const g = document.createElementNS(SVGNS, "g");
  svg.append(g);
  const collapsed = new Set();
  let view = { x: 20, y: 20, k: 1 };
  const applyView = () => g.setAttribute("transform", "translate(" + view.x + "," + view.y + ") scale(" + view.k + ")");

  const draw = () => {
    g.replaceChildren();
    let row = 0;
    const DX = 230, DY = 28;
    const place = (n, depth, parent) => {
      const me = { n, x: depth * DX, y: row * DY, parent };
      row += 1;
      const kids = collapsed.has(n.id) ? [] : n.children.map((c) => place(c, depth + 1, me));
      if (kids.length) me.y = (kids[0].y + kids[kids.length - 1].y) / 2;
      me.kids = kids;
      return me;
    };
    const root = place(a.content.root, 0, null);
    const all = [];
    const walk = (p) => { all.push(p); p.kids.forEach(walk); };
    walk(root);
    all.forEach((p) => {
      if (!p.parent) return;
      const path = document.createElementNS(SVGNS, "path");
      const x1 = p.parent.x + 6, y1 = p.parent.y, x2 = p.x - 6, y2 = p.y, mx = (x1 + x2) / 2;
      path.setAttribute("d", "M" + x1 + " " + y1 + " C" + mx + " " + y1 + " " + mx + " " + y2 + " " + x2 + " " + y2);
      g.append(path);
    });
    all.forEach((p) => {
      const node = document.createElementNS(SVGNS, "g");
      node.setAttribute("class", "mm-node" + (collapsed.has(p.n.id) ? " collapsed" : ""));
      node.setAttribute("transform", "translate(" + p.x + "," + p.y + ")");
      node.setAttribute("tabindex", "0");
      node.setAttribute("role", "treeitem");
      node.setAttribute("aria-label", p.n.label + (p.n.children.length ? (collapsed.has(p.n.id) ? ", collapsed" : ", expanded") : ""));
      const c = document.createElementNS(SVGNS, "circle");
      c.setAttribute("r", "5");
      const t = document.createElementNS(SVGNS, "text");
      t.setAttribute("x", "9");
      t.setAttribute("y", "4");
      t.textContent = mindMapLabel(p.n.label);
      const title = document.createElementNS(SVGNS, "title");
      title.textContent = p.n.label + " — click: ask about it · circle: expand/collapse";
      node.append(title, c, t);
      const toggle = () => { if (p.n.children.length) { collapsed.has(p.n.id) ? collapsed.delete(p.n.id) : collapsed.add(p.n.id); draw(); } };
      c.addEventListener("click", (e) => { e.stopPropagation(); toggle(); });
      t.addEventListener("click", (e) => { e.stopPropagation(); $("artifact-dialog").close(); ctx.ask("Tell me about " + p.n.label); });
      node.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { $("artifact-dialog").close(); ctx.ask("Tell me about " + p.n.label); }
        if (e.key === " ") { e.preventDefault(); toggle(); }
      });
      g.append(node);
    });
    applyView();
  };

  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    const k = Math.min(3, Math.max(0.3, view.k * (e.deltaY < 0 ? 1.1 : 0.9)));
    view = { ...view, k };
    applyView();
  }, { passive: false });
  let drag = null;
  svg.addEventListener("pointerdown", (e) => { drag = { x: e.clientX - view.x, y: e.clientY - view.y }; });
  window.addEventListener("pointerup", () => { drag = null; });
  svg.addEventListener("pointermove", (e) => { if (drag) { view = { ...view, x: e.clientX - drag.x, y: e.clientY - drag.y }; applyView(); } });

  tools.prepend(
    el("button", { type: "button", class: "btn ghost", "aria-label": "Zoom in", onclick: () => { view.k = Math.min(3, view.k * 1.2); applyView(); } }, "+"),
    el("button", { type: "button", class: "btn ghost", "aria-label": "Zoom out", onclick: () => { view.k = Math.max(0.3, view.k / 1.2); applyView(); } }, "−"),
    el("button", { type: "button", class: "btn ghost", onclick: () => {
      const walk = (n) => { if (n.children.length) collapsed.add(n.id); n.children.forEach(walk); };
      a.content.root.children.forEach(walk); draw();
    } }, "Collapse"),
    el("button", { type: "button", class: "btn ghost", onclick: () => { collapsed.clear(); draw(); } }, "Expand"),
    el("button", { type: "button", class: "btn ghost", onclick: () => exportPng(svg, a.title) }, "PNG"),
  );
  body.append(svg);
  draw();
}

function exportPng(svg, title) {
  const clone = svg.cloneNode(true);
  const box = svg.getBBox ? svg.querySelector("g").getBBox() : { width: 1000, height: 600 };
  const w = Math.ceil(box.width + 80), h = Math.ceil(box.height + 80);
  clone.setAttribute("width", w);
  clone.setAttribute("height", h);
  clone.setAttribute("xmlns", SVGNS);
  clone.querySelector("g").setAttribute("transform", "translate(40," + (40 - box.y) + ")");
  const style = document.createElementNS(SVGNS, "style");
  style.textContent = "text{font:12px sans-serif;fill:#15181d}circle{fill:#2a56c6}path{fill:none;stroke:#9aa4b2}";
  clone.prepend(style);
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)], { type: "image/svg+xml" }));
  const img = new Image();
  img.onload = () => {
    const c = document.createElement("canvas");
    c.width = w * 2; c.height = h * 2;
    const g2 = c.getContext("2d");
    g2.fillStyle = "#fff"; g2.fillRect(0, 0, c.width, c.height);
    g2.scale(2, 2); g2.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    const a = document.createElement("a");
    a.href = c.toDataURL("image/png");
    a.download = (title || "mind-map").replace(/[^A-Za-z0-9_-]+/g, "_") + ".png";
    a.click();
  };
  img.src = url;
}
