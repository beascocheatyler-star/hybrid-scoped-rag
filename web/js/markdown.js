// Escape-first rendering for the notebook UI. Every function here returns an
// HTML string built ONLY from escaped input: document text, model output and
// note bodies are all untrusted. tests/notebook_harness.js (run by pytest
// through Node or JavaScriptCore) pins the XSS guarantees for each renderer.
//
// escapeHtml / renderMarkdown / inline are copied from the classic UI
// (web/app.js), whose behaviour tests/markdown_harness.js already pins.

/** Escape before any markup is generated. Document text is untrusted. */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Sentinel marking where a fenced code block was lifted out of the text.
//
// It must be something `escapeHtml` can never emit, so a document cannot
// forge one. U+0000 qualifies: escapeHtml only ever produces `&...;`
// entities and the original characters. A readable marker like "FENCE0"
// would be forgeable — a document containing that literal text would be
// substituted away and render as "undefined".
const FENCE = "\u0000";
const FENCE_LINE_RE = /^\u0000\d+\u0000$/;
const FENCE_RE = /\u0000(\d+)\u0000/g;

/**
 * Minimal markdown renderer, operating strictly on already-escaped text.
 * Covers what the model actually emits: headings, lists, tables, code,
 * bold/italic, links, blockquotes.
 */
function renderMarkdown(raw) {
  // Strip any NUL already present so it cannot collide with the sentinel.
  const source = escapeHtml(String(raw).replace(/\u0000/g, ""));
  const fences = [];

  // Pull fenced code out first so its contents escape further processing.
  let text = source.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const cls = lang ? ` class="language-${lang}"` : "";
    fences.push(`<pre><code${cls}>${code.replace(/\n$/, "")}</code></pre>`);
    return `${FENCE}${fences.length - 1}${FENCE}`;
  });

  const lines = text.split("\n");
  const out = [];
  let list = null;      // "ul" | "ol" | null
  let inTable = false;
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      out.push(`<p>${paragraph.join(" ")}</p>`);
      paragraph = [];
    }
  };
  const closeList = () => {
    if (list) { out.push(`</${list}>`); list = null; }
  };
  const closeTable = () => {
    if (inTable) { out.push("</tbody></table>"); inTable = false; }
  };
  const closeAll = () => { flushParagraph(); closeList(); closeTable(); };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) { closeAll(); continue; }

    if (FENCE_LINE_RE.test(trimmed)) {
      closeAll();
      out.push(trimmed);
      continue;
    }

    // Table: a header row followed by a separator row.
    if (
      !inTable &&
      /^\|.*\|$/.test(trimmed) &&
      i + 1 < lines.length &&
      /^\|[\s:|-]+\|$/.test(lines[i + 1].trim())
    ) {
      closeAll();
      const headers = trimmed.slice(1, -1).split("|").map((c) => c.trim());
      out.push(
        "<table><thead><tr>" +
          headers.map((h) => `<th>${inline(h)}</th>`).join("") +
          "</tr></thead><tbody>"
      );
      inTable = true;
      i++; // consume the separator row
      continue;
    }
    if (inTable) {
      if (/^\|.*\|$/.test(trimmed)) {
        const cells = trimmed.slice(1, -1).split("|").map((c) => c.trim());
        out.push("<tr>" + cells.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>");
        continue;
      }
      closeTable();
    }

    let m;
    if ((m = trimmed.match(/^(#{1,6})\s+(.*)$/))) {
      closeAll();
      const level = Math.min(m[1].length + 2, 6); // demote: h1 -> h3
      out.push(`<h${level}>${inline(m[2])}</h${level}>`);
      continue;
    }

    if ((m = trimmed.match(/^[-*+]\s+(.*)$/))) {
      flushParagraph();
      closeTable();
      if (list !== "ul") { closeList(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inline(m[1])}</li>`);
      continue;
    }

    if ((m = trimmed.match(/^\d+[.)]\s+(.*)$/))) {
      flushParagraph();
      closeTable();
      if (list !== "ol") { closeList(); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${inline(m[1])}</li>`);
      continue;
    }

    if ((m = trimmed.match(/^&gt;\s?(.*)$/))) {
      closeAll();
      out.push(`<blockquote>${inline(m[1])}</blockquote>`);
      continue;
    }

    if (/^(---|\*\*\*|___)$/.test(trimmed)) {
      closeAll();
      out.push("<hr />");
      continue;
    }

    closeList();
    paragraph.push(inline(trimmed));
  }
  closeAll();

  let html = out.join("");
  html = html.replace(FENCE_RE, (_, n) => fences[Number(n)]);
  return html;
}

/** Inline span-level markdown. Input is already HTML-escaped. */
function inline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|\W)\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|\W)_([^_\n]+)_/g, "$1<em>$2</em>")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
}

// ---------------------------------------------------------------------------
// Citations
// ---------------------------------------------------------------------------

// Replace [n] markers (digits only, so nothing user-controlled reaches the
// attribute values) with focusable chip buttons. Runs on already-escaped HTML.
function citationChips(html, citations) {
  var byMarker = {};
  (citations || []).forEach(function (c) { byMarker[c.marker] = c; });
  return html.replace(/\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\](?!\()/g, function (whole, list) {
    return list.split(",").map(function (part) {
      var n = parseInt(part, 10);
      var c = byMarker[n];
      if (!c) return "";
      var weak = c.weak ? " cite-weak" : "";
      var label = "Citation " + n + ": " + escapeHtml(c.source_title || "source") +
        (c.weak ? " (weak support)" : "");
      return '<button type="button" class="cite' + weak + '" data-marker="' + n +
        '" aria-label="' + label + '">' + n + "</button>";
    }).join("");
  });
}

// Render an assistant answer. Plain prose answers are rendered sentence by
// sentence so unsupported sentences can be wrapped; answers with block
// markdown (lists, headings, code, tables) are rendered whole, and a note
// counts the unverified sentences instead.
function renderAnswer(text, citations, sentences) {
  var raw = String(text == null ? "" : text);
  var blocky = /(^|\n)\s*([-*+]\s|\d+[.)]\s|#{1,6}\s|\||```|>)/.test(raw);
  var unsupported = (sentences || []).filter(function (s) {
    return s.markers && s.markers.length && s.supported === false;
  });
  if (blocky || !sentences || !sentences.length) {
    var html = citationChips(renderMarkdown(raw), citations);
    if (unsupported.length) {
      html += '<p class="verify-note" role="note">' + unsupported.length +
        (unsupported.length === 1 ? " sentence" : " sentences") +
        " could not be verified against the cited passages.</p>";
    }
    return html;
  }
  var parts = sentences.map(function (s) {
    var body = inline(escapeHtml(String(s.text || "").replace(/\u0000/g, "")));
    var marks = (s.markers || []).map(function (n) { return "[" + parseInt(n, 10) + "]"; }).join("");
    var chips = citationChips(marks, citations);
    var cls = s.markers && s.markers.length && s.supported === false ? "unsupported" : "";
    var inner = body + (chips ? " " + chips : "");
    return cls
      ? '<span class="' + cls + '" title="Not verified against the cited passage">' + inner + "</span>"
      : inner;
  });
  return "<p>" + parts.join(" ") + "</p>";
}

function formatMs(ms) {
  var t = Math.max(0, Math.floor(Number(ms) / 1000));
  var h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = t % 60;
  var mm = (h ? String(m).padStart(2, "0") : String(m));
  return (h ? h + ":" : "") + mm + ":" + String(s).padStart(2, "0");
}

function locationLabel(c) {
  var bits = [];
  if (c.page != null) bits.push("p. " + parseInt(c.page, 10));
  if (c.slide != null) bits.push("slide " + parseInt(c.slide, 10));
  if (c.t_start_ms != null) bits.push(formatMs(c.t_start_ms));
  if (c.section_path) bits.push(escapeHtml(c.section_path));
  return bits.join(" · ");
}

// Hover / focus card for one citation.
function renderCitationCard(c) {
  if (!c) return "";
  var loc = locationLabel(c);
  return '<div class="cite-card-title">' + escapeHtml(c.source_title || "Source") + "</div>" +
    (loc ? '<div class="cite-card-loc">' + loc + "</div>" : "") +
    '<blockquote class="cite-card-quote">' + escapeHtml(c.quote || "") + "</blockquote>" +
    (c.weak ? '<div class="cite-card-weak">Weak support: the passage may not back this claim.</div>' : "") +
    '<div class="cite-card-hint">Click to open the source</div>';
}

// ---------------------------------------------------------------------------
// Notes and source viewer
// ---------------------------------------------------------------------------

function renderNote(note) {
  var title = escapeHtml((note && note.title) || "Untitled note");
  var body = citationChips(renderMarkdown((note && note.body_md) || ""), (note && note.citations) || []);
  return '<h3 class="note-title">' + title + "</h3>" + '<div class="note-body">' + body + "</div>";
}

// Source text with [start, end) highlighted. Each segment is escaped
// separately, so the <mark> is the only markup in the output.
function renderHighlighted(text, start, end) {
  var t = String(text == null ? "" : text);
  var a = Math.max(0, Math.min(t.length, parseInt(start, 10) || 0));
  var b = Math.max(a, Math.min(t.length, parseInt(end, 10) || 0));
  if (start == null || end == null || b <= a) return escapeHtml(t);
  return escapeHtml(t.slice(0, a)) + '<mark id="hl" tabindex="-1">' + escapeHtml(t.slice(a, b)) +
    "</mark>" + escapeHtml(t.slice(b));
}

// ---------------------------------------------------------------------------
// Studio
// ---------------------------------------------------------------------------

// Data table with per-cell citation chips. Cell text is escaped; chips come
// from citationChips over digit-only markers.
function renderDataTable(content) {
  var cols = (content && content.columns) || [];
  var head = "<thead><tr>" + cols.map(function (c) { return '<th scope="col">' + escapeHtml(c) + "</th>"; }).join("") + "</tr></thead>";
  var body = ((content && content.rows) || []).map(function (r) {
    return "<tr>" + (r.cells || []).map(function (c) {
      var marks = (c.citations || []).map(function (x) { return "[" + parseInt(x.marker, 10) + "]"; }).join("");
      return "<td>" + escapeHtml(c.text || "") + (marks ? " " + citationChips(marks, c.citations) : "") + "</td>";
    }).join("") + "</tr>";
  }).join("");
  return '<table class="data-table">' + head + "<tbody>" + body + "</tbody></table>";
}

// A flashcard or quiz text block: inline markdown on escaped text, plus chips.
function renderStudyText(text, citations) {
  var marks = (citations || []).map(function (x) { return "[" + parseInt(x.marker, 10) + "]"; }).join("");
  return inline(escapeHtml(String(text == null ? "" : text).replace(/\u0000/g, ""))) +
    (marks ? " " + citationChips(marks, citations) : "");
}

// Mind-map labels are drawn with SVG textContent (never innerHTML); this only
// normalises and truncates, and is exercised by the harness for safety.
function mindMapLabel(label, max) {
  var t = String(label == null ? "" : label).replace(/\s+/g, " ").trim();
  var m = max || 42;
  return t.length > m ? t.slice(0, m - 1) + "…" : t;
}

export {
  escapeHtml, renderMarkdown, inline, citationChips, renderAnswer, renderCitationCard,
  renderNote, renderHighlighted, formatMs, locationLabel, renderDataTable, renderStudyText,
  mindMapLabel
};
