"""Assemble a jsc-runnable test file from web/app.js + the harness."""
import sys

app_js, harness_js, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

src = open(app_js, encoding="utf-8").read()
body = src.split("(() => {", 1)[1].rsplit("})();", 1)[0]
# Keep only the pure-logic prefix (escapeHtml, sentinels, renderMarkdown, inline);
# everything from formatBytes onward touches the DOM.
body = body[: body.index("function formatBytes")]

stub = (
    "var document={getElementById:function(){return null;},"
    "createElement:function(){return {};}};\n"
)

h = open(harness_js, encoding="utf-8").read()
# Inject genuine U+0000 escape sequences for the NUL-stripping case.
nul = "\\u0000"
h = h.replace(
    'check("raw NUL in input stripped","a  0  b"',
    'check("raw NUL in input stripped","a' + nul + " 0 " + nul + 'b"',
)

open(out_path, "w", encoding="utf-8").write(stub + body + h)
print("wrote", out_path)
