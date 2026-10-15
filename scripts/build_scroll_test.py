"""Assemble a jsc-runnable test file from web/app.js + the scroll harness.

Extracts the pure scroll-geometry helpers (isPinnedToBottom, scrollToBottom,
scrollToTop, updateScrollButtons) so their logic is testable without a DOM.
"""
import sys

app_js, harness_js, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

src = open(app_js, encoding="utf-8").read()
body = src.split("(() => {", 1)[1].rsplit("})();", 1)[0]

# Keep from the scroll section up to (not including) bindScrollControls, which
# attaches real event listeners. The helpers above it are pure geometry.
start = body.index("const STICK_THRESHOLD")
end = body.index("function bindScrollControls")
body = body[start:end]

# `el` is populated from the DOM in the real app; the harness swaps in fakes.
stub = "var el = {};\n"

h = open(harness_js, encoding="utf-8").read()

open(out_path, "w", encoding="utf-8").write(stub + body + h)
print("wrote", out_path)
