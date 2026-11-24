"""Assemble a Node/JSC-runnable test file from web/js/markdown.js + harness.

The module's `import`/`export` lines are stripped so the same bundle runs in
JavaScriptCore (no module support from the CLI) and Node."""
import re
import sys

module_js, harness_js, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(module_js, encoding="utf-8").read()
src = re.sub(r"^\s*import\s.*?;\s*$", "", src, flags=re.M)
src = re.sub(r"^export\s*\{[^}]*\};?\s*$", "", src, flags=re.M | re.S)
src = re.sub(r"^export\s+(?=(function|const|let|var)\b)", "", src, flags=re.M)
shim = 'if (typeof print === "undefined") { globalThis.print = (...a) => console.log(...a); }\n'
h = open(harness_js, encoding="utf-8").read()
open(out_path, "w", encoding="utf-8").write(shim + src + "\n" + h)
print("wrote", out_path)
