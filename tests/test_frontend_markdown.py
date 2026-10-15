"""Run the frontend markdown/XSS tests through a JavaScript engine.

The renderer writes to `innerHTML`, so an escaping mistake there is a real XSS
hole. These assertions live in `tests/markdown_harness.js` and are executed here
so a normal `pytest` run covers them.

Skipped automatically when no JS engine is present — `node` if installed,
otherwise macOS's bundled JavaScriptCore.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
JSC = Path(
    "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
)


def _js_engine():
    node = shutil.which("node")
    if node:
        return [node]
    if JSC.exists():
        return [str(JSC)]
    return None


@pytest.mark.skipif(_js_engine() is None, reason="no JavaScript engine available")
def test_markdown_renderer_and_escaping(tmp_path):
    bundle = tmp_path / "md_run.js"

    build = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_md_test.py"),
            str(ROOT / "web" / "app.js"),
            str(ROOT / "tests" / "markdown_harness.js"),
            str(bundle),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    result = subprocess.run(
        _js_engine() + [str(bundle)], capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    print(output)

    assert "FAIL" not in output, "A frontend assertion failed:\n{}".format(output)
    assert result.returncode == 0, output
    assert "0 failed" in output
