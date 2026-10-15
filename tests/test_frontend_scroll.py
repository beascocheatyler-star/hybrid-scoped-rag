"""Run the transcript scroll-behaviour tests through a JavaScript engine.

Regression cover for the reported bug: once a conversation grew past a few
messages, every new message forced the view to the bottom, so the user could
not scroll up and stay there. These assertions live in
`tests/scroll_harness.js` and run here as part of a normal `pytest` run.

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
def test_transcript_scroll_behaviour(tmp_path):
    bundle = tmp_path / "scroll_run.js"

    build = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_scroll_test.py"),
            str(ROOT / "web" / "app.js"),
            str(ROOT / "tests" / "scroll_harness.js"),
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

    assert "FAIL" not in output, "A scroll assertion failed:\n{}".format(output)
    assert result.returncode == 0, output
    assert "0 failed" in output
