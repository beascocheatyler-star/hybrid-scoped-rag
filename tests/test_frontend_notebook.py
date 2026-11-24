"""Run the notebook-UI renderer XSS harness (tests/notebook_harness.js)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_frontend_markdown import _js_engine

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(_js_engine() is None, reason="no JavaScript engine available")
def test_notebook_renderers(tmp_path):
    bundle = tmp_path / "nb_run.js"
    build = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_nb_test.py"),
                            str(ROOT / "web" / "js" / "markdown.js"),
                            str(ROOT / "tests" / "notebook_harness.js"), str(bundle)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    result = subprocess.run(_js_engine() + [str(bundle)], capture_output=True, text=True)
    out = result.stdout + result.stderr
    print(out)
    assert "FAIL" not in out, out
    assert result.returncode == 0, out
    assert " 0 failed" in out
