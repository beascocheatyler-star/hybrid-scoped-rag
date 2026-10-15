"""Scoped Document Chatbot Engine.

Loading the package reads `.env` into the environment so that `app.config` sees
the operator's settings regardless of how the app was launched (uvicorn, pytest,
or the CLI).
"""

import os
from pathlib import Path

__version__ = "1.0.0"


def _load_dotenv() -> None:
    """Minimal .env loader. Avoids a dependency for a dozen lines of parsing.

    Real environment variables always win, so `ANTHROPIC_API_KEY=... uvicorn ...`
    overrides the file as a user would expect.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    try:
        lines = env_path.read_text("utf-8").splitlines()
    except OSError:
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()
