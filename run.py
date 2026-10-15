#!/usr/bin/env python3
"""Start the server.

A convenience wrapper so the project runs with `python run.py` and no uvicorn
incantation. Checks configuration first and prints something useful if the key
is missing, rather than failing later on the first question.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the scoped document chatbot.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Port.")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on edit.")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print(
            "Missing dependencies. Install them first:\n"
            "    pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    from app.config import get_settings

    settings = get_settings()

    print("\n  Scoped Document Chatbot")
    print("  " + "-" * 40)
    print("  Model:      {}".format(settings.model))
    print("  Data dir:   {}".format(settings.data_dir))
    print("  Scope floor:{:.2f}".format(settings.relevance_floor))

    if settings.has_api_key:
        print("  API key:    configured")
    else:
        print("  API key:    MISSING")
        print()
        print("  The app will start and you can upload documents, but questions")
        print("  will fail until a key is set. To fix:")
        print("      cp .env.example .env     then add your key to .env")
        print("  Get a key at https://console.anthropic.com/settings/keys")

    print()
    print("  → http://{}:{}".format(
        "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host, args.port
    ))
    print()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
