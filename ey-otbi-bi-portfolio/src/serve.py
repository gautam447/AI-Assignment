#!/usr/bin/env python3
"""Serve the generated dashboards on http://127.0.0.1:8765/"""

from __future__ import annotations

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not (OUT / "index.html").exists():
        raise SystemExit("Run python3 src/build_portfolio.py first.")
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(OUT))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"OTBI/BIP demo: http://127.0.0.1:{args.port}/index.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
