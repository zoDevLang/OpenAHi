"""Simple HTTP server and mirror utilities for OpenAHI models.

- serve_models(directory, port): serve the given directory over HTTP using Python's http.server.
- mirror_models(src_dir, target_dir): recursively copy installed models to a target directory so they are available offline in other locations.

These utilities are intentionally dependency-free (stdlib only) so they work on Termux and minimal systems.
"""
from __future__ import annotations
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import shutil
import threading
import os
from typing import Optional


def serve_models(directory: str | Path, port: int = 8000):
    directory = str(Path(directory).expanduser())
    # Use SimpleHTTPRequestHandler(directory=...) available in Python 3.7+
    handler_class = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=directory, **kwargs)
    server = ThreadingHTTPServer(('0.0.0.0', port), handler_class)
    try:
        print(f"Serving HTTP on 0.0.0.0 port {port} (directory: {directory})")
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.shutdown()
        server.server_close()


def mirror_models(src: str | Path, target: str | Path):
    src = Path(src)
    target = Path(target)
    if not src.exists():
        raise FileNotFoundError(f"Source models directory not found: {src}")
    # Copytree-like behavior but merge into existing target
    target.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest = target / item.name
        if item.is_dir():
            # copy entire directory
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
