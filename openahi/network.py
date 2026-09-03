"""Network utilities for OpenAHI: manage peers and push/pull model artifacts.

Peers are stored in a simple JSON file at ~/.openahi/peers.json (or $OPENAHI_PEERS).
This module provides functions to add/remove/list peers and to push a model to a peer via HTTP PUT of a tar.gz archive to /openahi/sync.
"""
from __future__ import annotations
import json
from pathlib import Path
import tarfile
import tempfile
import shutil
import base64
import urllib.request
import urllib.error
import os
from typing import Optional

PEERS_PATH = Path(os.environ.get('OPENAHI_PEERS', Path.home() / '.openahi' / 'peers.json'))


def _ensure_peers_file():
    if not PEERS_PATH.parent.exists():
        PEERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PEERS_PATH.exists():
        PEERS_PATH.write_text('[]')


def list_peers() -> list:
    _ensure_peers_file()
    with open(PEERS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_peers(peers: list):
    _ensure_peers_file()
    with open(PEERS_PATH, 'w', encoding='utf-8') as f:
        json.dump(peers, f, indent=2)


def add_peer(url: str, name: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
    peers = list_peers()
    entry = {'url': url}
    if name:
        entry['name'] = name
    if username and password:
        entry['auth'] = {'username': username, 'password': password}
    # avoid duplicate urls
    for p in peers:
        if p.get('url') == url:
            return
    peers.append(entry)
    save_peers(peers)


def remove_peer(identifier: str):
    peers = list_peers()
    new = []
    removed = False
    for p in peers:
        if p.get('url') == identifier or p.get('name') == identifier:
            removed = True
            continue
        new.append(p)
    save_peers(new)
    return removed


def _make_tar_gz(source_dir: Path, output_filename: Path):
    with tarfile.open(output_filename, 'w:gz') as tar:
        # add source_dir contents at root of archive
        for item in source_dir.iterdir():
            tar.add(item, arcname=item.name)


def push_model(model_name: str, model_dir: Path, peer: dict) -> bool:
    """Create a tar.gz of model_dir/<model_name> and PUT it to peer['url']/openahi/sync

    peer: dict with 'url' and optional 'auth': {'username','password'}
    """
    src = model_dir / model_name
    if not src.exists():
        raise FileNotFoundError(f"Model not found: {src}")
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / f"{model_name}.tar.gz"
        _make_tar_gz(src, tar_path)
        url = peer['url'].rstrip('/') + '/openahi/sync'
        req = urllib.request.Request(url, method='PUT')
        data = tar_path.read_bytes()
        req.add_header('Content-Type', 'application/gzip')
        req.add_header('Content-Length', str(len(data)))
        req.add_header('X-Filename', f'{model_name}.tar.gz')
        if 'auth' in peer:
            cred = f"{peer['auth']['username']}:{peer['auth']['password']}"
            token = base64.b64encode(cred.encode('utf-8')).decode('utf-8')
            req.add_header('Authorization', f'Basic {token}')
        try:
            with urllib.request.urlopen(req, data=data, timeout=60) as resp:
                status = resp.getcode()
                body = resp.read().decode('utf-8', errors='ignore')
                return status == 200
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP error: {e.code} {e.reason} {e.read().decode('utf-8', errors='ignore')}")
        except Exception as e:
            raise


def get_peer_by_name_or_url(identifier: str) -> Optional[dict]:
    peers = list_peers()
    for p in peers:
        if p.get('url') == identifier or p.get('name') == identifier:
            return p
    return None
