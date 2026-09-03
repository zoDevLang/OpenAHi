"""Simple HTTP(S) server and mirror utilities for OpenAHI models with optional HTTPS and Basic Auth.

- serve_models(directory, port, tls=False, certfile=None, keyfile=None, username=None, password=None, generate_self_signed=False):
    Serve the given directory over HTTP or HTTPS. If username/password are provided, require HTTP Basic Auth.
    If generate_self_signed is True and no certfile/keyfile provided, attempts to generate a self-signed certificate using the 'openssl' CLI.

- mirror_models(src_dir, target_dir): Recursively copy installed models to a target directory.

Notes:
- This module tries to remain stdlib-only. For self-signed cert generation it will call the system 'openssl' command if available.
- The Basic Auth implementation is simple and suitable for LAN use. Do not expose to untrusted networks without additional protections.
"""
from __future__ import annotations
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import shutil
import threading
import os
import ssl
import base64
import subprocess
import tempfile
from typing import Optional


class BasicAuthHandler(SimpleHTTPRequestHandler):
    """HTTP request handler that enforces Basic Auth when username/password provided."""

    def __init__(self, *args, username: Optional[str] = None, password: Optional[str] = None, directory: Optional[str] = None, **kwargs):
        self._auth_username = username
        self._auth_password = password
        super().__init__(*args, directory=directory, **kwargs)

    def _is_auth_ok(self) -> bool:
        if not self._auth_username or not self._auth_password:
            return True
        header = self.headers.get('Authorization')
        if not header:
            return False
        try:
            method, token = header.split(' ', 1)
            if method.lower() != 'basic':
                return False
            decoded = base64.b64decode(token).decode('utf-8')
            user, pwd = decoded.split(':', 1)
            return (user == self._auth_username) and (pwd == self._auth_password)
        except Exception:
            return False

    def _require_auth(self):
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="OpenAHI"')
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'Authentication required')

    def do_GET(self):
        if not self._is_auth_ok():
            self._require_auth()
            return
        return super().do_GET()

    def do_HEAD(self):
        if not self._is_auth_ok():
            self._require_auth()
            return
        return super().do_HEAD()

    def do_POST(self):
        if not self._is_auth_ok():
            self._require_auth()
            return
        return super().do_POST()


def _generate_self_signed_cert(cert_path: Path, key_path: Path) -> None:
    """Generate a self-signed cert using openssl CLI into cert_path/key_path.

    This requires 'openssl' to be present on the system PATH.
    """
    if shutil.which('openssl') is None:
        raise RuntimeError("OpenSSL not found on PATH; cannot generate self-signed certificate")
    # Generate private key and certificate in one command
    cmd = [
        'openssl', 'req', '-x509', '-nodes', '-newkey', 'rsa:2048',
        '-keyout', str(key_path), '-out', str(cert_path), '-days', '365',
        '-subj', '/CN=OpenAHI'
    ]
    subprocess.check_call(cmd)


def serve_models(directory: str | Path, port: int = 8000, tls: bool = False, certfile: Optional[str] = None, keyfile: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, generate_self_signed: bool = False):
    """Serve files from directory over HTTP or HTTPS.

    If username/password provided, Basic Auth is required for requests.
    If tls=True, certfile and keyfile must be provided, or generate_self_signed=True will attempt to create a temporary cert using openssl.
    """
    directory = str(Path(directory).expanduser())

    def handler_factory(*args, **kwargs):
        return BasicAuthHandler(*args, username=username, password=password, directory=directory, **kwargs)

    server = ThreadingHTTPServer(('0.0.0.0', port), handler_factory)

    # TLS handling
    temp_cert = None
    temp_key = None
    if tls:
        if (not certfile) or (not keyfile):
            if generate_self_signed:
                # create temp files
                tf_cert = Path(tempfile.mktemp(suffix='.pem'))
                tf_key = Path(tempfile.mktemp(suffix='.key'))
                _generate_self_signed_cert(tf_cert, tf_key)
                certfile = str(tf_cert)
                keyfile = str(tf_key)
                temp_cert = tf_cert
                temp_key = tf_key
            else:
                server.server_close()
                raise RuntimeError('TLS enabled but no cert/key provided. Pass certfile+keyfile or set generate_self_signed=True')
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile, keyfile)
        server.socket = context.wrap_socket(server.socket, server_side=True)

    try:
        proto = 'https' if tls else 'http'
        auth = ' with basic auth' if username and password else ''
        print(f"Serving {proto} on 0.0.0.0 port {port} (directory: {directory}){auth}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass
        # cleanup temp certs
        if temp_cert and temp_cert.exists():
            try:
                temp_cert.unlink()
            except Exception:
                pass
        if temp_key and temp_key.exists():
            try:
                temp_key.unlink()
            except Exception:
                pass


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
