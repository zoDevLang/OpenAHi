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
import tarfile
from typing import Optional


class BasicAuthHandler(SimpleHTTPRequestHandler):
    """HTTP request handler that enforces Basic Auth when username/password provided,
    and supports PUT uploads to /openahi/sync to receive model tarballs.
    """

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
        # Keep compatibility for existing static behavior (not used for sync)
        if not self._is_auth_ok():
            self._require_auth()
            return
        return super().do_POST()

    def do_PUT(self):
        # Accept tarball uploads to /openahi/sync
        if not self._is_auth_ok():
            self._require_auth()
            return
        if not self.path.startswith('/openahi/sync'):
            # Respond 405 Method Not Allowed for other PUTs
            self.send_response(405)
            self.end_headers()
            return
        # Read headers
        length = int(self.headers.get('Content-Length', '0'))
        filename = self.headers.get('X-Filename')
        if not filename:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Missing X-Filename header')
            return
        # write to temp file
        try:
            data = self.rfile.read(length)
            tf = tempfile.NamedTemporaryFile(delete=False)
            tf.write(data)
            tf.flush()
            tf.close()
            # extract tarball into the served directory (self.directory)
            target_root = Path(self.directory)
            target_root.mkdir(parents=True, exist_ok=True)
            # extract into a subdirectory named after filename (without extension)
            name = Path(filename).stem
            dest_dir = target_root / name
            # remove existing dest_dir to replace
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tf.name, 'r:gz') as t:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])
                for member in t.getmembers():
                    member_path = dest_dir / member.name
                    if not is_within_directory(str(dest_dir), str(member_path)):
                        raise Exception("Attempted Path Traversal in Tar File")
                t.extractall(path=dest_dir)
            os.unlink(tf.name)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            msg = f'Upload failed: {e}'
            try:
                self.wfile.write(msg.encode('utf-8'))
            except Exception:
                pass


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
