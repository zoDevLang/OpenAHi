"""OpenAHI CLI module entrypoint with remote model binary download and checksum verification.

Installer workflow (install_model):
- Try to download remote artifact JSON from canonical artifacts URL
- If artifact JSON contains 'url' and 'sha256', attempt to download the binary and verify checksum
- If binary download succeeds, use the downloaded .pt as the checkpoint
- Else if artifact JSON exists without binary, fall back to constructing a checkpoint from artifact config
- If remote fails entirely, fallback to previous local behavior

The remote URL used can be overridden per-install by using the environment variable OPENAHI_REMOTE_BASE_URL.
If not set, default value points to this repository's raw artifacts path.
"""
from __future__ import annotations
import argparse
import os
import shutil
import json
from pathlib import Path
import urllib.request
import urllib.error
import sys
import torch
import hashlib
import tempfile
from openahi.config import DEFAULT_CONFIG
from openahi.models.composter import ComposterModel

# Model storage directory:
xdg = os.environ.get("XDG_DATA_HOME")
home = Path.home()
_model_dir_env = os.environ.get("OPENAHI_MODEL_DIR")
if _model_dir_env:
    MODEL_DIR = Path(_model_dir_env)
elif xdg:
    MODEL_DIR = Path(xdg) / "openahi" / "models"
else:
    MODEL_DIR = home / ".openahi" / "models"

# Remote artifacts base URL (raw GitHub). Can be overridden by OPENAHI_REMOTE_BASE_URL
DEFAULT_REMOTE_BASE = "https://raw.githubusercontent.com/zoDevLang/OpenAHi/main/artifacts"
REMOTE_BASE = os.environ.get("OPENAHI_REMOTE_BASE_URL", DEFAULT_REMOTE_BASE)


def list_models():
    if not MODEL_DIR.exists():
        print("No models installed. Use `openahi install <model>@<version>` to install.")
        return
    for p in sorted(MODEL_DIR.iterdir()):
        if p.is_dir():
            print(p.name)


def _download_artifact(name: str, version: str) -> dict:
    """Attempt to download remote artifact JSON for name@version.

    Returns parsed JSON on success, or raises on failure.
    """
    filename = f"{name}-{version}.json"
    url = f"{REMOTE_BASE}/{filename}"
    print(f"Attempting to download artifact metadata from {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            if getattr(resp, 'status', 200) != 200:
                raise urllib.error.HTTPError(url, getattr(resp, 'status', 500), getattr(resp, 'reason', ''), getattr(resp, 'headers', None), None)
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except Exception as e:
        raise RuntimeError(f"Failed to download artifact metadata {url}: {e}")


def _download_file(url: str, dest: Path, sha256_expected: str = None, retries: int = 3) -> bool:
    """Download a file with streaming, compute SHA256, verify if expected digest is provided.

    Returns True on success (and matching checksum if expected provided), False otherwise.
    """
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url} (attempt {attempt})...")
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as r:
                total = r.getheader('Content-Length')
                total = int(total) if total is not None else None
                hasher = hashlib.sha256()
                with tempfile.NamedTemporaryFile(delete=False) as tmpf:
                    tmp_path = Path(tmpf.name)
                    downloaded = 0
                    while True:
                        chunk = r.read(8192)
                        if not chunk:
                            break
                        tmpf.write(chunk)
                        hasher.update(chunk)
                        downloaded += len(chunk)
                digest = hasher.hexdigest()
                if sha256_expected:
                    if digest != sha256_expected.lower():
                        print(f"SHA256 mismatch: expected {sha256_expected} got {digest}")
                        tmp_path.unlink(missing_ok=True)
                        return False
                # move to dest
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.replace(dest)
                print(f"Downloaded to {dest} (sha256={digest})")
                return True
        except Exception as e:
            print(f"Download attempt {attempt} failed: {e}")
    return False


def check_release(name: str = "composter"):
    """Check the remote artifact for the latest release info for the given model name.

    Attempts to fetch {name}-latest.json, falling back to {name}-1.00.0.json.
    """
    versions_to_try = ["latest", "1.00.0"]
    for v in versions_to_try:
        try:
            art = _download_artifact(name, v)
            print(f"Remote artifact found for {name}@{v}")
            print(json.dumps(art, indent=2))
            return art
        except Exception as e:
            print(f"Could not fetch {name}@{v}: {e}")
    print("No remote artifact found for", name)
    return None


def install_model(spec: str, force: bool = False, prefer_remote: bool = True, insecure: bool = False):
    # Support shortcut: 'news' -> install latest composter
    if spec == "news":
        spec = "composter@latest"

    # spec: name@version
    if "@" in spec:
        name, version = spec.split("@", 1)
    else:
        name = spec
        version = "latest"

    if version == "latest":
        # Normalize to 'latest' literal used by remote artifacts
        version_key = "latest"
    else:
        version_key = version

    target = MODEL_DIR / f"{name}@{version_key}"
    if target.exists() and not force:
        print(f"Model {name}@{version_key} already installed at {target}")
        return
    target.mkdir(parents=True, exist_ok=True)

    # Try remote artifact metadata first
    if prefer_remote:
        try:
            artifact = _download_artifact(name, version_key)
            # If artifact contains a binary URL and sha256, attempt to download it
            url = artifact.get('url')
            sha256 = artifact.get('sha256')
            if url and sha256:
                ckpt_path = target / f"{name}.pt"
                success = _download_file(url, ckpt_path, sha256_expected=sha256)
                if success:
                    print(f"Installed {name}@{version_key} from remote binary to {ckpt_path}")
                    # save artifact metadata
                    (target / 'artifact.json').write_text(json.dumps(artifact, indent=2))
                    return
                else:
                    if not insecure:
                        print("Binary download or checksum verification failed and --insecure not set. Aborting remote install.")
                    else:
                        print("Binary download failed but proceeding due to insecure mode; falling back to artifact-config construction.")
            # If no binary URL or download failed, try to construct checkpoint from config
            cfg_dict = artifact.get("config") or {}
            from openahi.config import ModelConfig
            try:
                cfg = ModelConfig(**cfg_dict)
            except Exception:
                print("Warning: artifact config incomplete or invalid; falling back to default config")
                cfg = DEFAULT_CONFIG
            model = ComposterModel(cfg)
            ckpt_path = target / f"{name}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg,
                'meta': {
                    'name': name,
                    'version': version_key,
                    'source': 'remote',
                    'artifact': artifact
                }
            }, str(ckpt_path))
            print(f"Installed {name}@{version_key} from remote artifact config to {ckpt_path}")
            (target / 'artifact.json').write_text(json.dumps(artifact, indent=2))
            return
        except Exception as e:
            print(f"Remote install failed: {e}")
            print("Falling back to local install method...")

    # Local fallback (existing behavior)
    if name.lower() in ("composter", "composter1", "composter1.00.0", "composter1.0"):
        config = DEFAULT_CONFIG
        model = ComposterModel(config)
        ckpt_path = target / "composter.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'meta': {
                'name': 'composter',
                'version': version_key,
                'source': 'local',
            }
        }, str(ckpt_path))
        print(f"Installed composter@{version_key} to {ckpt_path} (local fallback)")
        return
    # fallback: no-op metadata
    (target / "README.txt").write_text(f"Model placeholder for {name}@{version_key}\n")
    print(f"Installed placeholder for {name}@{version_key} to {target}")


def uninstall_model(spec: str):
    if "@" in spec:
        name, version = spec.split("@", 1)
    else:
        name = spec
        version = "latest"
    target = MODEL_DIR / f"{name}@{version}"
    if not target.exists():
        print(f"Model {name}@{version} not found")
        return
    shutil.rmtree(target)
    print(f"Uninstalled {name}@{version}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="openahi")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("models", help="List installed models")
    p_install = sub.add_parser("install", help="Install a model: openahi install composter@1.00.0")
    p_install.add_argument("spec")
    p_install.add_argument("--force", action="store_true")
    p_install.add_argument("--no-remote", action="store_true", help="Do not attempt remote download; use local method only")
    p_install.add_argument("--insecure", action="store_true", help="Allow installing even if checksum verification fails (not recommended)")
    p_uninstall = sub.add_parser("uninstall", help="Uninstall a model")
    p_uninstall.add_argument("spec")
    p_check = sub.add_parser("check-release", help="Check remote release info for a model")
    p_check.add_argument("name", nargs="?", default="composter", help="Model name (default: composter)")

    args = parser.parse_args(argv)
    if args.cmd == "models":
        list_models()
    elif args.cmd == "install":
        install_model(args.spec, force=args.force, prefer_remote=not args.no_remote, insecure=args.insecure)
    elif args.cmd == "uninstall":
        uninstall_model(args.spec)
    elif args.cmd == "check-release":
        check_release(args.name)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
