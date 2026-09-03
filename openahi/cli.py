"""OpenAHI CLI module entrypoint with remote model download support.

Installer workflow (install_model):
- Try to download a remote artifact JSON from a known canonical URL (GitHub raw URL in this repo)
- Parse the artifact metadata and build a checkpoint (torch.save) locally at the model storage directory
- If remote download fails, fallback to local method (existing behavior)

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
    print(f"Attempting to download artifact from {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            if resp.status != 200:
                raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
            data = resp.read().decode("utf-8")
            return json.loads(data)
    except Exception as e:
        raise RuntimeError(f"Failed to download artifact {url}: {e}")


def install_model(spec: str, force: bool = False, prefer_remote: bool = True):
    # spec: name@version
    if "@" in spec:
        name, version = spec.split("@", 1)
    else:
        name = spec
        version = "latest"
    target = MODEL_DIR / f"{name}@{version}"
    if target.exists() and not force:
        print(f"Model {name}@{version} already installed at {target}")
        return
    target.mkdir(parents=True, exist_ok=True)

    if prefer_remote:
        try:
            artifact = _download_artifact(name, version)
            # parse config from artifact
            cfg_dict = artifact.get("config") or {}
            # merge with DEFAULT_CONFIG for any missing fields
            cfg = DEFAULT_CONFIG
            # create a new ModelConfig-like object. We saved dataclass in config.py; create using kwargs
            from openahi.config import ModelConfig
            try:
                cfg = ModelConfig(**cfg_dict)
            except Exception:
                print("Warning: artifact config incomplete or invalid; falling back to default config")
                cfg = DEFAULT_CONFIG
            # build model and save checkpoint
            model = ComposterModel(cfg)
            ckpt_path = target / f"{name}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg,
                'meta': {
                    'name': name,
                    'version': version,
                    'source': 'remote',
                    'artifact': 'downloaded-json'
                }
            }, str(ckpt_path))
            print(f"Installed {name}@{version} from remote artifact to {ckpt_path}")
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
                'version': version,
                'source': 'local',
            }
        }, str(ckpt_path))
        print(f"Installed composter@{version} to {ckpt_path} (local fallback)")
        return
    # fallback: no-op metadata
    (target / "README.txt").write_text(f"Model placeholder for {name}@{version}\n")
    print(f"Installed placeholder for {name}@{version} to {target}")


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
    p_uninstall = sub.add_parser("uninstall", help="Uninstall a model")
    p_uninstall.add_argument("spec")

    args = parser.parse_args(argv)
    if args.cmd == "models":
        list_models()
    elif args.cmd == "install":
        install_model(args.spec, force=args.force, prefer_remote=not args.no_remote)
    elif args.cmd == "uninstall":
        uninstall_model(args.spec)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
