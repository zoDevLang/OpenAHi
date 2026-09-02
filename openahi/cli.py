"""Command-line interface for openahi package"""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
import torch
from openahi.config import DEFAULT_CONFIG
from openahi.models.composter import ComposterModel

MODEL_DIR = Path.home() / ".openahi" / "models"


def list_models():
    if not MODEL_DIR.exists():
        print("No models installed. Use `openahi install <model>@<version>` to install.")
        return
    for p in sorted(MODEL_DIR.iterdir()):
        if p.is_dir():
            print(p.name)


def install_model(spec: str, force: bool = False):
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
    # For now, if installing composter, create a checkpoint from the current package model
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
        print(f"Installed composter@{version} to {ckpt_path}")
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
    p_uninstall = sub.add_parser("uninstall", help="Uninstall a model")
    p_uninstall.add_argument("spec")

    args = parser.parse_args(argv)
    if args.cmd == "models":
        list_models()
    elif args.cmd == "install":
        install_model(args.spec, force=args.force)
    elif args.cmd == "uninstall":
        uninstall_model(args.spec)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
