"""A very small 'pkg' helper to install Python packages from terminal.

Behavior:
- pkg install <package>
  - pip install -r requirements.txt (if available in repo and package==openahi)
  - pip install <package>
  - by default, will auto-run `python -m openahi.cli install composter@1.00.0` after successful install unless --no-auto is provided.

This helper is a convenience wrapper to make onboarding easier for users. It prints helpful guidance if PyTorch installation fails.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import os
from pathlib import Path


def _run(cmd):
    print(f"> {cmd}")
    return subprocess.check_call(cmd, shell=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pkg")
    sub = parser.add_subparsers(dest="cmd")
    p_install = sub.add_parser("install")
    p_install.add_argument("package")
    p_install.add_argument("--no-auto", action="store_true", help="Do not auto-run model install after package installation")
    args = parser.parse_args(argv)
    if args.cmd == "install":
        pkg = args.package
        print(f"Attempting to install {pkg} via pip...")
        # If installing from repo and requirements exists, try to install requirements first
        try:
            repo_root = Path(__file__).resolve().parents[1]
        except Exception:
            repo_root = Path('.')
        req_file = repo_root / 'requirements.txt'
        try:
            if pkg == 'openahi' and req_file.exists():
                print(f"Installing runtime requirements from {req_file}...")
                _run(f"{sys.executable} -m pip install -r {str(req_file)}")
        except subprocess.CalledProcessError:
            print("Failed to install requirements. Proceeding to attempt to install the package itself.")
        # Try pip install package
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            print(f"Successfully installed {pkg}")
        except subprocess.CalledProcessError:
            print(f"pip install {pkg} failed.")
            if pkg == "openahi":
                print("Trying to install from the current directory with 'pip install .' (if you're in the repo root)")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "."])
                    print("Successfully installed openahi from current directory")
                except subprocess.CalledProcessError:
                    print("Failed to install openahi from current directory. Please run 'pip install .' manually or check pip output.")
                    return
            else:
                print("Please install the package manually.")
                return
        # Auto-run model install for openahi unless --no-auto
        if pkg == 'openahi' and not args.no_auto:
            print("Auto-running: openahi install composter@1.00.0")
            try:
                # Prefer console script; fallback to module invocation
                try:
                    _run("openahi install composter@1.00.0")
                except Exception:
                    print("Console script 'openahi' not available on PATH, using module invocation")
                    _run(f"{sys.executable} -m openahi.cli install composter@1.00.0")
            except Exception as e:
                print(f"Auto-install of composter failed: {e}")
                print("You can retry manually: openahi install composter@1.00.0 or python -m openahi.cli install composter@1.00.0")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
