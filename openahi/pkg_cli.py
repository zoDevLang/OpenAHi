"""A small 'pkg' helper to install Python packages and perform initial OpenAHI setup.

Usage:
  pkg install <package> [--no-auto]
  pkg setup [--no-model] [--no-check] [--no-deps]

Behavior:
- pkg install <package>
  - If package == 'openahi' and a requirements.txt exists in the repo root, it will attempt to install requirements first.
  - Attempts `pip install <package>`; falls back to `pip install .` for openahi if the first attempt fails.
  - By default, for openahi it will auto-run `openahi install composter@1.00.0` after installing unless --no-auto is provided.

- pkg setup
  - Convenience flow to perform a first-time setup for OpenAHI. It will:
      1) Install runtime dependencies (unless --no-deps)
      2) Pip-install the openahi package
      3) Install the default model (composter@1.00.0) unless --no-model
      4) Check remote release metadata (unless --no-check)
  - Designed so a single copy/paste command on a fresh machine will bootstrap OpenAHI.

Notes:
- If console scripts are not on PATH, the script falls back to module invocation (`python -m openahi.cli` / `python -m openahi.pkg_cli`).
- This helper is intended for convenience and does not replace system package managers.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import os
from pathlib import Path


def _run_list(cmd_list):
    print('> ' + ' '.join(cmd_list))
    subprocess.check_call(cmd_list)


def _run_shell(cmd):
    print(f"> {cmd}")
    return subprocess.check_call(cmd, shell=True)


def _pip_install(pkg: str):
    try:
        _run_list([sys.executable, '-m', 'pip', 'install', pkg])
        return True
    except subprocess.CalledProcessError:
        return False


def _pip_install_local():
    try:
        _run_list([sys.executable, '-m', 'pip', 'install', '.'])
        return True
    except subprocess.CalledProcessError:
        return False


def install_package(pkg: str, no_auto: bool = False):
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
            _run_list([sys.executable, '-m', 'pip', 'install', '-r', str(req_file)])
    except subprocess.CalledProcessError:
        print("Failed to install requirements. Proceeding to attempt to install the package itself.")

    # Try pip install package
    if _pip_install(pkg):
        print(f"Successfully installed {pkg}")
    else:
        print(f"pip install {pkg} failed.")
        if pkg == "openahi":
            print("Trying to install from the current directory with 'pip install .' (if you're in the repo root)")
            if _pip_install_local():
                print("Successfully installed openahi from current directory")
            else:
                print("Failed to install openahi from current directory. Please run 'pip install .' manually or check pip output.")
                return
        else:
            print("Please install the package manually.")
            return

    # Auto-run model install for openahi unless --no-auto
    if pkg == 'openahi' and not no_auto:
        print("Auto-running: openahi install composter@1.00.0")
        try:
            # Prefer console script; fallback to module invocation
            try:
                _run_shell('openahi install composter@1.00.0')
            except Exception:
                print("Console script 'openahi' not available on PATH, using module invocation")
                _run_list([sys.executable, '-m', 'openahi.cli', 'install', 'composter@1.00.0'])
        except Exception as e:
            print(f"Auto-install of composter failed: {e}")
            print("You can retry manually: openahi install composter@1.00.0 or python -m openahi.cli install composter@1.00.0")


def setup_openahi(no_model: bool = False, no_check: bool = False, no_deps: bool = False):
    print("Starting OpenAHI setup...\n")
    try:
        repo_root = Path(__file__).resolve().parents[1]
    except Exception:
        repo_root = Path('.')

    # 1) Install dependencies
    if not no_deps:
        req_file = repo_root / 'requirements.txt'
        if req_file.exists():
            print(f"Installing runtime requirements from {req_file}...")
            try:
                _run_list([sys.executable, '-m', 'pip', 'install', '-r', str(req_file)])
            except subprocess.CalledProcessError:
                print("Failed to install requirements. You may need to install PyTorch manually for your platform.")
        else:
            print("No requirements.txt found; skipping dependency install.")
    else:
        print("Skipping dependency install (--no-deps)")

    # 2) Install the openahi package
    print("Installing openahi package...")
    if not _pip_install('openahi'):
        print("pip install openahi failed; trying local install (pip install .)")
        if not _pip_install_local():
            print("Failed to install openahi. Aborting setup.")
            return

    # 3) Install default model
    if not no_model:
        print("Installing default model: composter@1.00.0")
        try:
            try:
                _run_shell('openahi install composter@1.00.0')
            except Exception:
                print("Console script 'openahi' not available on PATH, using module invocation")
                _run_list([sys.executable, '-m', 'openahi.cli', 'install', 'composter@1.00.0'])
        except Exception as e:
            print(f"Failed to install model: {e}")
            print("You can retry manually later: openahi install composter@1.00.0")
    else:
        print("Skipping default model install (--no-model)")

    # 4) Check for release metadata
    if not no_check:
        print("Checking remote release metadata for composter...")
        try:
            try:
                _run_shell('openahi check-release composter')
            except Exception:
                print("Console script 'openahi' not available on PATH, using module invocation")
                _run_list([sys.executable, '-m', 'openahi.cli', 'check-release', 'composter'])
        except Exception as e:
            print(f"check-release failed: {e}")
    else:
        print("Skipping remote release check (--no-check)")

    print("OpenAHI setup complete.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pkg")
    sub = parser.add_subparsers(dest="cmd")

    p_install = sub.add_parser("install")
    p_install.add_argument("package")
    p_install.add_argument("--no-auto", action="store_true", help="Do not auto-run model install after package installation")

    p_setup = sub.add_parser("setup")
    p_setup.add_argument("--no-model", action="store_true", help="Do not install default model during setup")
    p_setup.add_argument("--no-check", action="store_true", help="Do not run check-release during setup")
    p_setup.add_argument("--no-deps", action="store_true", help="Do not attempt to install requirements during setup")

    args = parser.parse_args(argv)
    if args.cmd == "install":
        install_package(args.package, no_auto=args.no_auto)
    elif args.cmd == "setup":
        setup_openahi(no_model=args.no_model, no_check=args.no_check, no_deps=args.no_deps)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
