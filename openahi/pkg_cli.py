"""A very small 'pkg' helper to install Python packages from terminal.

Usage: python -m openahi.pkg_cli install openahi
This will try to pip install the package; if that fails and package == 'openahi', it will try to install from the current directory.

This script is intended for local developer convenience only and does not replace system package managers.
"""
from __future__ import annotations
import argparse
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pkg")
    sub = parser.add_subparsers(dest="cmd")
    p_install = sub.add_parser("install")
    p_install.add_argument("package")
    args = parser.parse_args(argv)
    if args.cmd == "install":
        pkg = args.package
        print(f"Attempting to install {pkg} via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            print(f"Successfully installed {pkg}")
            return
        except subprocess.CalledProcessError:
            print(f"pip install {pkg} failed.")
            if pkg == "openahi":
                print("Trying to install from the current directory with 'pip install .'")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "."])
                    print("Successfully installed openahi from current directory")
                    return
                except subprocess.CalledProcessError:
                    print("Failed to install openahi from current directory. Please run 'pip install .' manually or check pip output.")
            else:
                print("Please install the package manually.")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
