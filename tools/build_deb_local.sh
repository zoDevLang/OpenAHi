#!/usr/bin/env bash
set -euo pipefail

# tools/build_deb_local.sh
# Build a local Debian package that installs the repository into /opt/openahi
# and creates shims in /usr/local/bin so `pkg` and `openahi` are available system-wide.
# This is fully local and does not use external package registries or APIs.
#
# Usage:
#   sudo ./tools/build_deb_local.sh --version 1.0.0
#
# Requirements:
#   - dpkg-deb (usually present on Debian/Ubuntu systems)
#   - rsync
#
# Output: dist/openahi-<version>_all.deb

VERSION="dev"
MAINTAINER="OpenAHI <noreply@example.com>"
DESCRIPTION="OpenAHI - local model server and tools (self-contained install)"
ARCH="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --maintainer) MAINTAINER="$2"; shift 2;;
    --description) DESCRIPTION="$2"; shift 2;;
    --help|-h) echo "Usage: $0 --version X.Y.Z"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR=$(mktemp -d)
PKG_DIR="$BUILD_DIR/openahi-$VERSION"

echo "Building local deb in $BUILD_DIR"

# Create package filesystem layout
mkdir -p "$PKG_DIR/opt/openahi"
mkdir -p "$PKG_DIR/usr/local/bin"
mkdir -p "$PKG_DIR/DEBIAN"

# Copy repository files to /opt/openahi (exclude dev/CI files)
rsync -a --exclude '.git' --exclude '.github' --exclude 'dist' --exclude '__pycache__' --exclude 'build' --exclude 'venv' "$REPO_ROOT/" "$PKG_DIR/opt/openahi/"

# Create simple shims that invoke the package module installed in /opt/openahi
cat > "$PKG_DIR/usr/local/bin/pkg" <<'SHIM'
#!/usr/bin/env bash
PYTHON=/usr/bin/python3
exec "$PYTHON" -m openahi.pkg_cli "$@"
SHIM
chmod 0755 "$PKG_DIR/usr/local/bin/pkg"

cat > "$PKG_DIR/usr/local/bin/openahi" <<'SHIM'
#!/usr/bin/env bash
PYTHON=/usr/bin/python3
# ensure /opt/openahi on sys.path
export PYTHONPATH="/opt/openahi:$PYTHONPATH"
exec "$PYTHON" -m openahi.cli_full "$@"
SHIM
chmod 0755 "$PKG_DIR/usr/local/bin/openahi"

# Create control file
cat > "$PKG_DIR/DEBIAN/control" <<CONTROL
Package: openahi
Version: $VERSION
Section: web
Priority: optional
Architecture: $ARCH
Maintainer: $MAINTAINER
Depends: python3 (>= 3.8)
Description: $DESCRIPTION
CONTROL

# Optionally create postinst to set permissions
cat > "$PKG_DIR/DEBIAN/postinst" <<'POST'
#!/bin/sh
set -e
# Ensure files are owned appropriately
chown -R root:root /opt/openahi
chmod -R u+rwX,go+rX /opt/openahi
chmod 755 /usr/local/bin/pkg /usr/local/bin/openahi
# create model dir
mkdir -p /var/lib/openahi/models
chown -R root:root /var/lib/openahi
exit 0
POST
chmod 0755 "$PKG_DIR/DEBIAN/postinst"

# Build deb
mkdir -p "$REPO_ROOT/dist"
OUT="$REPO_ROOT/dist/openahi-${VERSION}_all.deb"
if command -v dpkg-deb >/dev/null 2>&1; then
  dpkg-deb --build "$PKG_DIR" "$OUT"
  echo "Built package: $OUT"
else
  echo "dpkg-deb not found. Please install dpkg-deb and rerun." >&2
  exit 1
fi

# Cleanup
rm -rf "$BUILD_DIR"

