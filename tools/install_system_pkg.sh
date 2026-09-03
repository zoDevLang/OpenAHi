#!/usr/bin/env bash
set -euo pipefail

# tools/install_system_pkg.sh
# Install OpenAHI system-wide and create system-level shim commands (/usr/local/bin/pkg and /usr/local/bin/openahi)
# Usage:
#   sudo ./tools/install_system_pkg.sh [--enable-service] [--service-user openahi]
#
# What it does:
#  - Builds a wheel/sdist (python -m build)
#  - Installs the package system-wide via sudo python -m pip install dist/*.whl
#  - Creates shims in /usr/local/bin/pkg and /usr/local/bin/openahi that invoke the package module
#  - Optionally installs a systemd service unit to run an example OpenAHI server at boot
#
# Requirements:
#  - sudo privileges for system install
#  - python3, pip, and build module available
#  - On systems without /usr/local/bin in PATH for all users, adjust the script accordingly

ENABLE_SERVICE=false
SERVICE_USER="openahi"

usage() {
  cat <<EOF
Usage: sudo ./tools/install_system_pkg.sh [--enable-service] [--service-user <user>]

Options:
  --enable-service     Install and enable a systemd unit (openahi-server.service) to run a model server at boot
  --service-user USER  The system user to run the service as (default: openahi)
  -h, --help           Show this help

Examples:
  sudo ./tools/install_system_pkg.sh
  sudo ./tools/install_system_pkg.sh --enable-service --service-user openahi
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --enable-service) ENABLE_SERVICE=true; shift;;
    --service-user) SERVICE_USER="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3 and try again." >&2
  exit 1
fi

PYTHON=python3

# Build the wheel/sdist
echo "Building package wheel/sdist..."
$PYTHON -m pip install --upgrade build setuptools wheel
$PYTHON -m build

DIST_GLOB="dist/*.whl"
WHEEL=$(ls $DIST_GLOB 2>/dev/null | head -n1 || true)
if [[ -z "$WHEEL" ]]; then
  echo "No wheel found in dist/. Ensure the package builds correctly." >&2
  exit 1
fi

echo "Installing package system-wide using pip..."
# Use system pip via sudo to make console scripts available globally
sudo $PYTHON -m pip install --upgrade "$WHEEL"

# Create shims in /usr/local/bin
SHIM_DIR="/usr/local/bin"
if [[ ! -d "$SHIM_DIR" ]]; then
  echo "Creating $SHIM_DIR"
  sudo mkdir -p "$SHIM_DIR"
fi

echo "Creating /usr/local/bin/pkg shim..."
sudo tee "$SHIM_DIR/pkg" >/dev/null <<'SHIM'
#!/usr/bin/env bash
python3 -m openahi.pkg_cli "$@"
SHIM
sudo chmod +x "$SHIM_DIR/pkg"

echo "Creating /usr/local/bin/openahi shim..."
sudo tee "$SHIM_DIR/openahi" >/dev/null <<'SHIM'
#!/usr/bin/env bash
python3 -m openahi.cli_full "$@"
SHIM
sudo chmod +x "$SHIM_DIR/openahi"

# Optionally install a systemd service unit
if $ENABLE_SERVICE; then
  SERVICE_UNIT="/etc/systemd/system/openahi-server.service"
  echo "Installing systemd service unit to run openahi file server at boot: $SERVICE_UNIT"
  sudo tee "$SERVICE_UNIT" >/dev/null <<UNIT
[Unit]
Description=OpenAHI Local Model Server
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
ExecStart=/usr/local/bin/openahi serve --port 8443 --directory /var/lib/openahi/models -- --tls --generate-self-signed --username openahi --password changeme
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT
  sudo systemctl daemon-reload
  sudo systemctl enable openahi-server.service
  echo "Created systemd service (enabled). Edit $SERVICE_UNIT to tune options, then start with: sudo systemctl start openahi-server.service"
fi

echo "Installation complete. You can now run:\n  pkg install openahi\n  openahi --help"
