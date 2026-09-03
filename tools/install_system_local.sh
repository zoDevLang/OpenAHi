#!/usr/bin/env bash
set -euo pipefail

# tools/install_system_local.sh
# Local system installer that installs the repository into /opt/openahi,
# creates /usr/local/bin shims, and installs a simple systemd unit to run
# the model server at boot. All operations are local; no external APIs.
#
# Usage:
#  sudo ./tools/install_system_local.sh [--service] [--service-user openahi]

ENABLE_SERVICE=false
SERVICE_USER="openahi"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service) ENABLE_SERVICE=true; shift;;
    --service-user) SERVICE_USER="$2"; shift 2;;
    -h|--help) echo "Usage: $0 [--service] [--service-user user]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

if [[ $EUID -ne 0 ]]; then
  echo "This installer must be run as root (sudo)" >&2
  exit 1
fi

DEST_DIR="/opt/openahi"
SHIM_DIR="/usr/local/bin"
SERVICE_UNIT="/etc/systemd/system/openahi-server.service"

echo "Installing OpenAHI into $DEST_DIR"
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

# Copy repo contents into /opt/openahi
rsync -a --delete --exclude '.git' --exclude '.github' --exclude 'dist' --exclude '__pycache__' --exclude 'venv' "$REPO_ROOT/" "$DEST_DIR/"

# Create shims
cat > "$SHIM_DIR/pkg" <<'SHIM'
#!/usr/bin/env bash
PYTHON=/usr/bin/python3
exec "$PYTHON" -m openahi.pkg_cli "$@"
SHIM
chmod 0755 "$SHIM_DIR/pkg"

cat > "$SHIM_DIR/openahi" <<'SHIM'
#!/usr/bin/env bash
PYTHON=/usr/bin/python3
export PYTHONPATH="/opt/openahi:$PYTHONPATH"
exec "$PYTHON" -m openahi.cli_full "$@"
SHIM
chmod 0755 "$SHIM_DIR/openahi"

# Create model directory
mkdir -p /var/lib/openahi/models
chown -R root:root /var/lib/openahi

if $ENABLE_SERVICE; then
  echo "Installing systemd service: $SERVICE_UNIT (user: $SERVICE_USER)"
  # Create service user if not exists
  if ! id "$SERVICE_USER" >/dev/null 2>&1; then
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER" || true
  fi

  cat > "$SERVICE_UNIT" <<UNIT
[Unit]
Description=OpenAHI Local Model Server
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
ExecStart=/usr/local/bin/openahi serve --port 8443 --directory /var/lib/openahi/models --tls --generate-self-signed --username openahi --password changeme
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable openahi-server.service || true
  echo "Service installed (disabled) at $SERVICE_UNIT. Start with: systemctl start openahi-server.service"
fi

echo "Installation complete. Shims installed: $SHIM_DIR/pkg and $SHIM_DIR/openahi"

