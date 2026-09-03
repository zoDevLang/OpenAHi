#!/usr/bin/env bash
set -euo pipefail

# tools/install_user_pkg.sh
# User-level installer for OpenAHI (no clone required).
# Installs the package from GitHub into the user's site-packages, installs common dependencies
# (including attempting to install PyTorch CPU wheels), and creates a ~/bin/pkg shim so
# you can run "pkg install openahi" and get the full bootstrap behavior.
#
# Usage:
#   bash tools/install_user_pkg.sh        # run from repo (optional)
#   curl -fsSL https://raw.githubusercontent.com/zoDevLang/OpenAHi/main/tools/install_user_pkg.sh | bash -s --
# Options:
#   --no-deps      Skip dependency installation (only installs the openahi package)
#   --force        Reinstall everything
#   -h, --help     Show this help

REPO_RAW_BASE="https://raw.githubusercontent.com/zoDevLang/OpenAHi/main"
GIT_URL="https://github.com/zoDevLang/OpenAHi.git"

NO_DEPS=false
FORCE=false

usage(){
  cat <<EOF
User-level installer for OpenAHI (no clone needed)

Usage:
  curl -fsSL ${REPO_RAW_BASE}/tools/install_user_pkg.sh | bash -s -- [--no-deps] [--force]

Options:
  --no-deps      Skip installing runtime dependencies (useful if you manage deps yourself)
  --force        Force reinstall of package and dependencies
  -h, --help     Show this help

This script will:
  - Ensure python3 and pip are available
  - Install OpenAHI into the user site-packages via pip (git+https://...)
  - Attempt to install runtime deps from requirements.txt (unless --no-deps)
  - Try to install PyTorch (CPU wheels) if torch is not importable
  - Create a user shim at ~/bin/pkg so you can run `pkg install openahi` as a real command
  - Print post-install instructions to ensure ~/bin is on your PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-deps) NO_DEPS=true; shift;;
    --force) FORCE=true; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# Check python availability
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 command not found. Please install Python 3 and rerun." >&2
  exit 1
fi
PY=python3

# Upgrade pip etc in user env
echo "Upgrading pip/setuptools/wheel in user site..."
$PY -m pip install --user --upgrade pip setuptools wheel

# Install OpenAHI from GitHub into user site-packages
if $FORCE; then
  echo "Force reinstalling OpenAHI from $GIT_URL"
  $PY -m pip install --user --upgrade --force-reinstall "git+${GIT_URL}"
else
  echo "Installing OpenAHI from $GIT_URL (user site)..."
  $PY -m pip install --user --upgrade "git+${GIT_URL}"
fi

# Setup user shim for pkg if not exists or if force
USER_BIN_DIR="$HOME/bin"
SHIM_PATH="$USER_BIN_DIR/pkg"
mkdir -p "$USER_BIN_DIR"
if [[ -f "$SHIM_PATH" && "$FORCE" != true ]]; then
  echo "User shim already exists at $SHIM_PATH"
else
  cat > "$SHIM_PATH" <<'SHIM'
#!/usr/bin/env bash
python3 -m openahi.pkg_cli "$@"
SHIM
  chmod +x "$SHIM_PATH"
  echo "Created user shim: $SHIM_PATH"
fi

# Ensure user bin is on PATH for this shell session
if [[ ":$PATH:" != *":$USER_BIN_DIR:"* ]]; then
  echo "Adding $USER_BIN_DIR to PATH for current session"
  export PATH="$USER_BIN_DIR:$PATH"
fi

# Install runtime dependencies if requested
if ! $NO_DEPS; then
  echo "Attempting to install runtime requirements from repository (requirements.txt) via raw URL..."
  REQ_URL="$REPO_RAW_BASE/requirements.txt"
  # We attempt to pip install -r from the raw URL (pip supports http(s) URLs)
  set +e
  $PY -m pip install --user -r "$REQ_URL"
  RC=$?
  set -e
  if [[ $RC -ne 0 ]]; then
    echo "Installing requirements via requirements.txt failed (some packages may require manual installation)" >&2
  else
    echo "Installed requirements from requirements.txt"
  fi
else
  echo "Skipping dependency installation (--no-deps)"
fi

# Ensure torch is available (import test). If not, try CPU wheel install strategies.
echo "Checking for PyTorch (torch) import..."
set +e
$PY - <<PY
try:
    import torch
    print('OK')
except Exception as e:
    print('MISSING')
PY
TORCH_OK=$?
set -e

if [[ $TORCH_OK -ne 0 ]]; then
  echo "PyTorch not importable. Attempting to install CPU-only wheel via PyTorch index..."
  set +e
  # Try generic pip install first
  $PY -m pip install --user --upgrade torch
  RC1=$?
  if [[ $RC1 -ne 0 ]]; then
    echo "Generic pip install torch failed; trying CPU-only wheels index..."
    $PY -m pip install --user --upgrade torch --index-url https://download.pytorch.org/whl/cpu
    RC2=$?
  else
    RC2=0
  fi
  set -e
  if [[ ${RC1:-1} -ne 0 && ${RC2:-1} -ne 0 ]]; then
    echo "Automatic PyTorch installation failed. On some platforms (Android/Termux/ARM) official wheels may not be available."
    echo "You may need to install PyTorch manually for your platform. See https://pytorch.org for instructions."
  else
    echo "PyTorch installed (or already present)."
  fi
else
  echo "PyTorch already installed and importable."
fi

# Final messages
echo "\nInstallation finished."
cat <<EOF
Now make sure your user bin directory is on PATH permanently (add to ~/.profile or ~/.bashrc):

  mkdir -p \"$HOME/bin\"
  echo 'export PATH="$HOME/bin:\$PATH"' >> "$HOME/.profile"
  # then either re-login or run:
  source "$HOME/.profile"

After this, you can run the bootstrap command:

  pkg install openahi

which will install the openahi package and (by default) auto-run the model installer.

If you encounter any errors (especially related to PyTorch installation on your platform), paste the error output here and I will help.
EOF
