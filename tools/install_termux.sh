#!/usr/bin/env bash
set -euo pipefail

# tools/install_termux.sh
# Termux-specific installer to make `pkg install openahi` and `openahi` usable without a repo clone.
# - Installs Termux packages (python, openssl, git) if missing
# - Installs OpenAHI into the user site via pip
# - Creates shims in $PREFIX/bin so commands are available system-wide inside Termux
# - Sets up a default mirror directory in shared storage (requests storage permission)

PREFIX=${PREFIX:-/data/data/com.termux/files/usr}
USER_BIN="$PREFIX/bin"
USER_HOME="$HOME"
SHIM_PKG="$USER_BIN/pkg"
SHIM_OPENAHI="$USER_BIN/openahi"
GIT_URL="https://github.com/zoDevLang/OpenAHi.git"

echo "Termux installer for OpenAHI"

# Basic checks
if ! command -v pkg >/dev/null 2>&1; then
  echo "Warning: 'pkg' command not found. Are you in Termux?"
fi

echo "Updating Termux packages..."
pkg update -y || true

# Ensure required packages
echo "Installing python, git, openssl (if missing)..."
pkg install -y python git openssl

# Ensure pip is up-to-date
python -m pip install --upgrade pip setuptools wheel --user

# Attempt to install OpenAHI into user site-packages from GitHub
echo "Installing openahi into user site-packages (pip --user)..."
python -m pip install --user --upgrade "git+${GIT_URL}"

# Create shims in $PREFIX/bin so commands are available in Termux PATH
mkdir -p "$USER_BIN"

# Create pkg shim
cat > "$SHIM_PKG" <<'SHIM'
#!/usr/bin/env bash
# Termux shim for openahi.pkg_cli
python -m openahi.pkg_cli "$@"
SHIM
chmod +x "$SHIM_PKG"
echo "Created shim: $SHIM_PKG"

# Create openahi shim
cat > "$SHIM_OPENAHI" <<'SHIM'
#!/usr/bin/env bash
# Termux shim for openahi CLI
python -m openahi.cli_full "$@"
SHIM
chmod +x "$SHIM_OPENAHI"
echo "Created shim: $SHIM_OPENAHI"

# Setup shared storage mirror directory
MIRROR_DEFAULT="$HOME/storage/shared/openahi_models"
# Request storage permission if not present
if [[ ! -d "$HOME/storage" ]]; then
  echo "Setting up Termux storage access (termux-setup-storage). You will be prompted to grant permission."
  termux-setup-storage || true
fi
if [[ -d "$HOME/storage/shared" ]]; then
  mkdir -p "$MIRROR_DEFAULT"
  echo "Created default mirror directory: $MIRROR_DEFAULT"
else
  ALT_MIRROR="$HOME/openahi_mirror"
  mkdir -p "$ALT_MIRROR"
  echo "Shared storage unavailable; created mirror dir: $ALT_MIRROR"
  MIRROR_DEFAULT="$ALT_MIRROR"
fi

cat <<EOF
Termux installation complete.

Shims created:
  $SHIM_PKG
  $SHIM_OPENAHI

Default mirror directory: $MIRROR_DEFAULT

Make sure your PATH contains $USER_BIN (Termux normally does). If running a different shell, add:
  export PATH="$USER_BIN:\$PATH"

You can now run:
  pkg install openahi    # runs the pkg helper (installs deps & bootstraps model)
  openahi --help         # openahi CLI
  openahi mirror ~/openahi_mirror      # create mirror of installed models
  openahi serve --directory ~/openahi_mirror --port 8000  # serve mirrored models

If PyTorch install fails on Termux ("no matching distribution"), try using proot-distro with Debian and run the same installer inside that chroot:
  pkg install proot-distro
  proot-distro install debian
  proot-distro login debian

If you need help, paste the output of 'pkg install openahi' here.
EOF
