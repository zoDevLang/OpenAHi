#!/usr/bin/env bash
# tools/publish_release.sh
# Usage:
# 1) To create a release and upload assets (requires gh CLI authenticated):
#    ./tools/publish_release.sh --version 1.00.0 --model ./composter-1.00.0.pt --package ./dist/openahi-0.1.0-py3-none-any.whl
# 2) To only update artifacts JSON given public URLs and checksums:
#    ./tools/publish_release.sh --version 1.00.0 --model-url <url> --model-sha <sha256> --package-url <url>

set -euo pipefail

usage() {
  cat <<EOF
publish_release.sh - create GitHub release and upload model + package assets, then update artifacts JSON

Options:
  --version VERSION       Release version (e.g. 1.00.0) (required)
  --model PATH            Path to model .pt file to upload
  --package PATH          Path to package wheel/sdist to upload (optional)
  --model-url URL         If you already uploaded the model, provide its public URL instead of --model
  --model-sha SHA256      SHA256 checksum for the model URL (required if --model-url used)
  --package-url URL       If you already uploaded package, provide public URL
  --notes FILE            Release notes file (optional)
  --push                  Commit and push updated artifacts JSON (requires git push permissions)
  -h, --help              Show this help

Environment:
  GH_TOKEN should be available or run 'gh auth login' before running.

Examples:
  ./tools/publish_release.sh --version 1.00.0 --model ./composter-1.00.0.pt --package ./dist/openahi-0.1.0-py3-none-any.whl --push
  ./tools/publish_release.sh --version 1.00.0 --model-url https://.../composter-1.00.0.pt --model-sha abc... --push
EOF
}

VERSION=""
MODEL_PATH=""
PACKAGE_PATH=""
MODEL_URL=""
MODEL_SHA=""
PACKAGE_URL=""
NOTES_FILE=""
PUSH=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2;;
    --model) MODEL_PATH="$2"; shift 2;;
    --package) PACKAGE_PATH="$2"; shift 2;;
    --model-url) MODEL_URL="$2"; shift 2;;
    --model-sha) MODEL_SHA="$2"; shift 2;;
    --package-url) PACKAGE_URL="$2"; shift 2;;
    --notes) NOTES_FILE="$2"; shift 2;;
    --push) PUSH=true; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$VERSION" ]]; then
  echo "--version is required"; usage; exit 1
fi

# repo info
REPO_FULL=$(git config --get remote.origin.url || echo "")
if [[ -z "$REPO_FULL" ]]; then
  echo "Could not determine git remote origin URL. Run this script from a git repo with origin set."; exit 1
fi
# normalize origin URL to owner/repo
# support git@github.com:owner/repo.git and https://github.com/owner/repo.git
if [[ "$REPO_FULL" =~ ^git@github.com:(.+)/(.+)\.git$ ]]; then
  OWNER=${BASH_REMATCH[1]}
  REPO=${BASH_REMATCH[2]}
elif [[ "$REPO_FULL" =~ ^https://github.com/(.+)/(.+)\.git$ ]]; then
  OWNER=${BASH_REMATCH[1]}
  REPO=${BASH_REMATCH[2]}
else
  # try to parse with sed
  OWNER=$(echo "$REPO_FULL" | sed -E 's#.*github.com[:/](.*)/(.*)\.git#\1#')
  REPO=$(echo "$REPO_FULL" | sed -E 's#.*github.com[:/](.*)/(.*)\.git#\2#')
fi
TAG="v${VERSION}"

# compute model URL and sha if model path provided
if [[ -n "$MODEL_PATH" ]]; then
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Model path not found: $MODEL_PATH"; exit 1
  fi
  echo "Computing SHA256 for $MODEL_PATH..."
  MODEL_SHA=$(sha256sum "$MODEL_PATH" | awk '{print $1}')
  MODEL_FILENAME=$(basename "$MODEL_PATH")
fi

if [[ -n "$PACKAGE_PATH" ]]; then
  if [[ ! -f "$PACKAGE_PATH" ]]; then
    echo "Package path not found: $PACKAGE_PATH"; exit 1
  fi
  PACKAGE_FILENAME=$(basename "$PACKAGE_PATH")
fi

# If MODEL_URL already provided, require MODEL_SHA
if [[ -n "$MODEL_URL" && -z "$MODEL_SHA" ]]; then
  echo "--model-sha is required when providing --model-url"; exit 1
fi

# Create release via gh CLI
if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install GitHub CLI (https://cli.github.com/) and run 'gh auth login' before running this script."; exit 1
fi

# Prepare notes
NOTES_ARG="-n 'OpenAHI release $VERSION'"
if [[ -n "$NOTES_FILE" ]]; then
  NOTES_ARG="--notes-file $NOTES_FILE"
fi

# Create or update release
echo "Creating Git tag and release $TAG..."
# create tag locally if not exists
if ! git rev-parse "$TAG" >/dev/null 2>&1; then
  git tag -a "$TAG" -m "Release $TAG"
  echo "Created git tag $TAG"
else
  echo "Git tag $TAG already exists"
fi

# create release (gh will update if exists)
if [[ -n "$MODEL_PATH" || -n "$PACKAGE_PATH" ]]; then
  # create release with assets (gh release create accepts files)
  ASSETS=()
  if [[ -n "$MODEL_PATH" ]]; then ASSETS+=("$MODEL_PATH"); fi
  if [[ -n "$PACKAGE_PATH" ]]; then ASSETS+=("$PACKAGE_PATH"); fi
  echo "Creating release with assets: ${ASSETS[*]}"
  gh release create "$TAG" "${ASSETS[@]}" --repo "$OWNER/$REPO" --title "OpenAHI $TAG" --notes "Release $TAG"
else
  echo "Creating release without assets"
  gh release create "$TAG" --repo "$OWNER/$REPO" --title "OpenAHI $TAG" --notes "Release $TAG"
fi

# Compute public URLs for uploaded assets if we uploaded them
if [[ -n "$MODEL_PATH" ]]; then
  MODEL_URL="https://github.com/${OWNER}/${REPO}/releases/download/${TAG}/${MODEL_FILENAME}"
  echo "Model uploaded (or expected) at: $MODEL_URL"
fi
if [[ -n "$PACKAGE_PATH" ]]; then
  PACKAGE_URL="https://github.com/${OWNER}/${REPO}/releases/download/${TAG}/${PACKAGE_FILENAME}"
  echo "Package uploaded (or expected) at: $PACKAGE_URL"
fi

# Update artifacts JSON for composter
ARTIFACTS_DIR="artifacts"
mkdir -p "$ARTIFACTS_DIR"
ARTFILE="$ARTIFACTS_DIR/composter-${VERSION}.json"
jq -n --arg name "composter" --arg version "$VERSION" --arg url "$MODEL_URL" --arg sha "$MODEL_SHA" --argjson config '{}' '{name:$name,version:$version,source:"github",url:$url,sha256:$sha,config:{},notes:"release asset"}' > "$ARTFILE"

# Also update latest alias
LATEST_FILE="$ARTIFACTS_DIR/composter-latest.json"
jq -n --arg name "composter" --arg version "$VERSION" --arg url "$MODEL_URL" --arg sha "$MODEL_SHA" --argjson config '{}' '{name:$name,version:"latest",source:"github",url:$url,sha256:$sha,config:$config,notes:"alias latest -> '$VERSION'"}' > "$LATEST_FILE"

echo "Updated artifacts: $ARTFILE and $LATEST_FILE"

if $PUSH; then
  git add "$ARTFILE" "$LATEST_FILE"
  git commit -m "Update artifacts for composter $VERSION"
  git push origin HEAD
  git push origin "$TAG" || true
  echo "Pushed artifacts and tag"
else
  echo "Run with --push to commit and push artifact JSON updates"
fi

echo "Done. Please verify the release assets on GitHub: https://github.com/${OWNER}/${REPO}/releases/tag/${TAG}"
