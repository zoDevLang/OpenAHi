Publish release instructions

This file explains two workflows to publish the model and package assets for an OpenAHI release.

Workflow A — You already uploaded assets (mix of opt 2 & 3):
1) Upload the model file (composter-<version>.pt) and the package wheel (optional) as release assets on GitHub **or** host them on any public URL.
2) Compute SHA256 of the model file (example):
   sha256sum composter-1.00.0.pt
3) Run the publish script to update artifacts JSON and (optionally) create a release:
   ./tools/publish_release.sh --version 1.00.0 --model-url https://yourhost/.../composter-1.00.0.pt --model-sha <sha256> --package-url https://yourhost/.../openahi-0.1.0.whl --push

Workflow B — Use the gh CLI to upload assets and update artifacts (recommended):
1) Install GitHub CLI (https://cli.github.com/) and authenticate:
   gh auth login
2) Ensure you have the model file and package file locally (or generate them):
   # build or copy files to repo root
3) Run the publisher script (it will create the tag and release, upload assets, compute sha256, and update artifacts JSON):
   ./tools/publish_release.sh --version 1.00.0 --model ./composter-1.00.0.pt --package ./dist/openahi-0.1.0-py3-none-any.whl --push

Notes:
- The script expects to run in the repository root and requires git remote origin to be set (owner/repo will be inferred).
- The script uses gh CLI to create the release and upload assets. It then writes artifacts/composter-<version>.json and the latest alias composter-latest.json and optionally commits and pushes them.
- If you cannot run gh CLI, you may upload assets manually via the GitHub web UI and then run the script with --model-url and --model-sha to update artifacts JSON.

Security:
- Keep your GH credentials secure. The script uses gh CLI authentication stored by gh.
- The installer in the repo will verify SHA256 before installing downloaded model binaries.
