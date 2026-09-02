# Compatibility notes and advanced install instructions

This document collects platform-specific tips, especially for mobile (Termux) and low-resource environments.

1) Model storage location

The CLI stores models in a per-user data directory. The default locations, in order of precedence, are:
- $OPENAHI_MODEL_DIR (if set)
- $XDG_DATA_HOME/openahi/models
- ~/.openahi/models

If you want to change where models are stored, set the environment variable OPENAHI_MODEL_DIR to a writable path. Example:

  export OPENAHI_MODEL_DIR=/data/data/com.termux/files/home/storage/shared/openahi_models
  openahi install composter@1.00.0

2) Termux tips

- Termux's default $HOME is typically /data/data/com.termux/files/home; this will work out-of-the-box for model storage.
- Prefers using proot-distro for a more complete Linux environment if PyTorch installation fails.
- If you run into permission issues when creating ~/.openahi/models, ensure Termux has storage permissions and the target path is writable.

3) Running CLI without installing console scripts

On some systems (Termux, restricted PATH), console scripts may not be on PATH after `pip install .`. Use the module form instead:

  python -m openahi.cli models
  python -m openahi.cli install composter@1.00.0

or use the package module entrypoint:

  python -m openahi --help

4) Using the package on Windows

The package is cross-platform. For Windows, `python -m pip install .` will register console scripts to the Python Scripts directory. Ensure that directory is on PATH (e.g., C:\Users\<user>\AppData\Roaming\Python\Python39\Scripts).

5) Troubleshooting PyTorch

If `pip install torch` fails on Arm/Termux, consider one of:
- Use proot-distro and install Debian/Ubuntu, then install the official CPU wheel
- Use a remote server or desktop to do training/inference
- Use a lighter backend (not implemented here) — e.g., tiny numpy-based model for environments without PyTorch

6) Mobile-friendly invocation examples (Termux)

Install openahi locally (from repo):

  cd OpenAHi
  python -m pip install .

Install model using module invocation:

  python -m openahi.cli install composter@1.00.0

Generate text using module invocation:

  python -m openahi.inference.generate --checkpoint ~/.openahi/models/composter@1.00.0/composter.pt --prompt "Hello"


7) Security and sandboxing

Be mindful of executing untrusted model checkpoints. The checkpoint loader currently uses `torch.load`, which may execute arbitrary code if the file is malicious. Only load checkpoints from trusted sources.


If you'd like, I can:
- Add a lightweight numpy-only fallback model for environments where PyTorch is not available.
- Add automatic detection and guidance when running on Termux (print helpful hints when torch import fails).
