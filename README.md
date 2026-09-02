# OpenAHI - Composter model (minimal)

"""
OpenAHI - Composter 1.00.0

This package provides a tiny Transformer-based language model that runs locally using PyTorch.

Creator: ZoDev
License: MIT (see LICENSE file)

Usage (after installing requirements):

python -m openahi.generate --prompt "Hello"

or in Python:

from openahi import Composter
model = Composter.from_checkpoint("checkpoints/composter.pt")
print(model.generate("Hello", max_new_tokens=20))

"""
