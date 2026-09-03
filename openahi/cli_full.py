"""Comprehensive OpenAHI CLI implementing the requested command specification.

Commands implemented:
- openahi help
- openahi version
- openahi info
- openahi models
- openahi model info <model>
- openahi install <model> | <model>@<version> | news
- openahi check-release
- openahi update <model>
- openahi remove <model>
- openahi run <model>[@<version>] (with generation flags)
- openahi train <model> (small local training on tiny dataset)
- openahi evaluate <model> (evaluate on tiny dataset)
- openahi config
- openahi cache
- openahi doctor

Notes:
- Commands are conservative: they perform only supported operations, and print clear messages when functionality is limited.
- For remote installs, the installer uses artifact metadata in artifacts/ to determine how to proceed.
"""
from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
import shutil
import json
import textwrap
import platform
import tempfile

from openahi import __version__
from openahi import Composter
from openahi import tokenizer as _tokenizer_module
from openahi.config import DEFAULT_CONFIG
from openahi.training.trainer import Trainer
from openahi.data.dataset import load_dataset_from_file
from openahi.cli import list_models as _list_models, install_model as _install_model, uninstall_model as _uninstall_model, check_release as _check_release

# Re-export model storage location used by openahi.cli
xdg = os.environ.get("XDG_DATA_HOME")
home = Path.home()
_model_dir_env = os.environ.get("OPENAHI_MODEL_DIR")
if _model_dir_env:
    MODEL_DIR = Path(_model_dir_env)
elif xdg:
    MODEL_DIR = Path(xdg) / "openahi" / "models"
else:
    MODEL_DIR = home / ".openahi" / "models"


def cmd_version(args):
    print(f"OpenAHI version: {__version__}")


def cmd_info(args):
    print("OpenAHI — Composter")
    print(f"Version: {__version__}")
    print(f"Model storage: {MODEL_DIR}")
    print(f"Python: {platform.python_version()} ({platform.system()}/{platform.machine()})")


def cmd_models(args):
    _list_models()


def cmd_model_info(args):
    name = args.name
    # find installed versions
    found = []
    if MODEL_DIR.exists():
        for p in MODEL_DIR.iterdir():
            if p.name.startswith(name + "@") or p.name == name:
                found.append(p)
    if not found:
        print(f"No installed model named '{name}' found in {MODEL_DIR}")
        return
    for p in found:
        print(f"Model: {p.name}")
        art = p / 'artifact.json'
        ckpt = None
        for candidate in p.glob('*.pt'):
            ckpt = candidate
            break
        if ckpt and ckpt.exists():
            print(f"  checkpoint: {ckpt}")
        if art.exists():
            try:
                data = json.loads(art.read_text())
                print("  artifact metadata:")
                print(textwrap.indent(json.dumps(data, indent=2), '    '))
            except Exception:
                print("  artifact.json exists but could not be parsed")
        else:
            print("  no artifact metadata found (local/demo install)")


def cmd_install(args):
    spec = args.spec
    no_remote = args.no_remote
    force = args.force
    insecure = args.insecure
    print(f"Installing: {spec} (no_remote={no_remote}, force={force})")
    _install_model(spec, force=force, prefer_remote=not no_remote, insecure=insecure)


def cmd_check_release(args):
    _check_release(args.name)


def cmd_update(args):
    name = args.name
    print(f"Updating model {name} to latest (if available)")
    # install latest with force
    _install_model(f"{name}@latest", force=True, prefer_remote=True, insecure=False)


def cmd_remove(args):
    name = args.name
    print(f"Removing model {name}")
    _uninstall_model(f"{name}@latest")


def _find_installed_checkpoint(spec: str) -> Path | None:
    # spec may be 'composter' or 'composter@1.00.0'
    if '@' in spec:
        name, version = spec.split('@', 1)
    else:
        name = spec
        version = None
    candidates = []
    if MODEL_DIR.exists():
        for p in MODEL_DIR.iterdir():
            if p.is_dir() and p.name.startswith(name + '@'):
                if version is None or p.name == f"{name}@{version}":
                    for pt in p.glob('*.pt'):
                        candidates.append(pt)
    if not candidates:
        return None
    # prefer exact version match
    if version is not None:
        for c in candidates:
            if f"@{version}" in str(c.parent.name):
                return c
    return candidates[0]


def cmd_run(args):
    spec = args.model
    ckpt = _find_installed_checkpoint(spec)
    if ckpt is None or not ckpt.exists():
        print(f"Model checkpoint for '{spec}' not found. Install with: openahi install {spec}")
        return
    print(f"Loading model checkpoint: {ckpt}")
    # load via Composter API
    try:
        model = Composter.from_checkpoint(str(ckpt))
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    prompt = args.prompt
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    top_p = args.top_p if args.top_p and args.top_p > 0 else None
    deterministic = args.deterministic
    if deterministic:
        temperature = 0.0
    print("Generating...")
    try:
        out = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        print(out)
    except Exception as e:
        print(f"Generation failed: {e}")


def cmd_train(args):
    name = args.model
    # Only support training for composter on local tiny dataset for now
    if name.lower() not in ("composter", "composter@1.00.0", "composter@latest"):
        print("Training only supported for 'composter' in this prototype. Use 'openahi train composter'")
        return
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    print(f"Starting quick local training: epochs={epochs}, batch_size={batch_size}, lr={lr}")
    # load tokenizer and dataset
    tok = _tokenizer_module.SimpleTokenizer()
    ds = load_dataset_from_file(str(Path('data') / 'tiny.txt'), tok, DEFAULT_CONFIG.block_size)
    model = ComposterModel = None
    # instantiate model
    from openahi.models.composter import ComposterModel
    model = ComposterModel(DEFAULT_CONFIG)
    trainer = Trainer(model, tok, DEFAULT_CONFIG)
    ckpt_path = args.save_path or 'checkpoints/composter.pt'
    trainer.train(ds, batch_size=batch_size, epochs=epochs, lr=lr, save_path=ckpt_path)
    print(f"Training finished. Checkpoint saved to {ckpt_path}")


def cmd_evaluate(args):
    name = args.model
    ckpt = _find_installed_checkpoint(name)
    if ckpt is None:
        print(f"No installed checkpoint for '{name}' found. Install first: openahi install {name}")
        return
    print(f"Loading checkpoint {ckpt} for evaluation")
    # load tokenizer and dataset
    tok = _tokenizer_module.SimpleTokenizer()
    ds = load_dataset_from_file(str(Path('data') / 'tiny.txt'), tok, DEFAULT_CONFIG.block_size)
    from openahi.models.composter import ComposterModel
    data = torch_load_safe(str(ckpt))
    config = data.get('config', DEFAULT_CONFIG)
    model = ComposterModel(config)
    model.load_state_dict(data['model_state_dict'])
    trainer = Trainer(model, tok, DEFAULT_CONFIG)
    # build DataLoader
    from torch.utils.data import DataLoader
    val_loader = DataLoader(ds, batch_size=2)
    val_loss = trainer.evaluate(val_loader)
    print(f"Validation loss (toy dataset) = {val_loss:.4f}")


def torch_load_safe(path: str):
    import torch
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load torch checkpoint: {e}")
        raise


def cmd_config(args):
    print("Default model config:")
    print(DEFAULT_CONFIG)
    print(f"Model storage directory: {MODEL_DIR}")


def cmd_cache(args):
    if not MODEL_DIR.exists():
        print("No models installed.")
        return
    total = 0
    for p in MODEL_DIR.rglob('*'):
        try:
            total += p.stat().st_size
        except Exception:
            pass
    print(f"Models directory: {MODEL_DIR}")
    print(f"Total size: {total} bytes")
    print("Installed models:")
    _list_models()


def cmd_doctor(args):
    print("Running basic system checks...")
    import importlib
    ok = True
    print(f"Python: {platform.python_version()} ({sys.executable})")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except Exception as e:
        print("PyTorch: NOT INSTALLED or failed to import:", e)
        ok = False
    print(f"Model storage writable: {os.access(MODEL_DIR, os.W_OK) or not MODEL_DIR.exists()}")
    print(f"Disk free at home: {shutil.disk_usage(home).free} bytes")
    if ok:
        print("Doctor checks passed (basic)")
    else:
        print("Doctor found issues. Install PyTorch and ensure model storage is writable.")


def main(argv=None):
    parser = argparse.ArgumentParser(prog='openahi', description='OpenAHI CLI')
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('help')
    sub.add_parser('version')
    sub.add_parser('info')

    sub.add_parser('models')
    p_model_info = sub.add_parser('model')
    p_model_info_sub = p_model_info.add_subparsers(dest='model_cmd')
    p_model_info_show = p_model_info_sub.add_parser('info')
    p_model_info_show.add_argument('name')

    p_install = sub.add_parser('install')
    p_install.add_argument('spec')
    p_install.add_argument('--force', action='store_true')
    p_install.add_argument('--no-remote', action='store_true')
    p_install.add_argument('--insecure', action='store_true')

    sub.add_parser('check-release')

    p_update = sub.add_parser('update')
    p_update.add_argument('name')

    p_remove = sub.add_parser('remove')
    p_remove.add_argument('name')

    p_run = sub.add_parser('run')
    p_run.add_argument('model')
    p_run.add_argument('--prompt', type=str, default='Hello')
    p_run.add_argument('--max_new_tokens', type=int, default=50)
    p_run.add_argument('--temperature', type=float, default=1.0)
    p_run.add_argument('--top_k', type=int, default=0)
    p_run.add_argument('--top_p', type=float, default=0.0)
    p_run.add_argument('--deterministic', action='store_true')

    p_train = sub.add_parser('train')
    p_train.add_argument('model')
    p_train.add_argument('--epochs', type=int, default=1)
    p_train.add_argument('--batch_size', type=int, default=2)
    p_train.add_argument('--lr', type=float, default=1e-3)
    p_train.add_argument('--save_path', type=str, default='checkpoints/composter.pt')

    p_eval = sub.add_parser('evaluate')
    p_eval.add_argument('model')

    sub.add_parser('config')
    sub.add_parser('cache')
    sub.add_parser('doctor')

    args = parser.parse_args(argv)
    if args.cmd in (None, 'help'):
        parser.print_help()
        return
    if args.cmd == 'version':
        cmd_version(args)
    elif args.cmd == 'info':
        cmd_info(args)
    elif args.cmd == 'models':
        cmd_models(args)
    elif args.cmd == 'model' and getattr(args, 'model_cmd', None) == 'info':
        cmd_model_info(args)
    elif args.cmd == 'install':
        cmd_install(args)
    elif args.cmd == 'check-release':
        cmd_check_release(args)
    elif args.cmd == 'update':
        cmd_update(args)
    elif args.cmd == 'remove':
        cmd_remove(args)
    elif args.cmd == 'run':
        cmd_run(args)
    elif args.cmd == 'train':
        cmd_train(args)
    elif args.cmd == 'evaluate':
        cmd_evaluate(args)
    elif args.cmd == 'config':
        cmd_config(args)
    elif args.cmd == 'cache':
        cmd_cache(args)
    elif args.cmd == 'doctor':
        cmd_doctor(args)
    else:
        print(f"Unknown command: {args.cmd}")
        parser.print_help()


if __name__ == '__main__':
    main()
