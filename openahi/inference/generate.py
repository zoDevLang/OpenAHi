"""CLI generation entrypoint"""
from __future__ import annotations
import argparse
import torch
from openahi.tokenizer.tokenizer import SimpleTokenizer
from openahi.training.trainer import Trainer
from openahi.models.composter import ComposterModel
from openahi.config import DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(prog="openahi-generate")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt text")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/composter.pt")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
    args = parser.parse_args()

    tokenizer = SimpleTokenizer()
    # load checkpoint if exists
    try:
        data = torch.load(args.checkpoint, map_location='cpu')
        config = data.get('config', DEFAULT_CONFIG)
        model = ComposterModel(config)
        model.load_state_dict(data['model_state_dict'])
    except Exception:
        print(f"Warning: could not load checkpoint {args.checkpoint}, using a fresh model")
        config = DEFAULT_CONFIG
        model = ComposterModel(config)
    model.eval()

    ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    import torch
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=(args.top_k or None), top_p=(args.top_p or None))
    text = tokenizer.decode(out[0].tolist())
    print(text)


if __name__ == '__main__':
    main()
