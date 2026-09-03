"""Quick training script for Composter model — produces a small demo checkpoint.

Usage:
  python examples/quick_train.py --epochs 3 --batch_size 8 --save-path ./checkpoints/composter.pt

This trains a tiny model on a tiny generated dataset so you can exercise install/run flows without large GPU requirements.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import random
import torch
from openahi.config import DEFAULT_CONFIG
from openahi.models.composter import ComposterModel
from openahi.tokenizer.tokenizer import SimpleTokenizer


def build_tiny_dataset(tokenizer: SimpleTokenizer, num_samples: int = 200, seq_len: int = 32):
    samples = []
    alphabet = list("abcdefghijklmnopqrstuvwxyz .,!\n")
    for _ in range(num_samples):
        l = random.randint(5, seq_len - 2)
        text = "".join(random.choice(alphabet) for _ in range(l))
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        # pad or trim
        if len(ids) < seq_len:
            ids = ids + [tokenizer.vocab["<pad>"]] * (seq_len - len(ids))
        else:
            ids = ids[:seq_len]
        samples.append(ids)
    return samples


def collate_batch(batch):
    import torch
    return torch.tensor(batch, dtype=torch.long)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_path', type=str, default='checkpoints/composter.pt')
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args(argv)

    tokenizer = SimpleTokenizer()
    cfg = DEFAULT_CONFIG
    # ensure tokenizer vocab fits model
    cfg.vocab_size = tokenizer.vocab_size

    model = ComposterModel(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    samples = build_tiny_dataset(tokenizer, num_samples=500, seq_len=cfg.block_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)

    for epoch in range(args.epochs):
        random.shuffle(samples)
        total_loss = 0.0
        count = 0
        for i in range(0, len(samples), args.batch_size):
            batch = samples[i:i+args.batch_size]
            xb = collate_batch(batch).to(device)
            # inputs and targets: next token prediction
            inputs = xb[:, :-1]
            targets = xb[:, 1:]
            logits, loss = model(inputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            count += 1
        print(f"Epoch {epoch+1}/{args.epochs} — avg loss: {total_loss / max(1,count):.4f}")

    # Save checkpoint (cpu)
    model.to('cpu')
    from pathlib import Path
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'config': cfg,
        'model_state_dict': model.state_dict(),
    }
    torch.save(checkpoint, args.save_path)
    print(f"Saved demo checkpoint to {args.save_path}")


if __name__ == '__main__':
    main()
