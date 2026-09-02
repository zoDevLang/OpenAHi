"""Training utilities and trainer loop"""
from __future__ import annotations
from typing import Optional
import os
import math
import random
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from openahi.config import ModelConfig
from openahi.models.composter import ComposterModel


class Trainer:
    def __init__(self, model: ComposterModel, tokenizer, config: ModelConfig, device: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, dataset, batch_size: int = 16, epochs: int = 1, lr: float = 1e-4, save_path: Optional[str] = None):
        # train/val split
        n = len(dataset)
        val_size = max(1, n // 10)
        train_size = n - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=lr)

        best_val = float('inf')
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            count = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits, loss = self.model(xb, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
            avg_loss = total_loss / max(1, count)
            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}/{epochs} train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_val and save_path is not None:
                best_val = val_loss
                self.save_checkpoint(save_path)

    def evaluate(self, loader):
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                _, loss = self.model(xb, yb)
                total += loss.item()
                count += 1
        return total / max(1, count)

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, tokenizer):
        data = torch.load(path, map_location='cpu')
        config = data['config']
        model = ComposterModel(config)
        model.load_state_dict(data['model_state_dict'])
        return model
