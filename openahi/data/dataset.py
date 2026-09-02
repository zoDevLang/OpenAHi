"""Dataset utilities for language modeling"""
from __future__ import annotations
from typing import List
import random
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, tokens: List[int], block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.tokens[idx: idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1: idx + 1 + self.block_size], dtype=torch.long)
        return x, y


def load_dataset_from_file(path: str, tokenizer, block_size: int):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    # repeat or trim
    return TextDataset(ids, block_size)
