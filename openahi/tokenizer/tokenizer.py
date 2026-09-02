"""OpenAHI tokenizer: simple byte-level tokenizer with basic specials"""
from __future__ import annotations
from typing import List, Dict

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

class SimpleTokenizer:
    def __init__(self):
        # byte-level: 0-255
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        # reserve tokens
        idx = 0
        for tok in SPECIAL_TOKENS:
            self.vocab[tok] = idx
            self.inv_vocab[idx] = tok
            idx += 1
        for b in range(256):
            s = chr(b)
            if s not in self.vocab:
                self.vocab[s] = idx
                self.inv_vocab[idx] = s
                idx += 1
        self.vocab_size = idx

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.vocab["<bos>"])
        for ch in text:
            ids.append(self.vocab.get(ch, self.vocab["<unk>"]))
        if add_eos:
            ids.append(self.vocab["<eos>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        chars = []
        for i in ids:
            tok = self.inv_vocab.get(i, "<unk>")
            if tok in ("<bos>", "<eos>", "<pad>"):
                continue
            # single-char tokens
            chars.append(tok)
        return "".join(chars)

    def save(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        import json
        t = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        t.vocab = data["vocab"]
        t.inv_vocab = {int(v): k for k, v in t.vocab.items()}
        t.vocab_size = max(t.inv_vocab.keys()) + 1
        return t
