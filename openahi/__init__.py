"""Package API and Composter wrapper"""
from __future__ import annotations
from typing import Optional
import torch
from openahi.models.composter import ComposterModel
from openahi.tokenizer.tokenizer import SimpleTokenizer
from openahi.config import DEFAULT_CONFIG


def _load_checkpoint(path: str):
    data = torch.load(path, map_location='cpu')
    config = data.get('config', DEFAULT_CONFIG)
    model = ComposterModel(config)
    model.load_state_dict(data['model_state_dict'])
    tokenizer = SimpleTokenizer()
    return model, tokenizer


class Composter:
    def __init__(self, model: ComposterModel, tokenizer: SimpleTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_checkpoint(cls, path: str):
        model, tokenizer = _load_checkpoint(path)
        return cls(model, tokenizer)

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None) -> str:
        ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        import torch
        x = torch.tensor([ids], dtype=torch.long)
        out = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        return self.tokenizer.decode(out[0].tolist())
