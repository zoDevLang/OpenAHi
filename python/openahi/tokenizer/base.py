"""
Base Tokenizer Classes

Provides the fundamental tokenizer interface and vocabulary management.
"""

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class Vocabulary:
    """
    Vocabulary for tokenizer.
    
    Maps tokens to IDs and vice versa.
    Supports special tokens (PAD, BOS, EOS, UNK, etc.)
    """
    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)
    special_tokens: Dict[str, int] = field(default_factory=dict)
    vocab_size: int = 0
    
    def __post_init__(self):
        if not self.token_to_id:
            # Initialize with common special tokens
            self.pad_token = "[PAD]"
            self.bos_token = "[BOS]"
            self.eos_token = "[EOS]"
            self.unk_token = "[UNK]"
            
            self.token_to_id = {
                self.pad_token: 0,
                self.bos_token: 1,
                self.eos_token: 2,
                self.unk_token: 3,
            }
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            self.special_tokens = {
                "pad": 0,
                "bos": 1,
                "eos": 2,
                "unk": 3,
            }
            self.vocab_size = 4
    
    def add_token(self, token: str) -> int:
        """Add a token to the vocabulary. Returns the assigned ID."""
        if token in self.token_to_id:
            return self.token_to_id[token]
        
        token_id = self.vocab_size
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self.vocab_size += 1
        return token_id
    
    def add_special_token(self, token: str, token_id: Optional[int] = None) -> int:
        """Add a special token to the vocabulary."""
        if token_id is None:
            token_id = self.vocab_size
        
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self.special_tokens[token.lower()] = token_id
        
        if token_id >= self.vocab_size:
            self.vocab_size = token_id + 1
        
        return token_id
    
    def get_id(self, token: str) -> int:
        """Get ID for a token. Returns UNK token ID if not found."""
        return self.token_to_id.get(token, self.special_tokens.get("unk", 3))
    
    def get_token(self, token_id: int) -> str:
        """Get token for an ID. Returns UNK token if not found."""
        return self.id_to_token.get(token_id, self.id_to_token.get(self.special_tokens.get("unk", 3), "[UNK]"))
    
    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id
    
    def __len__(self) -> int:
        return self.vocab_size
    
    def save(self, filepath: str):
        """Save vocabulary to file."""
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Load vocabulary from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls()
        vocab.token_to_id = data["token_to_id"]
        vocab.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        vocab.special_tokens = {k: int(v) for k, v in data["special_tokens"].items()}
        vocab.vocab_size = data["vocab_size"]
        return vocab
    
    @classmethod
    def from_tokens(cls, tokens: List[str], special_tokens: Optional[Dict[str, str]] = None) -> "Vocabulary":
        """Create vocabulary from a list of tokens."""
        vocab = cls()
        
        # Clear default special tokens if custom ones provided
        if special_tokens:
            vocab.token_to_id.clear()
            vocab.id_to_token.clear()
            vocab.special_tokens.clear()
            vocab.vocab_size = 0
            
            for token_name, token_str in special_tokens.items():
                vocab.add_special_token(token_str, None)
        
        # Add all tokens
        for token in tokens:
            vocab.add_token(token)
        
        return vocab


class Tokenizer(ABC):
    """
    Abstract base class for tokenizers.
    
    Provides the interface that all OpenAHI tokenizers must implement.
    """
    
    def __init__(self, vocab: Optional[Vocabulary] = None):
        if vocab is None:
            vocab = Vocabulary()
        self.vocab = vocab
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs into text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text string
        """
        pass
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_id_batches: List[List[int]]) -> List[str]:
        """Decode a batch of token IDs."""
        return [self.decode(tids) for tids in token_id_batches]
    
    def text_to_tokens(self, text: str) -> List[str]:
        """Convert text to list of token strings."""
        token_ids = self.encode(text)
        return [self.vocab.get_token(tid) for tid in token_ids]
    
    def tokens_to_text(self, tokens: List[str]) -> str:
        """Convert list of token strings to text."""
        token_ids = [self.vocab.get_id(token) for token in tokens]
        return self.decode(token_ids)
    
    def save(self, filepath: str):
        """Save tokenizer to file."""
        # Save vocabulary
        vocab_path = filepath + ".vocab.json"
        self.vocab.save(vocab_path)
        
        # Save tokenizer-specific data
        self._save(filepath)
    
    @abstractmethod
    def _save(self, filepath: str):
        """Save tokenizer-specific data."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> "Tokenizer":
        """Load tokenizer from file."""
        pass
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def get_special_token_id(self, token_name: str) -> int:
        """Get ID for a special token."""
        return self.vocab.special_tokens.get(token_name, -1)
    
    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self.get_special_token_id("pad")
    
    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        return self.get_special_token_id("bos")
    
    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self.get_special_token_id("eos")
    
    @property
    def unk_token_id(self) -> int:
        """Unknown token ID."""
        return self.get_special_token_id("unk")
