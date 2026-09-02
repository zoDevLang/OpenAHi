"""
Dataset Classes

Provides dataset abstractions for training OpenAHI models.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator
import numpy as np
import torch


@dataclass
class DatasetConfig:
    """Configuration for dataset."""
    name: str = "default"
    train_split: float = 0.9
    val_split: float = 0.1
    test_split: float = 0.0
    seed: int = 42
    max_length: int = 512
    batch_size: int = 32
    shuffle: bool = True


class Dataset(ABC):
    """
    Abstract base class for datasets.
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self._data: List[Dict] = []
        self._train_data: Optional[List[Dict]] = None
        self._val_data: Optional[List[Dict]] = None
        self._test_data: Optional[List[Dict]] = None
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Load dataset from path."""
        pass
    
    def split(self, train_ratio: float = 0.9, val_ratio: float = 0.1) -> None:
        """
        Split dataset into train/val/test sets.
        
        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
        """
        if not self._data:
            raise ValueError("Dataset must be loaded before splitting")
        
        # Set random seed for reproducibility
        random.seed(self.config.seed)
        random.shuffle(self._data)
        
        total = len(self._data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        self._train_data = self._data[:train_end]
        self._val_data = self._data[train_end:val_end]
        self._test_data = self._data[val_end:]
    
    @property
    def train_data(self) -> List[Dict]:
        """Get training data."""
        if self._train_data is None:
            self.split()
        return self._train_data
    
    @property
    def val_data(self) -> List[Dict]:
        """Get validation data."""
        if self._val_data is None:
            self.split()
        return self._val_data
    
    @property
    def test_data(self) -> List[Dict]:
        """Get test data."""
        if self._test_data is None:
            self.split()
        return self._test_data
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self._data[idx]


class TextDataset(Dataset):
    """
    Dataset for text data.
    
    Stores raw text strings and provides tokenization on demand.
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        super().__init__(config)
        self._raw_texts: List[str] = []
    
    def load(self, path: Union[str, Path]) -> None:
        """Load text dataset from file."""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            self._raw_texts = [line.strip() for line in f if line.strip()]
        
        # Store as list of dicts
        self._data = [{"text": text} for text in self._raw_texts]
    
    def load_from_list(self, texts: List[str]) -> None:
        """Load text dataset from a list of strings."""
        self._raw_texts = texts
        self._data = [{"text": text} for text in texts]
    
    @property
    def texts(self) -> List[str]:
        """Get all raw texts."""
        return self._raw_texts
    
    def get_text(self, idx: int) -> str:
        """Get text at index."""
        return self._raw_texts[idx]


class TokenizedDataset(Dataset):
    """
    Dataset for pre-tokenized data.
    
    Stores token IDs and attention masks.
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        super().__init__(config)
        self._token_ids: List[List[int]] = []
        self._attention_masks: List[List[int]] = []
    
    def load(self, path: Union[str, Path]) -> None:
        """Load tokenized dataset from file."""
        # Implementation for loading pre-tokenized data
        # Typically this would be created from a TextDataset
        pass
    
    def from_text_dataset(self, text_dataset: TextDataset, tokenizer, max_length: Optional[int] = None) -> None:
        """
        Create tokenized dataset from text dataset.
        
        Args:
            text_dataset: Source text dataset
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        if max_length is None:
            max_length = self.config.max_length
        
        pad_token_id = tokenizer.pad_token_id
        
        for text in text_dataset.texts:
            # Tokenize
            token_ids = tokenizer.encode(text)
            
            # Truncate if necessary
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            # Pad
            padding_length = max_length - len(token_ids)
            if padding_length > 0:
                token_ids = token_ids + [pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            
            self._token_ids.append(token_ids)
            self._attention_masks.append(attention_mask)
        
        # Store as list of dicts
        self._data = [
            {"input_ids": tids, "attention_mask": mask}
            for tids, mask in zip(self._token_ids, self._attention_masks)
        ]
    
    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to PyTorch tensors."""
        input_ids = torch.tensor(self._token_ids, dtype=torch.long)
        attention_mask = torch.tensor(self._attention_masks, dtype=torch.long)
        return input_ids, attention_mask


def create_tiny_dataset() -> TextDataset:
    """
    Create a tiny example dataset for demonstration.
    
    This creates a small corpus of text that can be used to train
    and test the Composter model without external dependencies.
    """
    # Small set of example sentences
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "OpenAHI is an open source artificial intelligence project.",
        "Composter is the first model in the OpenAHI ecosystem.",
        "Machine learning models learn from data.",
        "Natural language processing helps computers understand text.",
        "Transformers are a type of neural network architecture.",
        "The model can generate text autoregressively.",
        "Training requires data and computation.",
        "Inference is the process of using a trained model.",
        "OpenAHI provides a complete model ecosystem.",
        "Developers can download install and run models.",
        "The project is created by ZoDev.",
        "Artificial hyper intelligence is the long term vision.",
        "Python is used for research and training.",
        "C++ provides high performance inference.",
        "Rust powers the runtime and CLI.",
        "TypeScript enables web applications.",
        "The architecture is modular and extensible.",
        "Composter 1.00.0 is a small transformer model.",
        "It includes attention feed forward and normalization layers.",
    ]
    
    # Duplicate and vary the texts to create more data
    more_texts = []
    for text in texts:
        # Add variations
        more_texts.append(text)
        more_texts.append(text.lower())
        more_texts.append(text.upper())
        
        # Add some modified versions
        words = text.split()
        if len(words) > 1:
            more_texts.append(' '.join(words[::-1]))  # Reversed
    
    # Add some simple patterns
    for i in range(100):
        more_texts.append(f"Example sentence number {i}.")
    
    for i in range(100):
        more_texts.append(f"Token {i} is a number.")
    
    dataset = TextDataset()
    dataset.load_from_list(texts + more_texts)
    return dataset
