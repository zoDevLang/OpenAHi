"""
DataLoader for OpenAHI

Provides batching and iteration utilities for training.
"""

import random
from typing import Iterator, List, Optional, Tuple
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

from openahi.data.dataset import TokenizedDataset


class OpenAHIDataLoader:
    """
    DataLoader for OpenAHI training.
    
    Provides batched data for model training with support for
    shuffling, batching, and device placement.
    """
    
    def __init__(self, dataset: TokenizedDataset, batch_size: int = 32,
                 shuffle: bool = True, device: str = "cpu", seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.seed = seed
        self._indices: Optional[List[int]] = None
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def _get_indices(self) -> List[int]:
        """Get shuffled or sequential indices."""
        if self._indices is None or self.shuffle:
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.seed(self.seed)
                random.shuffle(indices)
            self._indices = indices
        return self._indices
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over batches.
        
        Yields:
            Tuple of (input_ids, attention_mask) tensors
        """
        indices = self._get_indices()
        
        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch = [self.dataset[i] for i in batch_indices]
            
            # Stack into tensors
            input_ids = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
            attention_mask = torch.stack([torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch])
            
            # Move to device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            yield input_ids, attention_mask
    
    def get_batch(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a specific batch by index."""
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.dataset))
        
        batch = [self.dataset[i] for i in range(start, end)]
        
        input_ids = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
        attention_mask = torch.stack([torch.tensor(b["attention_mask"], dtype=torch.long) for b in batch])
        
        return input_ids.to(self.device), attention_mask.to(self.device)


class TorchDatasetWrapper(TorchDataset):
    """
    Wrapper to make OpenAHI datasets compatible with PyTorch DataLoader.
    """
    
    def __init__(self, tokenized_dataset: TokenizedDataset):
        self.dataset = tokenized_dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[idx]
        return (
            torch.tensor(item["input_ids"], dtype=torch.long),
            torch.tensor(item["attention_mask"], dtype=torch.long)
        )


def create_pytorch_dataloader(tokenized_dataset: TokenizedDataset, batch_size: int = 32,
                              shuffle: bool = True, num_workers: int = 0) -> TorchDataLoader:
    """
    Create a PyTorch DataLoader from a TokenizedDataset.
    
    Args:
        tokenized_dataset: Source dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        PyTorch DataLoader
    """
    wrapper = TorchDatasetWrapper(tokenized_dataset)
    return TorchDataLoader(
        wrapper,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([item[0] for item in batch]),
            "attention_mask": torch.stack([item[1] for item in batch])
        }
    )
