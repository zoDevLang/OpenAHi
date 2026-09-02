"""
Optimizer Utilities

Provides optimizer creation and configuration utilities.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def create_optimizer(model: nn.Module, optimizer_type: str = "adamw",
                     learning_rate: float = 3e-4,
                     weight_decay: float = 0.01,
                     betas: Tuple[float, float] = (0.9, 0.999)) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ("adamw", "adam", "sgd")
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        betas: Betas for Adam optimizers
        
    Returns:
        PyTorch optimizer
    """
    params = model.parameters()
    
    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
        )
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(
            params,
            lr=learning_rate,
            betas=betas,
        )
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_optimizer_state(optimizer: torch.optim.Optimizer) -> dict:
    """Get optimizer state as a dictionary."""
    return {
        "state_dict": optimizer.state_dict(),
        "defaults": optimizer.defaults,
    }


def set_optimizer_state(optimizer: torch.optim.Optimizer, state: dict) -> None:
    """Set optimizer state from a dictionary."""
    optimizer.load_state_dict(state["state_dict"])
