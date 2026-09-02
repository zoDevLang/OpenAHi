"""
Trainer for OpenAHI Models

Provides complete training pipeline with checkpoints, logging, and validation.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from openahi.models.composter import Composter, ComposterConfig
from openahi.data.dataset import TokenizedDataset
from openahi.data.dataloader import OpenAHIDataLoader, create_pytorch_dataloader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model configuration
    model_config: Optional[ComposterConfig] = None
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 10
    max_steps: Optional[int] = None
    
    # Optimization
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    
    # Regularization
    dropout: float = 0.1
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_freq: int = 500  # Steps between checkpoints
    save_best: bool = True
    
    # Validation
    val_freq: int = 100  # Steps between validation
    val_batch_size: int = 64
    
    # Logging
    log_dir: str = "./logs"
    log_freq: int = 10  # Steps between logging
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    seed: int = 42
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.001
    
    def __post_init__(self):
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


@dataclass
class TrainingState:
    """State of training."""
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float('inf')
    train_loss: float = 0.0
    val_loss: float = 0.0
    
    # Timestamps
    start_time: Optional[str] = None
    last_checkpoint_time: Optional[str] = None
    
    # Checkpoint info
    last_checkpoint_path: Optional[str] = None
    best_checkpoint_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainingState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Trainer:
    """
    Trainer for OpenAHI models.
    
    Handles the complete training loop including:
    - Forward/backward passes
    - Optimization
    - Checkpointing
    - Validation
    - Logging
    - Reproducibility
    """
    
    def __init__(self, config: TrainingConfig, model: Optional[Composter] = None):
        self.config = config
        self.state = TrainingState()
        
        # Set random seeds for reproducibility
        self._set_seeds()
        
        # Initialize model
        if model is None:
            if config.model_config is None:
                config.model_config = ComposterConfig()
            self.model = Composter(config.model_config)
        else:
            self.model = model
        
        # Move model to device
        self.model = self.model.to(config.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Loss function (cross-entropy)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Initialize logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Training state tracking
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        
        logger.info(f"Trainer initialized with {self.model.get_num_params():,} parameters")
        logger.info(f"Using device: {config.device}")
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import random
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        params = self.model.parameters()
        
        if self.config.optimizer.lower() == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
            )
        elif self.config.optimizer.lower() == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                betas=self.config.betas,
            )
        elif self.config.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _compute_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """
        Compute loss for a batch.
        
        Args:
            batch: Tuple of (input_ids, attention_mask)
            
        Returns:
            Tuple of (loss, loss_value)
        """
        input_ids, attention_mask = batch
        
        # Forward pass
        # Shift input for language modeling: predict next token
        # input_ids: (batch_size, seq_len)
        # We want to predict token at position i from tokens 0..i-1
        
        # Get logits
        logits = self.model(input_ids[:, :-1])  # (batch_size, seq_len-1, vocab_size)
        
        # Target is the next token
        targets = input_ids[:, 1:]  # (batch_size, seq_len-1)
        
        # Reshape for cross-entropy
        logits = logits.view(-1, logits.size(-1))  # (batch_size * (seq_len-1), vocab_size)
        targets = targets.view(-1)  # (batch_size * (seq_len-1),)
        
        # Mask out padding tokens
        # Create mask where target != pad_token_id (0)
        mask = targets != 0
        
        # Apply mask to logits and targets
        logits = logits[mask]
        targets = targets[mask]
        
        if len(targets) == 0:
            return torch.tensor(0.0, device=self.config.device), 0.0
        
        # Compute loss
        loss = self.criterion(logits, targets)
        
        return loss, loss.item()
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Perform a single training step.
        
        Args:
            batch: Tuple of (input_ids, attention_mask)
            
        Returns:
            Loss value
        """
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute loss
        loss, loss_value = self._compute_loss(batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return loss_value
    
    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Perform a validation step.
        
        Args:
            batch: Tuple of (input_ids, attention_mask)
            
        Returns:
            Loss value
        """
        self.model.eval()
        
        with torch.no_grad():
            _, loss_value = self._compute_loss(batch)
        
        return loss_value
    
    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint. If None, uses step-based naming.
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to saved checkpoint
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(
                self.config.checkpoint_dir,
                f"composter_{self.state.step:06d}_{timestamp}.pt"
            )
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.config,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_state": self.state.to_dict(),
            "config": asdict(self.config),
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
        }
        
        torch.save(checkpoint, path)
        
        # Update state
        self.state.last_checkpoint_path = path
        self.state.last_checkpoint_time = datetime.now().isoformat()
        
        if is_best:
            self.state.best_checkpoint_path = path
        
        logger.info(f"Checkpoint saved to {path}")
        
        return path
    
    @classmethod
    def load_checkpoint(cls, path: str, config: Optional[TrainingConfig] = None) -> Tuple["Trainer", str]:
        """
        Load trainer from checkpoint.
        
        Args:
            path: Path to checkpoint
            config: Optional training configuration (overrides checkpoint config)
            
        Returns:
            Tuple of (trainer, checkpoint_path)
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        # Get configuration
        if config is None:
            checkpoint_config = checkpoint.get("config", {})
            config = TrainingConfig(**checkpoint_config)
        
        # Create model
        model_config = checkpoint["model_config"]
        model = Composter(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Create trainer
        trainer = cls(config, model=model)
        
        # Restore training state
        training_state = checkpoint.get("training_state", {})
        trainer.state = TrainingState.from_dict(training_state)
        
        # Restore optimizer
        if "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore loss history
        trainer.train_loss_history = checkpoint.get("train_loss_history", [])
        trainer.val_loss_history = checkpoint.get("val_loss_history", [])
        
        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"Resuming training at step {trainer.state.step}, epoch {trainer.state.epoch}")
        
        return trainer, path
    
    def train(self, train_dataset: TokenizedDataset, 
              val_dataset: Optional[TokenizedDataset] = None,
              train_dataloader: Optional[OpenAHIDataLoader] = None,
              val_dataloader: Optional[OpenAHIDataLoader] = None) -> TrainingState:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            train_dataloader: Training dataloader (optional)
            val_dataloader: Validation dataloader (optional)
            
        Returns:
            Final training state
        """
        # Create dataloaders if not provided
        if train_dataloader is None:
            train_dataloader = OpenAHIDataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                device=self.config.device,
                seed=self.config.seed
            )
        
        if val_dataset is not None and val_dataloader is None:
            val_dataloader = OpenAHIDataLoader(
                val_dataset,
                batch_size=self.config.val_batch_size,
                shuffle=False,
                device=self.config.device,
                seed=self.config.seed
            )
        
        # Set start time
        if self.state.start_time is None:
            self.state.start_time = datetime.now().isoformat()
        
        # Training loop
        early_stopping_counter = 0
        
        for epoch in range(self.state.epoch, self.config.num_epochs):
            self.state.epoch = epoch
            
            epoch_train_loss = 0.0
            num_train_batches = 0
            
            # Iterate over batches
            for batch_idx, batch in enumerate(train_dataloader):
                # Check if we've reached max steps
                if self.config.max_steps and self.state.step >= self.config.max_steps:
                    logger.info(f"Reached max steps: {self.config.max_steps}")
                    return self.state
                
                # Training step
                loss = self.train_step(batch)
                epoch_train_loss += loss
                num_train_batches += 1
                self.state.step += 1
                
                # Update training loss history
                self.train_loss_history.append(loss)
                
                # Logging
                if self.state.step % self.config.log_freq == 0:
                    avg_loss = epoch_train_loss / num_train_batches
                    logger.info(f"Step {self.state.step:6d} | Epoch {epoch:3d} | "
                               f"Train Loss: {avg_loss:.4f}")
                    
                    # Log to TensorBoard
                    self.writer.add_scalar("Loss/train", avg_loss, self.state.step)
                    self.writer.add_scalar("Loss/train_batch", loss, self.state.step)
                
                # Validation
                if val_dataloader is not None and self.state.step % self.config.val_freq == 0:
                    val_loss = self.validate(val_dataloader)
                    self.val_loss_history.append(val_loss)
                    
                    logger.info(f"Step {self.state.step:6d} | Val Loss: {val_loss:.4f}")
                    
                    # Log to TensorBoard
                    self.writer.add_scalar("Loss/val", val_loss, self.state.step)
                    
                    # Check for best model
                    if val_loss < self.state.best_val_loss:
                        self.state.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                    
                    # Early stopping
                    if (self.config.early_stopping_patience is not None and
                        early_stopping_counter >= self.config.early_stopping_patience):
                        logger.info(f"Early stopping triggered after {early_stopping_counter} steps")
                        return self.state
                
                # Checkpointing
                if self.state.step % self.config.checkpoint_freq == 0:
                    self.save_checkpoint()
            
            # End of epoch
            epoch_avg_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
            logger.info(f"Epoch {epoch:3d} | Avg Train Loss: {epoch_avg_loss:.4f}")
            
            # Log epoch to TensorBoard
            self.writer.add_scalar("Loss/train_epoch", epoch_avg_loss, epoch)
        
        # Final checkpoint
        self.save_checkpoint()
        
        return self.state
    
    def validate(self, val_dataloader: OpenAHIDataLoader) -> float:
        """
        Validate the model on validation data.
        
        Args:
            val_dataloader: Validation dataloader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                loss = self.val_step(batch)
                total_loss += loss
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def evaluate(self, test_dataloader: OpenAHIDataLoader) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_dataloader: Test dataloader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                loss = self.val_step(batch)
                total_loss += loss
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "perplexity": math.exp(avg_loss) if num_batches > 0 else 0.0,
        }
    
    def generate(self, prompt: str, tokenizer, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            tokenizer: Tokenizer to use
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_k: Number of top tokens to sample from
            
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.config.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        
        return generated_text
    
    def close(self):
        """Clean up resources."""
        self.writer.close()


import math
