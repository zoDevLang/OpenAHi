#!/usr/bin/env python3
"""
Example: Train Composter 1.00.0

This script demonstrates how to train the Composter model on a small dataset.
"""

import torch
import argparse
import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openahi import (
    Composter, ComposterConfig, Trainer, TrainingConfig,
    create_tiny_dataset, BPETokenizer, Vocabulary
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_composter(
    vocab_size: int = 8192,
    context_length: int = 512,
    embedding_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    num_epochs: int = 10,
    max_steps: int = None,
    device: str = None,
    checkpoint_dir: str = "./checkpoints",
    log_dir: str = "./logs",
    save_checkpoint: bool = True,
):
    """Train the Composter model."""
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Create model configuration
    model_config = ComposterConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
    
    logger.info(f"Model configuration: {model_config}")
    
    # Create model
    model = Composter(model_config)
    model.to(device)
    
    logger.info(f"Model created with {model.get_num_params():,} parameters")
    
    # Create training configuration
    train_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_steps=max_steps,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        seed=42,
    )
    
    # Create trainer
    trainer = Trainer(train_config, model)
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = create_tiny_dataset()
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    # Create tokenizer
    logger.info("Creating tokenizer...")
    vocab = Vocabulary()
    tokenizer = BPETokenizer(vocab=vocab)
    
    # Tokenize dataset
    from openahi.data import TokenizedDataset
    tokenized = TokenizedDataset()
    tokenized.from_text_dataset(dataset, tokenizer, max_length=context_length)
    
    # Split dataset
    tokenized.split(train_ratio=0.9, val_ratio=0.1)
    
    logger.info(f"Train samples: {len(tokenized.train_data)}")
    logger.info(f"Val samples: {len(tokenized.val_data)}")
    
    # Train
    logger.info("Starting training...")
    try:
        final_state = trainer.train(
            tokenized,
            val_dataset=tokenized,
        )
        
        logger.info(f"Training completed at step {final_state.step}, epoch {final_state.epoch}")
        logger.info(f"Best validation loss: {final_state.best_val_loss}")
        
        # Save final checkpoint
        if save_checkpoint:
            final_checkpoint = os.path.join(
                checkpoint_dir,
                f"composter_{final_state.step:06d}_final.pt"
            )
            trainer.save_checkpoint(final_checkpoint)
            logger.info(f"Final checkpoint saved to {final_checkpoint}")
        
        return trainer
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save checkpoint on interrupt
        if save_checkpoint:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"composter_interrupted_{trainer.state.step:06d}.pt"
            )
            trainer.save_checkpoint(checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        return trainer


def test_generation(trainer):
    """Test text generation with the trained model."""
    logger.info("Testing text generation...")
    
    # Get the model from trainer
    model = trainer.model
    
    # Create a simple tokenizer
    vocab = Vocabulary()
    tokenizer = BPETokenizer(vocab=vocab)
    
    # Test prompts
    prompts = [
        "The quick brown fox",
        "OpenAHI is",
        "Machine learning",
        "Once upon a time",
    ]
    
    for prompt in prompts:
        # Tokenize
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(trainer.config.device)
        
        # Generate
        generated = model.generate(
            input_tensor,
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
        )
        
        # Decode
        output = tokenizer.decode(generated[0].tolist())
        
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Output: {output}")
        logger.info("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Train Composter 1.00.0 - The first OpenAHI model"
    )
    
    # Model arguments
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=512, help="Context length")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps")
    
    # System arguments
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save checkpoints")
    
    # Mode arguments
    parser.add_argument("--test-only", action="store_true", help="Only test generation (no training)")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Load from checkpoint")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Train or load
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        logger.info(f"Loading from checkpoint: {args.load_checkpoint}")
        trainer, _ = Trainer.load_checkpoint(args.load_checkpoint, None)
    else:
        trainer = train_composter(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_steps=args.max_steps,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            save_checkpoint=not args.no_save,
        )
    
    # Test generation
    if not args.test_only:
        test_generation(trainer)


if __name__ == "__main__":
    main()
