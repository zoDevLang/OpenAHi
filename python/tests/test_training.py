"""
Tests for OpenAHI Training
"""

import unittest
import tempfile
import os

import torch

from openahi.models.composter import Composter, ComposterConfig
from openahi.training import Trainer, TrainingConfig, TrainingState
from openahi.data import TextDataset, TokenizedDataset, create_tiny_dataset
from openahi.tokenizer import BPETokenizer, Vocabulary
from openahi.data.dataloader import OpenAHIDataLoader


class TestTrainingConfig(unittest.TestCase):
    """Test training configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig()
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.num_epochs, 10)
        self.assertEqual(config.optimizer, "adamw")
        self.assertEqual(config.device, "cuda" if torch.cuda.is_available() else "cpu")


class TestTrainer(unittest.TestCase):
    """Test trainer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small model for testing
        self.model_config = ComposterConfig(
            vocab_size=128,
            context_length=32,
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,  # Disable dropout for deterministic testing
        )
        self.model = Composter(self.model_config)
        
        # Create training config
        self.train_config = TrainingConfig(
            batch_size=2,
            learning_rate=1e-3,
            num_epochs=1,
            device="cpu",  # Use CPU for testing
            checkpoint_dir="./test_checkpoints",
            log_dir="./test_logs",
        )
        
        # Create trainer
        self.trainer = Trainer(self.train_config, self.model)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists("./test_checkpoints"):
            shutil.rmtree("./test_checkpoints")
        if os.path.exists("./test_logs"):
            shutil.rmtree("./test_logs")
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        self.assertIsNotNone(self.trainer)
        self.assertEqual(self.trainer.config.batch_size, 2)
        self.assertEqual(self.trainer.config.learning_rate, 1e-3)
    
    def test_trainer_state(self):
        """Test trainer state."""
        self.assertEqual(self.trainer.state.step, 0)
        self.assertEqual(self.trainer.state.epoch, 0)
        self.assertEqual(self.trainer.state.best_val_loss, float('inf'))
    
    def test_model_info(self):
        """Test getting model info."""
        info = self.trainer.get_model_info()
        
        self.assertEqual(info["model_type"], "composter")
        self.assertEqual(info["version"], "1.00.0")
        self.assertEqual(info["vocab_size"], str(self.model_config.vocab_size))


class TestTrainingPipeline(unittest.TestCase):
    """Test the complete training pipeline."""
    
    def test_tiny_training_run(self):
        """Test a tiny training run with the tiny dataset."""
        # Create a very small model
        model_config = ComposterConfig(
            vocab_size=64,
            context_length=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        )
        model = Composter(model_config)
        
        # Create training config
        train_config = TrainingConfig(
            batch_size=2,
            learning_rate=1e-3,
            num_epochs=1,
            device="cpu",
            checkpoint_dir="./test_checkpoints_tiny",
            log_dir="./test_logs_tiny",
            max_steps=10,  # Only 10 steps for testing
        )
        
        trainer = Trainer(train_config, model)
        
        # Create tiny dataset
        dataset = create_tiny_dataset()
        
        # Create a simple tokenizer
        vocab = Vocabulary()
        tokenizer = BPETokenizer(vocab=vocab)
        
        # Tokenize dataset
        tokenized = TokenizedDataset()
        tokenized.from_text_dataset(dataset, tokenizer, max_length=16)
        
        # Split dataset
        tokenized.split(train_ratio=0.9, val_ratio=0.1)
        
        # Train (this will be very slow with pure Python, but should work)
        # Note: This test might take a while depending on the hardware
        try:
            state = trainer.train(
                tokenized,
                val_dataset=None,
            )
            
            # Check that training completed
            self.assertGreater(state.step, 0)
            
        finally:
            # Clean up
            import shutil
            if os.path.exists("./test_checkpoints_tiny"):
                shutil.rmtree("./test_checkpoints_tiny")
            if os.path.exists("./test_logs_tiny"):
                shutil.rmtree("./test_logs_tiny")


class TestCheckpointing(unittest.TestCase):
    """Test checkpointing functionality."""
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading trainer checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model and trainer
            model_config = ComposterConfig(
                vocab_size=64,
                context_length=16,
                embedding_dim=32,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
            )
            model = Composter(model_config)
            
            train_config = TrainingConfig(
                checkpoint_dir=tmpdir,
                log_dir=tmpdir,
                device="cpu",
            )
            
            trainer = Trainer(train_config, model)
            
            # Save checkpoint
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
            trainer.save_checkpoint(checkpoint_path)
            
            # Check file exists
            self.assertTrue(os.path.exists(checkpoint_path))
            
            # Load checkpoint
            loaded_trainer, _ = Trainer.load_checkpoint(checkpoint_path, None)
            
            # Check loaded trainer
            self.assertIsNotNone(loaded_trainer)


if __name__ == '__main__':
    unittest.main()
