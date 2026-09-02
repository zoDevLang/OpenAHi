"""
Tests for OpenAHI Models
"""

import unittest
import torch
import tempfile
import os

from openahi.models.composter import Composter, ComposterConfig, ModelConfig


class TestComposterConfig(unittest.TestCase):
    """Test Composter configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ComposterConfig()
        
        self.assertEqual(config.vocab_size, 8192)
        self.assertEqual(config.context_length, 512)
        self.assertEqual(config.embedding_dim, 512)
        self.assertEqual(config.num_layers, 6)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.head_dim, 512 // 8)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ComposterConfig(
            vocab_size=4096,
            context_length=256,
            embedding_dim=256,
            num_layers=4,
            num_heads=4,
            dropout=0.05,
        )
        
        self.assertEqual(config.vocab_size, 4096)
        self.assertEqual(config.context_length, 256)
        self.assertEqual(config.embedding_dim, 256)
        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.num_heads, 4)
        self.assertEqual(config.dropout, 0.05)
        self.assertEqual(config.head_dim, 256 // 4)
    
    def test_invalid_heads(self):
        """Test that invalid head count raises error."""
        with self.assertRaises(AssertionError):
            ComposterConfig(
                vocab_size=8192,
                context_length=512,
                embedding_dim=512,
                num_layers=6,
                num_heads=7,  # 512 is not divisible by 7
                dropout=0.1,
            )


class TestComposterModel(unittest.TestCase):
    """Test Composter model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ComposterConfig(
            vocab_size=128,
            context_length=32,
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,  # Disable dropout for deterministic testing
        )
        self.model = Composter(self.config)
    
    def test_model_creation(self):
        """Test model creation."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.config.vocab_size, 128)
        self.assertEqual(self.model.config.context_length, 32)
    
    def test_parameter_count(self):
        """Test parameter count calculation."""
        num_params = self.model.get_num_params()
        self.assertGreater(num_params, 0)
        
        # Calculate expected parameters
        # Token embeddings: vocab_size * embedding_dim
        # Position embeddings: context_length * embedding_dim
        # Each transformer block:
        #   - Attention: 4 * (embedding_dim * embedding_dim) for Q, K, V, O
        #   - FFN: 2 * (embedding_dim * hidden_dim) + hidden_dim * embedding_dim
        #   - Layer norm: 2 * embedding_dim (gamma + beta)
        # Output projection: embedding_dim * vocab_size + vocab_size
        
        expected_params = 0
        expected_params += self.config.vocab_size * self.config.embedding_dim  # Token embeddings
        expected_params += self.config.context_length * self.config.embedding_dim  # Position embeddings
        
        hidden_dim = self.config.embedding_dim * 4
        for _ in range(self.config.num_layers):
            # Attention
            expected_params += 4 * (self.config.embedding_dim * self.config.embedding_dim)
            # FFN
            expected_params += 2 * (self.config.embedding_dim * hidden_dim)
            # Layer norm
            expected_params += 2 * self.config.embedding_dim
        
        # Output projection
        expected_params += self.config.embedding_dim * self.config.vocab_size
        expected_params += self.config.vocab_size  # Bias
        
        self.assertEqual(num_params, expected_params)
    
    def test_forward_pass(self):
        """Test forward pass."""
        # Create input
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Check output shape
        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[1], seq_len)
        self.assertEqual(logits.shape[2], self.config.vocab_size)
    
    def test_generation(self):
        """Test text generation."""
        # Create input
        input_ids = torch.tensor([[1, 2, 3]])  # Small input
        
        # Generate
        generated = self.model.generate(input_ids, max_new_tokens=10)
        
        # Check output shape
        self.assertEqual(generated.shape[0], 1)
        self.assertGreater(generated.shape[1], 3)  # Should have new tokens
        self.assertLessEqual(generated.shape[1], 13)  # Input + max_new_tokens
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading model checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "composter_test.pt")
            
            # Save checkpoint
            self.model.save_checkpoint(filepath)
            
            # Check file exists
            self.assertTrue(os.path.exists(filepath))
            
            # Load checkpoint
            loaded_model = Composter.load_checkpoint(filepath)
            
            # Check loaded model
            self.assertEqual(loaded_model.config.vocab_size, self.config.vocab_size)
            self.assertEqual(loaded_model.config.context_length, self.config.context_length)
    
    def test_get_config_dict(self):
        """Test getting configuration as dictionary."""
        config_dict = self.model.get_config_dict()
        
        self.assertEqual(config_dict["model_type"], "composter")
        self.assertEqual(config_dict["version"], "1.00.0")
        self.assertEqual(config_dict["vocab_size"], self.config.vocab_size)
        self.assertEqual(config_dict["context_length"], self.config.context_length)


class TestModelConfig(unittest.TestCase):
    """Test model configuration wrapper."""
    
    def test_default_model_config(self):
        """Test default model configuration."""
        config = ModelConfig()
        
        self.assertEqual(config.model_type, "composter")
        self.assertEqual(config.version, "1.00.0")
        self.assertIsNotNone(config.composter)


if __name__ == '__main__':
    unittest.main()
