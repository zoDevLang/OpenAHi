"""
Tests for OpenAHI Inference
"""

import unittest
import tempfile
import os

import torch

from openahi.models.composter import Composter, ComposterConfig
from openahi.inference import InferenceEngine, InferenceConfig
from openahi.tokenizer import BPETokenizer, Vocabulary


class TestInferenceConfig(unittest.TestCase):
    """Test inference configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = InferenceConfig()
        
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.temperature, 1.0)
        self.assertIsNone(config.top_k)
        self.assertEqual(config.batch_size, 1)
        self.assertEqual(config.device, "cuda" if torch.cuda.is_available() else "cpu")


class TestInferenceEngine(unittest.TestCase):
    """Test inference engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small model
        self.model_config = ComposterConfig(
            vocab_size=128,
            context_length=32,
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
        )
        self.model = Composter(self.model_config)
        
        # Create inference config
        self.config = InferenceConfig(
            max_length=16,
            temperature=1.0,
            device="cpu",
        )
        
        # Create a simple tokenizer
        vocab = Vocabulary()
        self.tokenizer = BPETokenizer(vocab=vocab)
        
        # Create inference engine
        self.engine = InferenceEngine(self.config, model=self.model, tokenizer=self.tokenizer)
    
    def test_engine_creation(self):
        """Test inference engine creation."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.config.max_length, 16)
    
    def test_model_info(self):
        """Test getting model info."""
        info = self.engine.get_model_info()
        
        self.assertEqual(info["model_type"], "composter")
        self.assertEqual(info["version"], "1.00.0")
        self.assertEqual(info["vocab_size"], str(self.model_config.vocab_size))
    
    def test_generation(self):
        """Test text generation."""
        # Set tokenizer on engine
        self.engine.set_tokenizer(self.tokenizer)
        
        # Generate text
        result = self.engine.generate("hello world", max_new_tokens=10)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)  # Single return sequence
        self.assertIsInstance(result[0], str)
    
    def test_batch_generation(self):
        """Test batch generation."""
        # Set tokenizer on engine
        self.engine.set_tokenizer(self.tokenizer)
        
        # Batch generate
        prompts = ["hello", "world", "test"]
        results = self.engine.batch_generate(prompts, max_new_tokens=5)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, str)
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        usage = self.engine.get_memory_usage()
        
        self.assertIn("param_size_mb", usage)
        self.assertIn("buffer_size_mb", usage)
        self.assertIn("total_size_mb", usage)
        self.assertGreater(usage["total_size_mb"], 0)


class TestLoadFromCheckpoint(unittest.TestCase):
    """Test loading inference engine from checkpoint."""
    
    def test_load_from_checkpoint(self):
        """Test loading from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save a model
            model_config = ComposterConfig(
                vocab_size=64,
                context_length=16,
                embedding_dim=32,
                num_layers=1,
                num_heads=2,
                dropout=0.0,
            )
            model = Composter(model_config)
            
            checkpoint_path = os.path.join(tmpdir, "model.pt")
            model.save_checkpoint(checkpoint_path)
            
            # Create inference config
            config = InferenceConfig(
                model_path=checkpoint_path,
                device="cpu",
            )
            
            # Create inference engine
            engine = InferenceEngine(config)
            
            # Check model loaded
            info = engine.get_model_info()
            self.assertEqual(info["model_type"], "composter")


class TestBenchmark(unittest.TestCase):
    """Test benchmarking functionality."""
    
    def test_benchmark(self):
        """Test benchmarking."""
        # Create a small model
        model_config = ComposterConfig(
            vocab_size=64,
            context_length=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        )
        model = Composter(model_config)
        
        # Create inference engine
        config = InferenceConfig(
            max_length=10,
            device="cpu",
        )
        
        vocab = Vocabulary()
        tokenizer = BPETokenizer(vocab=vocab)
        
        engine = InferenceEngine(config, model=model, tokenizer=tokenizer)
        
        # Run benchmark
        results = engine.benchmark("test prompt", num_tokens=10, num_runs=3)
        
        self.assertIn("avg_time_per_run", results)
        self.assertIn("tokens_per_second", results)
        self.assertIn("num_runs", results)
        self.assertEqual(results["num_runs"], 3)


if __name__ == '__main__':
    unittest.main()
