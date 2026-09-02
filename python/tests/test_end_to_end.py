"""
End-to-End Tests for OpenAHI

Tests the complete workflow from dataset to training to inference.
"""

import unittest
import tempfile
import os
import shutil

import torch

from openahi import (
    Composter, ComposterConfig, Trainer, TrainingConfig,
    create_tiny_dataset, BPETokenizer, Vocabulary,
    TokenizedDataset, InferenceEngine, InferenceConfig
)


class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for OpenAHI ecosystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        self.log_dir = os.path.join(self.temp_dir, "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self):
        """Test the complete workflow: dataset -> training -> inference."""
        
        # Step 1: Create dataset
        dataset = create_tiny_dataset()
        self.assertGreater(len(dataset), 0, "Dataset should have samples")
        
        # Step 2: Create tokenizer
        vocab = Vocabulary()
        tokenizer = BPETokenizer(vocab=vocab)
        
        # Step 3: Tokenize dataset
        tokenized = TokenizedDataset()
        tokenized.from_text_dataset(dataset, tokenizer, max_length=32)
        tokenized.split(train_ratio=0.9, val_ratio=0.1)
        
        self.assertGreater(len(tokenized.train_data), 0, "Should have training data")
        
        # Step 4: Create model
        model_config = ComposterConfig(
            vocab_size=128,
            context_length=32,
            embedding_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.0,  # Disable dropout for deterministic testing
        )
        model = Composter(model_config)
        
        # Step 5: Create trainer
        train_config = TrainingConfig(
            batch_size=2,
            learning_rate=1e-3,
            num_epochs=1,
            max_steps=10,  # Only 10 steps for testing
            device="cpu",
            checkpoint_dir=self.checkpoint_dir,
            log_dir=self.log_dir,
            seed=42,
        )
        trainer = Trainer(train_config, model)
        
        # Step 6: Train
        final_state = trainer.train(tokenized)
        
        self.assertGreater(final_state.step, 0, "Should have trained at least one step")
        self.assertGreaterEqual(final_state.epoch, 0, "Should have completed at least one epoch")
        
        # Step 7: Save checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, "test_checkpoint.pt")
        trainer.save_checkpoint(checkpoint_path)
        self.assertTrue(os.path.exists(checkpoint_path), "Checkpoint should exist")
        
        # Step 8: Load checkpoint
        loaded_trainer, _ = Trainer.load_checkpoint(checkpoint_path, None)
        self.assertIsNotNone(loaded_trainer, "Should be able to load checkpoint")
        
        # Step 9: Create inference engine
        inference_config = InferenceConfig(
            max_length=20,
            temperature=1.0,
            device="cpu",
        )
        
        # Use the model from the trainer
        inference_engine = InferenceEngine(
            inference_config,
            model=loaded_trainer.model,
            tokenizer=tokenizer
        )
        
        # Step 10: Generate text
        result = inference_engine.generate("hello world", max_new_tokens=10)
        
        self.assertIsInstance(result, list, "Should return list of results")
        self.assertEqual(len(result), 1, "Should return one result")
        self.assertIsInstance(result[0], str, "Result should be a string")
        
        # Step 11: Batch generate
        results = inference_engine.batch_generate(
            ["hello", "world", "test"],
            max_new_tokens=5
        )
        
        self.assertEqual(len(results), 3, "Should return three results")
        for result in results:
            self.assertIsInstance(result, str, "Each result should be a string")
        
        print("\n=== End-to-End Test Complete ===")
        print(f"Trained for {final_state.step} steps")
        print(f"Generated text: {result[0][:100]}...")
    
    def test_model_save_load_cycle(self):
        """Test saving and loading model with checkpoint."""
        
        # Create model
        model_config = ComposterConfig(
            vocab_size=64,
            context_length=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        )
        model = Composter(model_config)
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "model_test.pt")
        model.save_checkpoint(checkpoint_path)
        
        # Load checkpoint
        loaded_model = Composter.load_checkpoint(checkpoint_path)
        
        # Verify loaded model
        self.assertEqual(loaded_model.config.vocab_size, model_config.vocab_size)
        self.assertEqual(loaded_model.config.context_length, model_config.context_length)
        self.assertEqual(loaded_model.get_num_params(), model.get_num_params())
    
    def test_generation_with_loaded_model(self):
        """Test generation with a model loaded from checkpoint."""
        
        # Create and save model
        model_config = ComposterConfig(
            vocab_size=64,
            context_length=16,
            embedding_dim=32,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        )
        model = Composter(model_config)
        
        checkpoint_path = os.path.join(self.temp_dir, "model_gen.pt")
        model.save_checkpoint(checkpoint_path)
        
        # Load model
        loaded_model = Composter.load_checkpoint(checkpoint_path)
        
        # Create tokenizer
        vocab = Vocabulary()
        tokenizer = BPETokenizer(vocab=vocab)
        
        # Generate with loaded model
        input_ids = tokenizer.encode("test")
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        generated = loaded_model.generate(input_tensor, max_new_tokens=10)
        
        self.assertIsInstance(generated, torch.Tensor)
        self.assertGreater(generated.shape[1], len(input_ids), "Should have new tokens")


if __name__ == '__main__':
    unittest.main()
