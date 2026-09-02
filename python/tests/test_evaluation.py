"""
Tests for OpenAHI Evaluation
"""

import unittest
import torch

from openahi.evaluation import compute_perplexity, compute_accuracy, evaluate_model
from openahi.models.composter import Composter, ComposterConfig
from openahi.data import TextDataset, TokenizedDataset, create_tiny_dataset
from openahi.tokenizer import BPETokenizer, Vocabulary
from openahi.data.dataloader import OpenAHIDataLoader


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics."""
    
    def test_compute_perplexity(self):
        """Test perplexity computation."""
        # Create simple logits and targets
        logits = torch.randn(2, 3, 10)  # batch_size=2, seq_len=3, vocab_size=10
        targets = torch.randint(0, 10, (2, 3))
        
        perplexity = compute_perplexity(logits, targets)
        
        self.assertGreater(perplexity, 0)
        self.assertIsInstance(perplexity, float)
    
    def test_compute_accuracy(self):
        """Test accuracy computation."""
        # Create logits where predictions are correct
        logits = torch.zeros(2, 3, 10)
        targets = torch.tensor([[0, 1, 2], [3, 4, 5]])
        
        # Set high logits for correct answers
        for i in range(2):
            for j in range(3):
                logits[i, j, targets[i, j]] = 10.0
        
        accuracy = compute_accuracy(logits, targets)
        
        self.assertEqual(accuracy, 1.0)  # 100% accuracy
    
    def test_compute_accuracy_with_ignore_index(self):
        """Test accuracy with ignore index."""
        logits = torch.randn(2, 3, 10)
        targets = torch.tensor([[0, -1, 2], [3, 4, -1]])  # -1 is padding
        
        accuracy = compute_accuracy(logits, targets, ignore_index=-1)
        
        # Should only count non-padding tokens
        self.assertIsInstance(accuracy, float)
    
    def test_compute_accuracy_top_k(self):
        """Test top-k accuracy."""
        # Create logits where correct answer is in top-2
        logits = torch.zeros(2, 3, 10)
        targets = torch.tensor([[0, 1, 2], [3, 4, 5]])
        
        # Set high logits for correct and one other
        for i in range(2):
            for j in range(3):
                logits[i, j, targets[i, j]] = 10.0
                logits[i, j, (targets[i, j] + 1) % 10] = 9.0
        
        accuracy_top1 = compute_accuracy(logits.clone(), targets.clone(), top_k=1)
        accuracy_top2 = compute_accuracy(logits, targets, top_k=2)
        
        self.assertEqual(accuracy_top1, 1.0)  # Correct is highest
        self.assertEqual(accuracy_top2, 1.0)  # Correct is in top-2


class TestEvaluateModel(unittest.TestCase):
    """Test model evaluation."""
    
    def test_evaluate_model(self):
        """Test model evaluation."""
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
        model.eval()
        
        # Create a simple dataset
        dataset = create_tiny_dataset()
        
        # Create tokenizer
        vocab = Vocabulary()
        tokenizer = BPETokenizer(vocab=vocab)
        
        # Tokenize dataset
        tokenized = TokenizedDataset()
        tokenized.from_text_dataset(dataset, tokenizer, max_length=16)
        
        # Create dataloader
        dataloader = OpenAHIDataLoader(
            tokenized,
            batch_size=2,
            shuffle=False,
            device="cpu",
        )
        
        # Evaluate
        results = evaluate_model(model, dataloader, device="cpu")
        
        self.assertIn("loss", results)
        self.assertIn("perplexity", results)
        self.assertIn("accuracy", results)
        self.assertIn("num_batches", results)
        self.assertIn("num_tokens", results)


if __name__ == '__main__':
    unittest.main()
