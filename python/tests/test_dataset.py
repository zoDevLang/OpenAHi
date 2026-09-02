"""
Tests for OpenAHI Dataset
"""

import unittest
import tempfile
import os

from openahi.data import TextDataset, TokenizedDataset, DatasetConfig, create_tiny_dataset
from openahi.tokenizer import BPETokenizer, Vocabulary


class TestTextDataset(unittest.TestCase):
    """Test text dataset functionality."""
    
    def test_load_from_list(self):
        """Test loading dataset from list."""
        texts = ["hello world", "test sentence", "another text"]
        dataset = TextDataset()
        dataset.load_from_list(texts)
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.get_text(0), "hello world")
        self.assertEqual(dataset.get_text(1), "test sentence")
        self.assertEqual(dataset.get_text(2), "another text")
    
    def test_load_from_file(self):
        """Test loading dataset from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.txt")
            
            # Create test file
            with open(filepath, 'w') as f:
                f.write("line 1\nline 2\nline 3\n")
            
            # Load dataset
            dataset = TextDataset()
            dataset.load(filepath)
            
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.get_text(0), "line 1")
            self.assertEqual(dataset.get_text(1), "line 2")
            self.assertEqual(dataset.get_text(2), "line 3")
    
    def test_split(self):
        """Test dataset splitting."""
        texts = ["text " + str(i) for i in range(100)]
        dataset = TextDataset()
        dataset.load_from_list(texts)
        
        # Split
        dataset.split(train_ratio=0.8, val_ratio=0.1)
        
        # Check splits
        self.assertEqual(len(dataset.train_data), 80)
        self.assertEqual(len(dataset.val_data), 10)
        self.assertEqual(len(dataset.test_data), 10)
    
    def test_texts_property(self):
        """Test texts property."""
        texts = ["hello", "world"]
        dataset = TextDataset()
        dataset.load_from_list(texts)
        
        self.assertEqual(dataset.texts, texts)


class TestTokenizedDataset(unittest.TestCase):
    """Test tokenized dataset functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple tokenizer
        vocab = Vocabulary()
        self.tokenizer = BPETokenizer(vocab=vocab)
        
        # Create text dataset
        self.text_dataset = TextDataset()
        self.text_dataset.load_from_list(["hello world", "test text"])
    
    def test_from_text_dataset(self):
        """Test creating tokenized dataset from text dataset."""
        tokenized = TokenizedDataset()
        tokenized.from_text_dataset(self.text_dataset, self.tokenizer, max_length=10)
        
        self.assertEqual(len(tokenized), 2)
        
        # Check first item
        first_item = tokenized[0]
        self.assertIn("input_ids", first_item)
        self.assertIn("attention_mask", first_item)
    
    def test_to_tensors(self):
        """Test converting to tensors."""
        tokenized = TokenizedDataset()
        tokenized.from_text_dataset(self.text_dataset, self.tokenizer, max_length=10)
        
        input_ids, attention_mask = tokenized.to_tensors()
        
        self.assertEqual(input_ids.shape[0], 2)  # Batch size
        self.assertEqual(attention_mask.shape[0], 2)


class TestCreateTinyDataset(unittest.TestCase):
    """Test tiny dataset creation."""
    
    def test_tiny_dataset_creation(self):
        """Test creating tiny dataset."""
        dataset = create_tiny_dataset()
        
        self.assertGreater(len(dataset), 0)
        self.assertGreater(len(dataset.texts), 0)
    
    def test_tiny_dataset_content(self):
        """Test tiny dataset has expected content."""
        dataset = create_tiny_dataset()
        
        # Check that some expected phrases are in the dataset
        all_texts = ' '.join(dataset.texts).lower()
        
        self.assertIn("openahi", all_texts)
        self.assertIn("composter", all_texts)


class TestDatasetConfig(unittest.TestCase):
    """Test dataset configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DatasetConfig()
        
        self.assertEqual(config.name, "default")
        self.assertEqual(config.train_split, 0.9)
        self.assertEqual(config.val_split, 0.1)
        self.assertEqual(config.test_split, 0.0)
        self.assertEqual(config.seed, 42)


if __name__ == '__main__':
    unittest.main()
