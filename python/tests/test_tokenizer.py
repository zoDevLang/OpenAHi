"""
Tests for OpenAHI Tokenizer
"""

import unittest
import tempfile
import os

from openahi.tokenizer import Tokenizer, BPETokenizer, Vocabulary


class TestVocabulary(unittest.TestCase):
    """Test vocabulary functionality."""
    
    def test_default_vocabulary(self):
        """Test default vocabulary creation."""
        vocab = Vocabulary()
        
        # Check special tokens
        self.assertIn("[PAD]", vocab.token_to_id)
        self.assertIn("[BOS]", vocab.token_to_id)
        self.assertIn("[EOS]", vocab.token_to_id)
        self.assertIn("[UNK]", vocab.token_to_id)
        
        # Check IDs
        self.assertEqual(vocab.get_id("[PAD]"), 0)
        self.assertEqual(vocab.get_id("[BOS]"), 1)
        self.assertEqual(vocab.get_id("[EOS]"), 2)
        self.assertEqual(vocab.get_id("[UNK]"), 3)
        
        # Check size
        self.assertEqual(len(vocab), 4)
    
    def test_add_token(self):
        """Test adding tokens to vocabulary."""
        vocab = Vocabulary()
        
        # Add a token
        token_id = vocab.add_token("hello")
        self.assertEqual(token_id, 4)  # After special tokens
        
        # Check token to ID
        self.assertEqual(vocab.get_id("hello"), 4)
        
        # Check ID to token
        self.assertEqual(vocab.get_token(4), "hello")
        
        # Check size
        self.assertEqual(len(vocab), 5)
    
    def test_add_duplicate_token(self):
        """Test adding duplicate token."""
        vocab = Vocabulary()
        
        # Add token
        id1 = vocab.add_token("hello")
        id2 = vocab.add_token("hello")
        
        # Should return same ID
        self.assertEqual(id1, id2)
        self.assertEqual(len(vocab), 5)  # Only 1 new token
    
    def test_add_special_token(self):
        """Test adding special tokens."""
        vocab = Vocabulary()
        
        # Add a special token
        token_id = vocab.add_special_token("[CLS]")
        
        # Check it's in special tokens
        self.assertIn("[cls]", vocab.special_tokens)
        self.assertEqual(vocab.special_tokens["[cls]"], token_id)
    
    def test_save_and_load_vocabulary(self):
        """Test saving and loading vocabulary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "vocab.json")
            
            # Create and populate vocabulary
            vocab = Vocabulary()
            vocab.add_token("hello")
            vocab.add_token("world")
            
            # Save
            vocab.save(filepath)
            
            # Check file exists
            self.assertTrue(os.path.exists(filepath))
            
            # Load
            loaded_vocab = Vocabulary.load(filepath)
            
            # Check loaded vocabulary
            self.assertEqual(loaded_vocab.get_id("hello"), vocab.get_id("hello"))
            self.assertEqual(loaded_vocab.get_id("world"), vocab.get_id("world"))
    
    def test_from_tokens(self):
        """Test creating vocabulary from tokens."""
        tokens = ["hello", "world", "test"]
        vocab = Vocabulary.from_tokens(tokens)
        
        # Check tokens are in vocabulary
        self.assertIn("hello", vocab.token_to_id)
        self.assertIn("world", vocab.token_to_id)
        self.assertIn("test", vocab.token_to_id)


class TestBPETokenizer(unittest.TestCase):
    """Test BPE tokenizer functionality."""
    
    def test_tokenizer_creation(self):
        """Test tokenizer creation."""
        vocab = Vocabulary()
        vocab.add_token("hello")
        vocab.add_token("world")
        
        tokenizer = BPETokenizer(vocab=vocab)
        self.assertIsNotNone(tokenizer)
        self.assertEqual(tokenizer.get_vocab_size(), 6)  # 4 special + 2 tokens
    
    def test_encode_decode(self):
        """Test basic encoding and decoding."""
        vocab = Vocabulary()
        tokenizer = BPETokenizer(vocab=vocab)
        
        # Simple test with special tokens
        text = "[BOS] hello [EOS]"
        
        # For now, BPE tokenizer without merges will tokenize character by character
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        # With character-level tokenization, this should work
        self.assertEqual(decoded, text)
    
    def test_from_pretokenized(self):
        """Test creating tokenizer from pre-tokenized data."""
        tokens = ["hello", "world", "test"]
        tokenizer = BPETokenizer.from_pretokenized(tokens)
        
        self.assertIsNotNone(tokenizer)
        self.assertEqual(tokenizer.get_vocab_size(), 7)  # 4 special + 3 tokens


class TestTokenizerBase(unittest.TestCase):
    """Test base tokenizer functionality."""
    
    def test_special_token_ids(self):
        """Test special token ID properties."""
        vocab = Vocabulary()
        tokenizer = BPETokenizer(vocab=vocab)
        
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertEqual(tokenizer.bos_token_id, 1)
        self.assertEqual(tokenizer.eos_token_id, 2)
        self.assertEqual(tokenizer.unk_token_id, 3)


if __name__ == '__main__':
    unittest.main()
