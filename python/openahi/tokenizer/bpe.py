"""
Byte-Pair Encoding (BPE) Tokenizer

A simple BPE tokenizer implementation for OpenAHI.
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from openahi.tokenizer.base import Tokenizer, Vocabulary


class BPETokenizer(Tokenizer):
    """
    Byte-Pair Encoding tokenizer.
    
    Implements BPE algorithm for subword tokenization.
    """
    
    def __init__(self, vocab: Optional[Vocabulary] = None, merges: Optional[Dict[Tuple[str, str], str]] = None):
        super().__init__(vocab)
        
        if merges is None:
            merges = {}
        self.merges = merges  # (pair) -> merged_token
        self.pair_to_id = {}  # For faster lookup
        
        # Build reverse merges for splitting
        self._build_reverse_merges()
    
    def _build_reverse_merges(self):
        """Build reverse merge dictionary for token splitting."""
        self.reverse_merges = defaultdict(list)
        for (a, b), merged in self.merges.items():
            self.reverse_merges[merged].append((a, b))
    
    def _get_base_tokens(self, text: str) -> List[str]:
        """
        Get base tokens (characters + pre-defined tokens) from text.
        
        For simplicity, we start with character-level tokens.
        """
        # Basic whitespace tokenization
        tokens = []
        for char in text:
            tokens.append(char)
        return tokens
    
    def _merge_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply BPE merges to tokens.
        
        Greedily merge pairs until no more merges are possible.
        """
        if not self.merges:
            return tokens
        
        # Build a set of all possible merge pairs
        merge_pairs = set(self.merges.keys())
        
        # Iteratively merge
        changed = True
        while changed:
            changed = False
            new_tokens = []
            i = 0
            
            while i < len(tokens):
                # Check if current and next token form a mergeable pair
                if i + 1 < len(tokens):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in merge_pairs:
                        merged = self.merges[pair]
                        new_tokens.append(merged)
                        i += 2
                        changed = True
                        continue
                
                new_tokens.append(tokens[i])
                i += 1
            
            tokens = new_tokens
        
        return tokens
    
    def _split_token(self, token: str) -> List[str]:
        """
        Split a token back into its constituent parts.
        
        Used for decoding.
        """
        if token in self.reverse_merges:
            # Return one of the splits (simplified)
            splits = self.reverse_merges[token]
            if splits:
                a, b = splits[0]
                return [a, b]
        return [token]
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using BPE.
        
        Returns list of token strings.
        """
        # Get base tokens
        tokens = self._get_base_tokens(text)
        
        # Apply merges
        tokens = self._merge_tokens(tokens)
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text into token IDs."""
        tokens = self._tokenize(text)
        token_ids = [self.vocab.get_id(token) for token in tokens]
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs into text."""
        tokens = [self.vocab.get_token(tid) for tid in token_ids]
        
        # For simple BPE, just concatenate
        # More sophisticated decoders would handle spacing, etc.
        text = ''.join(tokens)
        
        # Clean up any special token markers
        text = text.replace('\n', ' ')
        return text
    
    def train(self, texts: List[str], vocab_size: int = 8192, 
              min_frequency: int = 2, special_tokens: Optional[Dict[str, str]] = None):
        """
        Train BPE tokenizer on a corpus of texts.
        
        Args:
            texts: List of text strings to train on
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for a pair to be merged
            special_tokens: Special tokens to include
        """
        # Initialize with special tokens
        if special_tokens:
            self.vocab = Vocabulary.from_tokens([], special_tokens)
        
        # Count initial token frequencies
        all_tokens = []
        for text in texts:
            all_tokens.extend(self._get_base_tokens(text))
        
        # Count frequencies
        freq = defaultdict(int)
        for token in all_tokens:
            freq[token] += 1
        
        # Add all unique tokens to vocabulary initially
        for token in freq:
            self.vocab.add_token(token)
        
        # Get all possible pairs
        def get_pairs(tokens: List[str]) -> Dict[Tuple[str, str], int]:
            pairs = defaultdict(int)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += 1
            return pairs
        
        # Collect all pairs from all texts
        all_pairs = defaultdict(int)
        for text in texts:
            tokens = self._get_base_tokens(text)
            pairs = get_pairs(tokens)
            for pair, count in pairs.items():
                all_pairs[pair] += count
        
        # Perform merges until we reach target vocabulary size
        while len(self.vocab) < vocab_size and all_pairs:
            # Find the most frequent pair
            best_pair = max(all_pairs.items(), key=lambda x: x[1])
            pair, count = best_pair
            
            if count < min_frequency:
                break
            
            # Create merged token
            merged_token = pair[0] + pair[1]
            
            # Add to merges
            self.merges[pair] = merged_token
            
            # Add to vocabulary
            self.vocab.add_token(merged_token)
            
            # Update reverse merges
            self._build_reverse_merges()
            
            # Remove this pair from consideration
            del all_pairs[pair]
            
            # Update counts for new pairs involving the merged token
            new_pairs = defaultdict(int)
            for (a, b), cnt in all_pairs.items():
                if a == pair[0] and b == pair[1]:
                    continue
                # Check if this pair now involves the merged token
                # (This is simplified; a full implementation would track this properly)
            
            # For simplicity, just re-count all pairs
            all_pairs = defaultdict(int)
            for text in texts:
                tokens = self._get_base_tokens(text)
                tokens = self._merge_tokens(tokens)
                pairs = get_pairs(tokens)
                for p, c in pairs.items():
                    all_pairs[p] += c
    
    def _save(self, filepath: str):
        """Save BPE-specific data."""
        data = {
            "merges": {f"{a},{b}": m for (a, b), m in self.merges.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> "BPETokenizer":
        """Load BPE tokenizer from file."""
        # Load vocabulary
        vocab_path = filepath + ".vocab.json"
        vocab = Vocabulary.load(vocab_path)
        
        # Load merges
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        merges = {}
        for key, value in data["merges"].items():
            a, b = key.split(',')
            merges[(a, b)] = value
        
        tokenizer = cls(vocab=vocab, merges=merges)
        return tokenizer
    
    @classmethod
    def from_pretokenized(cls, tokens: List[str], vocab: Optional[Vocabulary] = None) -> "BPETokenizer":
        """
        Create a BPE tokenizer from pre-tokenized data.
        
        This is useful for creating a simple character-level or word-level tokenizer
        without training BPE merges.
        """
        if vocab is None:
            vocab = Vocabulary.from_tokens(tokens)
        
        tokenizer = cls(vocab=vocab)
        return tokenizer
