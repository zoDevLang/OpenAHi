"""
Composter 1.00.0 Model

A small Transformer language model for the OpenAHI ecosystem.
Architecture includes:
- Token embeddings
- Positional information (learned positional embeddings)
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Residual connections
- Output projection
- Autoregressive generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ComposterConfig:
    """Configuration for Composter model."""
    vocab_size: int = 8192
    context_length: int = 512
    embedding_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # Derived values
    head_dim: int = None
    
    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.embedding_dim // self.num_heads
        assert self.embedding_dim % self.num_heads == 0, \
            f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})"


@dataclass
class ModelConfig:
    """General model configuration wrapper."""
    model_type: str = "composter"
    version: str = "1.00.0"
    composter: Optional[ComposterConfig] = None
    
    def __post_init__(self):
        if self.composter is None:
            self.composter = ComposterConfig()


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(self, config: ComposterConfig):
        super().__init__()
        self.config = config
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.k_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.v_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # Output projection
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(config.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)  # (batch_size, seq_len, embedding_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, embedding_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, embedding_dim)
        
        # Reshape for multi-head attention
        # (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask (autoregressive: can only attend to past tokens)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.embedding_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network with ReLU activation."""
    
    def __init__(self, config: ComposterConfig):
        super().__init__()
        self.config = config
        
        # Hidden dimension is 4x embedding dimension (common in Transformers)
        hidden_dim = config.embedding_dim * 4
        
        self.fc1 = nn.Linear(config.embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """A single transformer block with self-attention and feed-forward network."""
    
    def __init__(self, config: ComposterConfig):
        super().__init__()
        self.config = config
        
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        # Self-attention with residual connection
        attn_output = self.attention(x)
        x = x + self.dropout(attn_output)
        x = self.ln1(x)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.ln2(x)
        
        return x


class Composter(nn.Module):
    """
    Composter 1.00.0 - The first OpenAHI model.
    
    A small Transformer language model for text generation.
    """
    
    def __init__(self, config: Optional[ComposterConfig] = None):
        super().__init__()
        
        if config is None:
            config = ComposterConfig()
        
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # Positional embeddings (learned)
        self.position_embeddings = nn.Embedding(config.context_length, config.embedding_dim)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.final_ln = nn.LayerNorm(config.embedding_dim)
        
        # Output projection (to vocabulary size)
        self.output_proj = nn.Linear(config.embedding_dim, config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        
        # Initialize all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Composter model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Get positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.position_embeddings(positions)
        
        # Combine embeddings
        x = self.dropout(token_embeds + pos_embeds)
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm
        x = self.final_ln(x)
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, 
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Args:
            input_ids: Starting input IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling (1.0 = no temperature scaling)
            top_k: Number of top tokens to sample from (None = all tokens)
            eos_token_id: End-of-sequence token ID (stops generation if generated)
            
        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        batch_size, seq_len = input_ids.shape
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for current sequence
                logits = self.forward(generated[:, -self.config.context_length:])
                
                # Get logits for the last position only
                logits = logits[:, -1, :]  # (batch_size, vocab_size)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    # Get top-k logits and set others to -inf
                    top_logits, _ = torch.topk(logits, top_k, dim=-1)
                    mask = logits >= top_logits[:, [-1]]
                    logits = logits.masked_fill(~mask, float('-inf'))
                
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
                
                # Check for EOS token
                if eos_token_id is not None:
                    if (next_token == eos_token_id).any():
                        break
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we've reached context length
                if generated.shape[1] >= self.config.context_length:
                    break
        
        return generated
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            "config": self.config,
            "model_state_dict": self.state_dict(),
            "model_type": "composter",
            "version": "1.00.0",
        }
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> "Composter":
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location="cpu")
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_config_dict(self) -> dict:
        """Get configuration as dictionary."""
        return {
            "model_type": "composter",
            "version": "1.00.0",
            "vocab_size": self.config.vocab_size,
            "context_length": self.config.context_length,
            "embedding_dim": self.config.embedding_dim,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "dropout": self.config.dropout,
            "parameter_count": self.get_num_params(),
        }
