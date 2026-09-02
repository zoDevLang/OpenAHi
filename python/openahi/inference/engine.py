"""
Inference Engine

Provides high-level inference API for OpenAHI models.
"""

import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
import torch

from openahi.models.composter import Composter, ComposterConfig
from openahi.tokenizer import Tokenizer


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    model_path: Optional[str] = None
    model_config: Optional[ComposterConfig] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    num_return_sequences: int = 1
    
    # Performance
    batch_size: int = 1
    
    def __post_init__(self):
        if self.model_path is None and self.model_config is None:
            self.model_config = ComposterConfig()


class InferenceEngine:
    """
    High-level inference engine for OpenAHI models.
    
    Provides a simple API for:
    - Loading models
    - Running inference
    - Generating text
    - Benchmarking
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None,
                 model: Optional[Composter] = None,
                 tokenizer: Optional[Tokenizer] = None):
        if config is None:
            config = InferenceConfig()
        
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self._loaded_from_checkpoint = False
        
        # Load model if path provided
        if config.model_path is not None:
            self.load_model(config.model_path)
        elif model is None:
            # Create default model
            if config.model_config is None:
                config.model_config = ComposterConfig()
            self.model = Composter(config.model_config)
            self.model.to(config.device)
    
    def load_model(self, path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            path: Path to model checkpoint
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        # Get model config
        model_config = checkpoint.get("model_config", ComposterConfig())
        
        # Create model
        self.model = Composter(model_config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.config.device)
        
        self._loaded_from_checkpoint = True
        
        # Update config
        self.config.model_path = path
        self.config.model_config = model_config
    
    def set_tokenizer(self, tokenizer: Tokenizer) -> None:
        """Set the tokenizer."""
        self.tokenizer = tokenizer
    
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 eos_token_id: Optional[int] = None) -> List[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_k: Number of top tokens to sample from
            eos_token_id: End-of-sequence token ID
            
        Returns:
            List of generated texts (one per return sequence)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set")
        
        # Use config defaults if not specified
        if max_new_tokens is None:
            max_new_tokens = self.config.max_length
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.config.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_token_id
            )
        
        # Decode
        generated_texts = []
        for i in range(min(self.config.num_return_sequences, generated_ids.shape[0])):
            text = self.tokenizer.decode(generated_ids[i].tolist())
            generated_texts.append(text)
        
        return generated_texts
    
    def batch_generate(self, prompts: List[str], max_new_tokens: Optional[int] = None,
                       temperature: Optional[float] = None,
                       top_k: Optional[int] = None) -> List[str]:
        """
        Generate text from multiple prompts in a batch.
        
        Args:
            prompts: List of input text prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_k: Number of top tokens to sample from
            
        Returns:
            List of generated texts
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set")
        
        if max_new_tokens is None:
            max_new_tokens = self.config.max_length
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        
        self.model.eval()
        
        # Tokenize all prompts
        input_tensors = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt)
            input_tensor = torch.tensor(input_ids, dtype=torch.long).to(self.config.device)
            input_tensors.append(input_tensor)
        
        # Pad to same length
        max_len = max(t.size(0) for t in input_tensors)
        padded_tensors = []
        for t in input_tensors:
            if t.size(0) < max_len:
                padding = torch.zeros(max_len - t.size(0), dtype=torch.long, device=self.config.device)
                padded = torch.cat([t, padding])
            else:
                padded = t
            padded_tensors.append(padded)
        
        input_batch = torch.stack(padded_tensors).to(self.config.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_texts = []
        for i in range(generated_ids.shape[0]):
            text = self.tokenizer.decode(generated_ids[i].tolist())
            generated_texts.append(text)
        
        return generated_texts
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "no_model_loaded"}
        
        return {
            "model_type": "composter",
            "version": "1.00.0",
            "vocab_size": str(self.model.config.vocab_size),
            "context_length": str(self.model.config.context_length),
            "embedding_dim": str(self.model.config.embedding_dim),
            "num_layers": str(self.model.config.num_layers),
            "num_heads": str(self.model.config.num_heads),
            "parameter_count": str(self.model.get_num_params()),
            "device": self.config.device,
            "loaded_from_checkpoint": str(self._loaded_from_checkpoint),
        }
    
    def benchmark(self, prompt: str, num_tokens: int = 100, num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            prompt: Input prompt for benchmarking
            num_tokens: Number of tokens to generate per run
            num_runs: Number of runs to average
            
        Returns:
            Dictionary with benchmark metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set")
        
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.config.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = self.model.generate(input_tensor, max_new_tokens=10)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_tensor,
                    max_new_tokens=num_tokens
                )
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        tokens_per_sec = num_tokens / avg_time
        
        return {
            "avg_time_per_run": avg_time,
            "tokens_per_second": tokens_per_sec,
            "min_time": min(times),
            "max_time": max(times),
            "num_runs": num_runs,
            "num_tokens": num_tokens,
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information."""
        if self.model is None:
            return {"model_loaded": False}
        
        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size = param_size + buffer_size
        
        # Convert to MB
        param_size_mb = param_size / (1024 * 1024)
        buffer_size_mb = buffer_size / (1024 * 1024)
        total_size_mb = total_size / (1024 * 1024)
        
        return {
            "param_size_mb": param_size_mb,
            "buffer_size_mb": buffer_size_mb,
            "total_size_mb": total_size_mb,
            "device": self.config.device,
        }
