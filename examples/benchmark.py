#!/usr/bin/env python3
"""
Example: Benchmark Composter

This script benchmarks the Composter model's inference performance.
"""

import torch
import argparse
import os
import sys
import logging
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openahi import (
    Composter, ComposterConfig, InferenceEngine, InferenceConfig,
    BPETokenizer, Vocabulary
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_model(
    model: Composter,
    tokenizer: BPETokenizer,
    device: str,
    prompt: str = "The quick brown fox jumps over the lazy dog",
    num_tokens: int = 100,
    num_runs: int = 10,
    warmup_runs: int = 3,
):
    """Benchmark model inference performance."""
    
    model.eval()
    model.to(device)
    
    # Create inference engine
    inference_config = InferenceConfig(
        device=device,
    )
    engine = InferenceEngine(inference_config, model=model, tokenizer=tokenizer)
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    logger.info(f"Benchmarking with prompt: '{prompt}'")
    logger.info(f"Generating {num_tokens} tokens, {num_runs} runs, {warmup_runs} warmup runs")
    
    # Warmup
    logger.info("Running warmup...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(input_tensor, max_new_tokens=10)
    
    # Benchmark
    logger.info("Running benchmark...")
    times = []
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        
        with torch.no_grad():
            generated = model.generate(input_tensor, max_new_tokens=num_tokens)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        logger.info(f"Run {i+1}: {elapsed:.4f}s ({num_tokens/elapsed:.2f} tokens/s)")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    avg_tokens_per_sec = num_tokens / avg_time
    
    logger.info("\n" + "=" * 50)
    logger.info("Benchmark Results")
    logger.info("=" * 50)
    logger.info(f"Model: Composter 1.00.0")
    logger.info(f"Device: {device}")
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Tokens to generate: {num_tokens}")
    logger.info(f"\nAverage time: {avg_time:.4f}s")
    logger.info(f"Min time: {min_time:.4f}s")
    logger.info(f"Max time: {max_time:.4f}s")
    logger.info(f"Std dev: {std_time:.4f}s")
    logger.info(f"\nAverage tokens/s: {avg_tokens_per_sec:.2f}")
    logger.info(f"Min tokens/s: {num_tokens/min_time:.2f}")
    logger.info(f"Max tokens/s: {num_tokens/max_time:.2f}")
    
    # Get memory usage
    memory_info = engine.get_memory_usage()
    logger.info(f"\nMemory Usage:")
    logger.info(f"  Parameters: {memory_info['param_size_mb']:.2f} MB")
    logger.info(f"  Buffers: {memory_info['buffer_size_mb']:.2f} MB")
    logger.info(f"  Total: {memory_info['total_size_mb']:.2f} MB")
    
    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "memory_mb": memory_info["total_size_mb"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Composter - The first OpenAHI model"
    )
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=512, help="Context length")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    
    # Benchmark arguments
    parser.add_argument("--prompt", type=str, 
                        default="The quick brown fox jumps over the lazy dog",
                        help="Input prompt")
    parser.add_argument("--num-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--warmup-runs", type=int, default=3, help="Number of warmup runs")
    
    # System arguments
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create or load model
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"Loading model from: {args.model_path}")
        
        checkpoint = torch.load(args.model_path, map_location="cpu")
        model_config = checkpoint.get("model_config", ComposterConfig())
        
        model = Composter(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
    else:
        logger.info("Creating new model")
        model_config = ComposterConfig(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
        model = Composter(model_config)
    
    logger.info(f"Model: {model.get_num_params():,} parameters")
    
    # Create tokenizer
    vocab = Vocabulary()
    tokenizer = BPETokenizer(vocab=vocab)
    
    # Run benchmark
    results = benchmark_model(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
    )
    
    logger.info(f"\nBenchmark complete!")


if __name__ == "__main__":
    main()
