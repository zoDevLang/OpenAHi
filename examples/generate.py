#!/usr/bin/env python3
"""
Example: Generate text with Composter

This script demonstrates how to use the Composter model for text generation.
"""

import torch
import argparse
import os
import sys
import logging

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


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with Composter - The first OpenAHI model"
    )
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default=None, 
                        help="Path to model checkpoint")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=512, help="Context length")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    
    # Generation arguments
    parser.add_argument("--prompt", type=str, default="The quick brown fox",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    
    # System arguments
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--batch", action="store_true", help="Batch mode (multiple prompts)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
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
        
        # Load checkpoint to get config
        checkpoint = torch.load(args.model_path, map_location="cpu")
        model_config = checkpoint.get("model_config", ComposterConfig())
        
        # Create model
        model = Composter(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
    else:
        # Create new model
        logger.info("Creating new model")
        model_config = ComposterConfig(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
        model = Composter(model_config)
        model.to(device)
    
    logger.info(f"Model loaded with {model.get_num_params():,} parameters")
    
    # Create inference config
    inference_config = InferenceConfig(
        max_length=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    
    # Create tokenizer
    vocab = Vocabulary()
    tokenizer = BPETokenizer(vocab=vocab)
    
    # Create inference engine
    engine = InferenceEngine(inference_config, model=model, tokenizer=tokenizer)
    
    if args.interactive:
        # Interactive mode
        logger.info("Interactive mode - Type 'quit' or 'exit' to end")
        
        while True:
            try:
                prompt = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            try:
                result = engine.generate(prompt, max_new_tokens=args.max_tokens)
                print(result[0])
            except Exception as e:
                logger.error(f"Error: {e}")
    
    elif args.batch:
        # Batch mode
        logger.info("Batch mode - Enter prompts (one per line, empty line to finish)")
        
        prompts = []
        while True:
            try:
                prompt = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            
            if prompt.strip():
                prompts.append(prompt)
            else:
                break
        
        if prompts:
            results = engine.batch_generate(prompts, max_new_tokens=args.max_tokens)
            for i, (prompt, result) in enumerate(zip(prompts, results)):
                logger.info(f"\nPrompt {i+1}: {prompt}")
                logger.info(f"Result {i+1}: {result}")
    
    else:
        # Single prompt mode
        logger.info(f"Generating from prompt: {args.prompt}")
        
        result = engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        
        logger.info(f"\n{result[0]}")


if __name__ == "__main__":
    main()
