"""
Evaluation Metrics

Provides metrics for evaluating OpenAHI models.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from openahi.models.composter import Composter
from openahi.data.dataset import TokenizedDataset
from openahi.data.dataloader import OpenAHIDataLoader


def compute_perplexity(logits: torch.Tensor, targets: torch.Tensor, 
                       ignore_index: int = 0) -> float:
    """
    Compute perplexity from logits and targets.
    
    Args:
        logits: Model logits of shape (batch_size, seq_len, vocab_size)
        targets: Target token IDs of shape (batch_size, seq_len)
        ignore_index: Token ID to ignore (typically padding)
        
    Returns:
        Perplexity (exp of average loss)
    """
    # Reshape for cross-entropy
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    
    # Mask out ignored indices
    mask = targets != ignore_index
    logits = logits[mask]
    targets = targets[mask]
    
    if len(targets) == 0:
        return float('inf')
    
    # Compute loss
    loss = F.cross_entropy(logits, targets, reduction='mean')
    
    # Perplexity is exp of loss
    return math.exp(loss.item())


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, 
                     ignore_index: int = 0, top_k: int = 1) -> float:
    """
    Compute accuracy from logits and targets.
    
    Args:
        logits: Model logits of shape (batch_size, seq_len, vocab_size)
        targets: Target token IDs of shape (batch_size, seq_len)
        ignore_index: Token ID to ignore
        top_k: Number of top predictions to consider
        
    Returns:
        Accuracy (0.0 to 1.0)
    """
    # Reshape
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    
    # Mask out ignored indices
    mask = targets != ignore_index
    logits = logits[mask]
    targets = targets[mask]
    
    if len(targets) == 0:
        return 0.0
    
    # Get top-k predictions
    if top_k == 1:
        preds = torch.argmax(logits, dim=-1)
    else:
        preds = torch.topk(logits, top_k, dim=-1).indices
    
    # Check if target is in top-k predictions
    if top_k == 1:
        correct = (preds == targets).float().sum()
    else:
        # For top-k, check if target is in any of the top-k positions
        targets_expanded = targets.unsqueeze(-1).expand(-1, top_k)
        correct = (preds == targets_expanded).any(dim=-1).float().sum()
    
    accuracy = correct.item() / len(targets)
    return accuracy


def evaluate_model(model: Composter, dataloader: OpenAHIDataLoader,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Composter model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    total_loss = 0.0
    total_perplexity = 0.0
    total_accuracy = 0.0
    num_batches = 0
    num_tokens = 0
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            
            # Forward pass
            logits = model(input_ids[:, :-1])
            targets = input_ids[:, 1:]
            
            # Compute loss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            
            # Compute metrics
            batch_loss = loss.item()
            batch_perplexity = math.exp(batch_loss)
            batch_accuracy = compute_accuracy(logits, targets)
            
            # Count non-padding tokens
            batch_num_tokens = (targets != 0).sum().item()
            
            total_loss += batch_loss * batch_num_tokens
            total_perplexity += batch_perplexity * batch_num_tokens
            total_accuracy += batch_accuracy * batch_num_tokens
            num_tokens += batch_num_tokens
            num_batches += 1
    
    # Compute averages
    avg_loss = total_loss / num_tokens if num_tokens > 0 else 0.0
    avg_perplexity = total_perplexity / num_tokens if num_tokens > 0 else 0.0
    avg_accuracy = total_accuracy / num_tokens if num_tokens > 0 else 0.0
    
    return {
        "loss": avg_loss,
        "perplexity": avg_perplexity,
        "accuracy": avg_accuracy,
        "num_batches": num_batches,
        "num_tokens": num_tokens,
    }


def compute_bleu(references: List[str], hypotheses: List[str], 
                 max_n: int = 4) -> Dict[str, float]:
    """
    Compute BLEU score between references and hypotheses.
    
    This is a simplified implementation. For production use,
    consider using nltk or sacrebleu.
    
    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        max_n: Maximum n-gram order
        
    Returns:
        Dictionary with BLEU scores for different n-gram orders
    """
    import re
    from collections import Counter
    
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\w+|\p{P}', text.lower())
    
    scores = {}
    
    for n in range(1, max_n + 1):
        total_precision = 0.0
        total_count = 0
        
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = tokenize(ref)
            hyp_tokens = tokenize(hyp)
            
            # Count n-grams in hypothesis
            hyp_ngrams = Counter()
            for i in range(len(hyp_tokens) - n + 1):
                ngram = tuple(hyp_tokens[i:i+n])
                hyp_ngrams[ngram] += 1
            
            # Count maximum n-grams in reference
            ref_ngrams = Counter()
            for i in range(len(ref_tokens) - n + 1):
                ngram = tuple(ref_tokens[i:i+n])
                ref_ngrams[ngram] += 1
            
            # Compute clipped counts
            clipped_count = 0
            for ngram, count in hyp_ngrams.items():
                clipped_count += min(count, ref_ngrams.get(ngram, 0))
            
            # Compute precision
            if len(hyp_tokens) >= n:
                precision = clipped_count / max(1, len(hyp_tokens) - n + 1)
            else:
                precision = 0.0
            
            total_precision += precision
            total_count += 1
        
        scores[f"bleu_{n}"] = total_precision / total_count if total_count > 0 else 0.0
    
    return scores
