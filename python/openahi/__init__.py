"""
OpenAHI - Open Artificial Hyper Intelligence

A model ecosystem created by ZoDev.
Composter 1.00.0 is the first OpenAHI model.

OpenAHI is a model ecosystem, not a chatbot.
"""

__version__ = "0.1.0"
__author__ = "ZoDev"
__project__ = "OpenAHI"

from openahi.models import Composter, ComposterConfig, ModelConfig
from openahi.tokenizer import Tokenizer, BPETokenizer, Vocabulary
from openahi.training import Trainer, TrainingConfig
from openahi.inference import InferenceEngine, InferenceConfig
from openahi.data import Dataset, TextDataset, TokenizedDataset, create_tiny_dataset
from openahi.evaluation import evaluate_model, compute_perplexity, compute_accuracy

__all__ = [
    "Composter",
    "ComposterConfig",
    "ModelConfig",
    "Tokenizer",
    "BPETokenizer",
    "Vocabulary",
    "Trainer",
    "TrainingConfig",
    "InferenceEngine",
    "InferenceConfig",
    "Dataset",
    "TextDataset",
    "TokenizedDataset",
    "create_tiny_dataset",
    "evaluate_model",
    "compute_perplexity",
    "compute_accuracy",
]
