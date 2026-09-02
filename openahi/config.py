from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.1


DEFAULT_CONFIG = ModelConfig()
