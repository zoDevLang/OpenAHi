# OpenAHI - Open Artificial Hyper Intelligence

> **OpenAHI is a model ecosystem, not a chatbot.**

**Composter 1.00.0** is the first OpenAHI model.

## Overview

OpenAHI is an open-source Artificial Hyper Intelligence ecosystem created by ZoDev. It provides a complete stack for downloading, installing, running, integrating, modifying, and building AI models.

The first milestone is: **Build a real, working Composter 1.00.0 model and the infrastructure required to run it.**

## Key Principles

1. **OpenAHI is a model ecosystem** - Not a chatbot wrapper
2. **The intelligence comes from OpenAHI models** - Not from external AI APIs
3. **Composter is an actual AI model** - That developers can run and integrate into their own software

## Architecture

OpenAHI uses a multi-language architecture:

```
                         OpenAHI
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
          Python           Rust           C++
             │              │              │
         Training       Runtime/API     Fast inference
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                     Composter Model
                            │
                            ▼
                       OpenAHI SDK
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          Python        TypeScript        Rust
            SDK             SDK            SDK
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                       Applications
```

## Repository Structure

```
OpenAHI/
├── models/                    # Model definitions and checkpoints
│   └── composter/
│       └── 1.00.0/
│           ├── model.json     # Model metadata
│           ├── config.json    # Model configuration
│           └── composter_1.00.0.pt  # Model checkpoint
│
├── python/                    # Python components
│   └── openahi/
│       ├── __init__.py
│       ├── models/           # Model implementations
│       │   └── composter.py  # Composter model
│       ├── training/         # Training pipeline
│       │   ├── __init__.py
│       │   ├── trainer.py    # Trainer class
│       │   └── optimizer.py  # Optimizer utilities
│       ├── tokenizer/        # Tokenizer implementations
│       │   ├── __init__.py
│       │   ├── base.py       # Base tokenizer
│       │   └── bpe.py        # BPE tokenizer
│       ├── inference/        # Inference engine
│       │   ├── __init__.py
│       │   └── engine.py     # Inference engine
│       ├── data/             # Data utilities
│       │   ├── __init__.py
│       │   ├── dataset.py    # Dataset classes
│       │   └── dataloader.py # DataLoader
│       └── evaluation/       # Evaluation metrics
│           ├── __init__.py
│           └── metrics.py    # Metrics
│
├── cpp/                      # C++ components
│   └── inference/            # C++ inference engine
│       ├── CMakeLists.txt
│       ├── include/
│       │   └── openahi/
│       │       └── inference/
│       │           ├── tensor.h      # Tensor class
│       │           └── model.h      # Model class
│       ├── tensor.cpp        # Tensor implementation
│       └── model.cpp         # Model implementation
│
├── rust/                     # Rust components
│   ├── Cargo.toml            # Workspace Cargo.toml
│   ├── runtime/              # OpenAHI Runtime
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs       # Runtime library
│   │       ├── error.rs     # Error types
│   │       ├── config.rs    # Configuration
│   │       ├── model.rs     # Model types
│   │       ├── runtime.rs   # Runtime implementation
│   │       └── inference.rs # Inference types
│   ├── cli/                  # Command-line interface
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── main.rs      # CLI main
│   ├── pkg/                  # Package management
│   │   └── ...
│   └── sdk/                  # Rust SDK
│       └── ...
│
├── typescript/               # TypeScript components
│   ├── web/                  # Web platform
│   │   └── ...
│   ├── sdk/                  # TypeScript SDK
│   │   └── ...
│   └── apps/                 # Applications
│       └── ...
│
├── configs/                   # Configuration files
│   ├── training.json
│   ├── inference.json
│   └── runtime.json
│
├── datasets/                  # Datasets
│   └── ...
│
├── examples/                 # Example scripts
│   ├── train_composter.py
│   ├── generate.py
│   └── benchmark.py
│
├── tests/                    # Tests
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   ├── test_training.py
│   ├── test_inference.py
│   └── test_evaluation.py
│
├── docs/                     # Documentation
│   └── ...
│
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
└── .gitignore
```

## Composter 1.00.0

Composter is the first OpenAHI model. It's a small Transformer language model with:

- **Token embeddings**
- **Positional information** (learned positional embeddings)
- **Multi-head self-attention**
- **Feed-forward networks**
- **Layer normalization**
- **Residual connections**
- **Output projection**
- **Autoregressive generation**

### Architecture Configuration

```json
{
  "vocab_size": 8192,
  "context_length": 512,
  "embedding_dim": 512,
  "num_layers": 6,
  "num_heads": 8,
  "dropout": 0.1
}
```

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/zoDevLang/OpenAHi.git
cd OpenAHi

# Install Python dependencies
pip install -e ./python
```

### Train Composter 1.00.0

```bash
# Train with default configuration
python examples/train_composter.py

# Train with custom configuration
python examples/train_composter.py \
    --vocab-size 4096 \
    --embedding-dim 256 \
    --num-layers 4 \
    --num-heads 4 \
    --batch-size 16 \
    --num-epochs 5
```

### Generate Text

```bash
# Generate with default model
python examples/generate.py --prompt "The quick brown fox"

# Generate with custom model
python examples/generate.py \
    --model-path ./checkpoints/composter_000001_final.pt \
    --prompt "OpenAHI is" \
    --max-tokens 50 \
    --temperature 0.7
```

### Benchmark

```bash
# Run benchmark
python examples/benchmark.py --num-runs 10 --num-tokens 100
```

## Rust CLI

```bash
# Build the CLI
cd rust/cli
cargo build --release

# Run commands
./target/release/openahi info
./target/release/openahi models
./target/release/openahi install composter@1.00.0
./target/release/openahi run composter@1.00.0
./target/release/openahi generate --prompt "Hello world" composter@1.00.0
./target/release/openahi evaluate composter@1.00.0
```

## Python API

```python
from openahi import Composter, ComposterConfig, Trainer, TrainingConfig
from openahi import create_tiny_dataset, BPETokenizer, Vocabulary

# Create model
config = ComposterConfig()
model = Composter(config)

# Train
train_config = TrainingConfig()
trainer = Trainer(train_config, model)

dataset = create_tiny_dataset()
tokenizer = BPETokenizer()

# Tokenize and train
# ... (see examples/train_composter.py for full example)

# Generate
output = model.generate(input_ids, max_new_tokens=50)
```

## Rust API

```rust
use openahi_runtime::{OpenAHIRuntime, RuntimeConfig, InferenceConfig};

// Create runtime
let runtime = OpenAHIRuntime::new()?;

// Load model
let model = runtime.load_model("composter", "1.00.0")?;

// Generate text
let result = runtime.generate("composter", "1.00.0", "Hello world", None)?;

println!("{}", result);
```

## Important Notes

1. **No External AI APIs**: OpenAHI does not use external AI providers (Mistral, OpenAI, etc.) as the intelligence engine. The actual model is OpenAHI's own Composter.

2. **Model Ecosystem**: OpenAHI is designed as a model ecosystem where you can download, install, run, integrate, modify, and build models.

3. **Prototype**: Composter 1.00.0 is a small prototype model, not a frontier model. It's designed for demonstration and development purposes.

4. **Extensible**: The architecture is modular and designed to be extended with new models, tokenizers, and features.

## License

OpenAHI is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Security

See [SECURITY.md](SECURITY.md) for security information.

## Attribution

OpenAHI is created by **ZoDev**.

When using OpenAHI, please provide appropriate attribution according to the license.

---

**OpenAHI - Open Artificial Hyper Intelligence**

*Created by ZoDev*
