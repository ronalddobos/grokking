# Grokking Transformer

A clean reproduction of Neel Nanda's grokking experiments, implementing a simple transformer model that demonstrates the grokking phenomenon on modular arithmetic tasks.

## Overview

This project is a modular, well-structured reproduction of the grokking experiments originally implemented by Neel Nanda. It implements a minimal transformer architecture to study grokking - the phenomenon where neural networks suddenly transition from memorization to generalization well after achieving perfect training accuracy. The model learns modular arithmetic operations (addition by default) over finite fields.

The codebase has been refactored from the original implementation to provide better modularity, type safety, and maintainability while preserving the core experimental setup and results.

## Original Work

This reproduction is based on:
- **Paper**: "Progress measures for grokking via mechanistic interpretability" by Neel Nanda et al. ([arXiv:2301.05217](https://arxiv.org/abs/2301.05217))
- **Code**: [Neel Nanda's original implementation](https://github.com/neelnanda-io/Grokking)
- **Interactive notebook**: [Colab notebook](https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20)

## Project Structure

```
grokking/
├── src/
│   ├── config.py      # Configuration dataclass
│   ├── model.py       # SimpleTransformer implementation
│   ├── trainer.py     # Training logic
│   ├── utils.py       # Checkpoint utilities
│   ├── plotting.py    # Visualization functions
│   └── main.py        # Entry point
├── logs/              # Training logs (created automatically)
├── models/            # Model checkpoints (created automatically)
└── README.md
```

## Features

- **Resumable Training**: Automatic checkpoint saving and loading
- **Continuous Logging**: Training metrics saved every 100 epochs
- **Visualization**: Automatic plotting of training curves
- **Configurable**: Easy parameter adjustment through Config class
- **Type Annotated**: Full type hints for better code quality

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib

## Usage

### Basic Training

```python
from src.config import Config
from src.trainer import train_grokking

# Create configuration
config = Config(
    tag='experiment_1',
    p=113,              # Prime number for modular arithmetic
    num_epochs=50000,
    lr=1e-3
)

# Start training
train_grokking(config, resume=True)
```

### Configuration Options

Key parameters in the `Config` class:

- `tag`: Experiment identifier (used for saving logs/models)
- `p`: Prime number for modular arithmetic (default: 113)
- `d_model`: Model dimension (default: 128)
- `num_heads`: Number of attention heads (default: 4)
- `lr`: Learning rate (default: 1e-3)
- `weight_decay`: Weight decay for regularization (default: 1.0)
- `frac_train`: Fraction of data for training (default: 0.3)
- `num_epochs`: Total training epochs (default: 50000)
- `seed`: Random seed for reproducibility (default: 0)

### Resuming Training

The training automatically resumes from the last checkpoint if available:

```python
# This will continue from the last saved checkpoint
train_grokking(config, resume=True)

# Force fresh start (ignores existing checkpoints)
train_grokking(config, resume=False)
```

### Plotting Results

```python
from src.plotting import plot_training_curves

# Plot training curves for an experiment
config = Config(tag='experiment_1')
plot_training_curves(config, show_plot=True)
```

## Model Architecture

The `SimpleTransformer` implements:

- Token and positional embeddings
- Single-layer self-attention with causal masking
- MLP with ReLU activation
- Residual connections
- Unembedding to vocabulary logits

## Training Process

1. **Data Generation**: Creates all possible (a, b, result) pairs for modular arithmetic
2. **Train/Test Split**: Uses `frac_train` to determine the split (default 30% training)
3. **Training Loop**: 
   - Computes loss and accuracy on both splits
   - Logs metrics every 100 epochs
   - Saves checkpoints every 1000 epochs
   - Prints progress every 1000 epochs
4. **Automatic Plotting**: Generates training curves after completion

## Grokking Phenomenon

The model typically exhibits:

1. **Phase 1**: Rapid memorization of training data (high train accuracy, low test accuracy)
2. **Phase 2**: Extended period of apparent stagnation
3. **Phase 3**: Sudden generalization breakthrough (test accuracy rapidly improves)

The timing and occurrence of grokking depend on:
- Weight decay strength
- Learning rate
- Model capacity
- Dataset size and split

## Output Files

For each experiment with `tag='experiment_name'`:

```
logs/experiment_name/
├── config.json           # Experiment configuration
├── training_log.jsonl    # Metrics for each logged epoch
└── training_curves.png   # Loss and accuracy plots

models/experiment_name/
└── checkpoint.pth        # Latest model checkpoint
```

## Example

```python
from src.config import Config
from src.trainer import train_grokking

# Small-scale experiment
config = Config(
    tag='quick_test',
    p=47,
    num_epochs=10000,
    d_model=64
)

train_grokking(config)
```

This will train a model on modular addition with prime p=47, save all results to `logs/quick_test/` and `models/quick_test/`, and display training curves upon completion.
