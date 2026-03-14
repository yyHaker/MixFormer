# MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders

A PyTorch implementation of **MixFormer**, a unified Transformer-style architecture for jointly modeling sequence behavior and feature interactions in recommender systems.

> **Paper**: [MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders](https://arxiv.org/abs/2602.14110)  
> **Authors**: Xu Huang, Hao Zhang, Zhifang Fan, et al. (ByteDance)

## Overview

MixFormer addresses the **co-scaling challenge** in industrial recommender systems: how to efficiently scale both dense feature interactions and user sequence modeling within a single architecture under limited computational budgets.

### Key Features

- **Unified Architecture**: Joint modeling of dense features and user behavior sequences in a single Transformer backbone
- **MixFormer Block**: Three core modules — Query Mixer, Cross Attention, Output Fusion
- **UI-MixFormer**: User-Item decoupled variant for efficient inference via Request Level Batching
- **Training Pipeline**: Complete training with BCE loss, RMSProp optimizer, AUC/LogLoss metrics, checkpointing, and mixed precision support
- **Inference Pipeline**: Standard inference and decoupled inference modes

## Architecture

```
Input → Feature Encoding → [MixFormer Block × L] → Task Head → CTR Prediction

MixFormer Block:
  ├── Query Mixer (QM): HeadMixing + Per-Head SwiGLU FFN
  ├── Cross Attention (CA): Sequence preprocessing + Scaled Dot-Product Attention
  └── Output Fusion (OF): Per-Head SwiGLU FFN
```

### Model Configurations

| Config | N (heads) | L (layers) | D (dim) | Parameters |
|--------|-----------|------------|---------|------------|
| Small  | 16        | 4          | 386     | ~50M       |
| Medium | 16        | 4          | 768     | ~200M      |

## Installation

```bash
# Clone the repository
cd MixFormer

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.10.0
- NumPy >= 1.24.0
- tqdm >= 4.65.0
- scikit-learn >= 1.3.0 (optional, for AUC computation)

## Quick Start

### Training

```bash
# Train MixFormer-small with default settings
python train.py --config small --epochs 10 --batch_size 256

# Train MixFormer-medium with mixed precision
python train.py --config medium --epochs 20 --batch_size 128 --amp

# Train UI-MixFormer (decoupled variant)
python train.py --model_type ui_mixformer --config small --epochs 10

# Train on GPU
python train.py --config small --device cuda --epochs 10

# Train on Apple Silicon
python train.py --config small --device mps --epochs 10
```

### Inference

```bash
# Standard inference with checkpoint
python inference.py --checkpoint checkpoints/best_model.pt --mode standard

# Decoupled inference (UI-MixFormer)
python inference.py --checkpoint checkpoints/best_model.pt --mode decoupled

# Quick test without checkpoint (random weights)
python inference.py --config small --mode standard --no_checkpoint

# UI-MixFormer decoupled inference test
python inference.py --config small --model_type ui_mixformer --mode decoupled --no_checkpoint
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_model.py::TestMixFormer -v

# Run with coverage
python -m pytest tests/ -v --tb=short
```

## Project Structure

```
MixFormer/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── mixformer/
│   ├── __init__.py              # Package exports
│   ├── config.py                # MixFormerConfig (small/medium presets)
│   ├── modules.py               # SwiGLUFFN, HeadMixing, PerHeadSwiGLUFFN
│   ├── layers.py                # QueryMixer, CrossAttention, OutputFusion, MixFormerBlock
│   ├── model.py                 # MixFormer, UIMixFormer, FeatureEncoder, TaskHead
│   └── data.py                  # SyntheticRecDataset, DataLoader factory
├── train.py                     # Training script with Trainer class
├── inference.py                 # Inference script (standard + decoupled)
└── tests/
    └── test_model.py            # Unit tests
```

## Module Details

### SwiGLU FFN
```
SwiGLU(x) = (W₁·x) ⊙ SiLU(W_gate·x)
FFN(x) = W₂ · SwiGLU(x)
```

### Query Mixer
```
P = HeadMixing(Norm(X)) + X          # Cross-head information exchange
q_i = SwiGLUFFN_i(Norm(p_i)) + p_i   # Per-head transformation
```

### Cross Attention
```
h_t = SwiGLUFFN(Norm(s_t)) + s_t      # Sequence preprocessing
k_t^i = W_k^i · h_t^i                 # Key projection
v_t^i = W_v^i · h_t^i                 # Value projection
z_i = Σ softmax(q_i·k_t^i/√D)·v_t^i + q_i  # Scaled attention + residual
```

### Output Fusion
```
o_i = SwiGLUFFN_i(Norm(z_i)) + z_i    # Deep fusion with per-head FFN
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `small` | Model preset (small/medium) |
| `--model_type` | `mixformer` | Model variant (mixformer/ui_mixformer) |
| `--epochs` | `10` | Number of training epochs |
| `--batch_size` | `256` | Batch size |
| `--lr` | `0.01` | Learning rate |
| `--weight_decay` | `1e-5` | Weight decay |
| `--amp` | `false` | Enable mixed precision training |
| `--device` | `auto` | Compute device (cpu/cuda/mps/auto) |
| `--save_dir` | `checkpoints` | Checkpoint save directory |

## Citation

```bibtex
@article{huang2026mixformer,
  title={MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders},
  author={Huang, Xu and Zhang, Hao and Fan, Zhifang and Huang, Yunwen and Wei, Zhuoxing and Chai, Zheng and Ni, Jinan and Zheng, Yuchao and Chen, Qiwei},
  journal={arXiv preprint arXiv:2602.14110},
  year={2026}
}
```

## License

This implementation is for research and educational purposes.
