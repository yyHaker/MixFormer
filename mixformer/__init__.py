"""
MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders

A PyTorch implementation of MixFormer, a unified Transformer-style architecture
for jointly modeling sequence behavior and feature interactions in recommender systems.

Uses Alibaba UserBehavior dataset (DIN paper) with TorchRec integration.

Reference: arXiv:2602.14110
"""

from .config import MixFormerConfig
from .modules import SwiGLUFFN, HeadMixing, PerHeadSwiGLUFFN, SparseMoE, PerHeadSparseMoE
from .layers import QueryMixer, CrossAttention, OutputFusion, MixFormerBlock
from .model import (
    FeatureEncoder,
    TaskHead,
    MixFormer,
)
from .data import (
    AlibabaDataset,
    create_dataloader,
)

__all__ = [
    "MixFormerConfig",
    "SwiGLUFFN",
    "HeadMixing",
    "PerHeadSwiGLUFFN",
    "SparseMoE",
    "PerHeadSparseMoE",
    "QueryMixer",
    "CrossAttention",
    "OutputFusion",
    "MixFormerBlock",
    "FeatureEncoder",
    "TaskHead",
    "MixFormer",
    "AlibabaDataset",
    "create_dataloader",
]

__version__ = "0.4.0"
