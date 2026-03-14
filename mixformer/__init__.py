"""
MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders

A PyTorch implementation of MixFormer, a unified Transformer-style architecture
for jointly modeling sequence behavior and feature interactions in recommender systems.

Reference: arXiv:2602.14110
"""

from .config import MixFormerConfig
from .modules import SwiGLUFFN, HeadMixing, PerHeadSwiGLUFFN
from .layers import QueryMixer, CrossAttention, OutputFusion, MixFormerBlock
from .model import FeatureEncoder, TaskHead, MixFormer, UIMixFormer
from .data import SyntheticRecDataset, create_dataloader

__all__ = [
    "MixFormerConfig",
    "SwiGLUFFN",
    "HeadMixing",
    "PerHeadSwiGLUFFN",
    "QueryMixer",
    "CrossAttention",
    "OutputFusion",
    "MixFormerBlock",
    "FeatureEncoder",
    "TaskHead",
    "MixFormer",
    "UIMixFormer",
    "SyntheticRecDataset",
    "create_dataloader",
]

__version__ = "0.1.0"
