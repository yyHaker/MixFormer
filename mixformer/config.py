"""
MixFormer 模型配置模块。

使用 dataclass 定义模型超参数，提供 small/medium 两种预设配置。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MixFormerConfig:
    """MixFormer 模型配置。

    Attributes:
        num_heads: N — 头数，非序列特征拆分的子向量数量
        num_layers: L — MixFormer Block 堆叠层数
        hidden_dim: D — 每个头的隐藏维度
        seq_length: T — 用户行为序列最大长度
        num_non_seq_features: M — 非序列特征数量
        feature_embed_dim: 每个非序列特征的嵌入维度
        seq_feature_embed_dim: 序列中每个物品的嵌入维度（等于 N * D）
        num_items: 物品词表大小
        num_users: 用户词表大小
        vocab_sizes: 各类别特征的词表大小映射 {feature_name: vocab_size}
        ffn_multiplier: SwiGLU FFN 中间维度倍数 (≈8/3)
        dropout: Dropout 概率
        user_heads: N_U — 用户侧头数 (UI-MixFormer)
        item_heads: N_G — 物品侧头数 (UI-MixFormer)
        task_head_hidden_dims: 任务头 MLP 的隐藏层维度列表
    """

    # 核心架构参数
    num_heads: int = 16
    num_layers: int = 4
    hidden_dim: int = 386
    seq_length: int = 50

    # 特征参数
    num_non_seq_features: int = 50
    feature_embed_dim: int = 16
    seq_feature_embed_dim: Optional[int] = None  # 默认自动计算为 num_heads * hidden_dim
    num_items: int = 10000
    num_users: int = 10000
    vocab_sizes: Optional[Dict[str, int]] = None

    # FFN 参数
    ffn_multiplier: float = 2.667  # ≈ 8/3
    dropout: float = 0.0

    # UI-MixFormer 解耦参数
    user_heads: int = 8
    item_heads: int = 8

    # 任务头参数
    task_head_hidden_dims: Optional[list] = None

    def __post_init__(self):
        """参数校验和自动计算。"""
        # 自动计算序列特征嵌入维度
        if self.seq_feature_embed_dim is None:
            self.seq_feature_embed_dim = self.num_heads * self.hidden_dim

        # 设置默认任务头隐藏层
        if self.task_head_hidden_dims is None:
            self.task_head_hidden_dims = [256, 128]

        # 设置默认词表大小
        if self.vocab_sizes is None:
            self.vocab_sizes = self._default_vocab_sizes()

        self._validate()

    def _validate(self):
        """校验配置参数的合法性。"""
        assert self.num_heads > 0, f"num_heads must be positive, got {self.num_heads}"
        assert self.num_layers > 0, f"num_layers must be positive, got {self.num_layers}"
        assert self.hidden_dim > 0, f"hidden_dim must be positive, got {self.hidden_dim}"
        assert self.seq_length > 0, f"seq_length must be positive, got {self.seq_length}"
        assert self.num_non_seq_features > 0, (
            f"num_non_seq_features must be positive, got {self.num_non_seq_features}"
        )
        assert self.feature_embed_dim > 0, (
            f"feature_embed_dim must be positive, got {self.feature_embed_dim}"
        )
        assert self.user_heads + self.item_heads == self.num_heads, (
            f"user_heads ({self.user_heads}) + item_heads ({self.item_heads}) "
            f"must equal num_heads ({self.num_heads})"
        )
        assert self.seq_feature_embed_dim == self.num_heads * self.hidden_dim, (
            f"seq_feature_embed_dim ({self.seq_feature_embed_dim}) must equal "
            f"num_heads * hidden_dim ({self.num_heads * self.hidden_dim})"
        )

    def _default_vocab_sizes(self) -> Dict[str, int]:
        """生成默认的特征词表大小映射。"""
        vocab = {
            "user_id": self.num_users,
            "item_id": self.num_items,
        }
        # 为其他非序列特征生成默认词表大小
        remaining = self.num_non_seq_features - 2
        for i in range(remaining):
            vocab[f"feature_{i}"] = 100  # 默认词表大小
        return vocab

    @property
    def total_embed_dim(self) -> int:
        """非序列特征嵌入拼接后的总维度 D_ns = M * feature_embed_dim。"""
        return self.num_non_seq_features * self.feature_embed_dim

    @property
    def head_input_dim(self) -> int:
        """每个头的输入维度 d = D_ns / N。"""
        d_ns = self.total_embed_dim
        assert d_ns % self.num_heads == 0, (
            f"total_embed_dim ({d_ns}) must be divisible by num_heads ({self.num_heads})"
        )
        return d_ns // self.num_heads

    @property
    def ffn_hidden_dim(self) -> int:
        """SwiGLU FFN 的中间维度，向上取整到最近的 64 的倍数。"""
        raw_dim = int(self.hidden_dim * self.ffn_multiplier)
        return int(math.ceil(raw_dim / 64) * 64)

    @property
    def model_dim(self) -> int:
        """模型总维度 N * D。"""
        return self.num_heads * self.hidden_dim

    @classmethod
    def small(cls) -> MixFormerConfig:
        """MixFormer-small 预设: N=16, L=4, D=386。"""
        return cls(
            num_heads=16,
            num_layers=4,
            hidden_dim=386,
            seq_length=50,
            num_non_seq_features=48,  # 确保 M * embed_dim 能被 N 整除: 48*16=768, 768/16=48
            feature_embed_dim=16,
            num_items=10000,
            num_users=10000,
            ffn_multiplier=2.667,
            dropout=0.0,
            user_heads=8,
            item_heads=8,
            task_head_hidden_dims=[256, 128],
        )

    @classmethod
    def medium(cls) -> MixFormerConfig:
        """MixFormer-medium 预设: N=16, L=4, D=768。"""
        return cls(
            num_heads=16,
            num_layers=4,
            hidden_dim=768,
            seq_length=50,
            num_non_seq_features=48,
            feature_embed_dim=16,
            num_items=10000,
            num_users=10000,
            ffn_multiplier=2.667,
            dropout=0.0,
            user_heads=8,
            item_heads=8,
            task_head_hidden_dims=[512, 256],
        )

    def __repr__(self) -> str:
        return (
            f"MixFormerConfig(\n"
            f"  num_heads={self.num_heads}, num_layers={self.num_layers}, "
            f"hidden_dim={self.hidden_dim},\n"
            f"  seq_length={self.seq_length}, model_dim={self.model_dim},\n"
            f"  num_non_seq_features={self.num_non_seq_features}, "
            f"feature_embed_dim={self.feature_embed_dim},\n"
            f"  total_embed_dim={self.total_embed_dim}, "
            f"head_input_dim={self.head_input_dim},\n"
            f"  ffn_hidden_dim={self.ffn_hidden_dim}, dropout={self.dropout},\n"
            f"  user_heads={self.user_heads}, item_heads={self.item_heads}\n"
            f")"
        )
