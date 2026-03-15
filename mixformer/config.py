"""
MixFormer 模型配置模块。

使用 dataclass 定义模型超参数，默认使用 Alibaba UserBehavior 数据集配置。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MixFormerConfig:
    """MixFormer 模型配置。

    Attributes:
        num_heads: N — 头数，非序列特征拆分的子向量数量
        num_layers: L — MixFormer Block 堆叠层数
        hidden_dim: D — 每个头的隐藏维度
        seq_length: T — 用户行为序列最大长度
        num_non_seq_features: M — 非序列特征数量（虚拟，用于头数整除）
        feature_embed_dim: 每个非序列特征的嵌入维度
        seq_feature_embed_dim: 序列中每个物品的嵌入维度（等于 N * D）
        num_items: 物品词表大小
        num_users: 用户词表大小
        num_categories: 商品类目数量
        ffn_multiplier: SwiGLU FFN 中间维度倍数 (≈8/3)
        dropout: Dropout 概率
        user_heads: N_U — 用户侧头数 (UI-MixFormer)
        item_heads: N_G — 物品侧头数 (UI-MixFormer)
        task_head_hidden_dims: 任务头 MLP 的隐藏层维度列表

        # 稀疏特征嵌入相关
        sparse_feature_names: 稀疏特征名列表
        sparse_vocab_sizes: 稀疏特征词表大小列表（与 sparse_feature_names 对应）
        sparse_embed_dim: EmbeddingBag 的嵌入维度
        use_torchrec: 是否使用 TorchRec 管理嵌入（否则用 nn.EmbeddingBag fallback）
        target_item_mlp_dims: 目标物品特征 MLP 的隐藏层维度列表
    """

    # 核心架构参数
    num_heads: int = 8
    num_layers: int = 3
    hidden_dim: int = 64
    seq_length: int = 50

    # 特征参数
    num_non_seq_features: int = 8
    feature_embed_dim: int = 16
    seq_feature_embed_dim: Optional[int] = None  # 默认自动计算为 num_heads * hidden_dim
    num_items: int = 4200000
    num_users: int = 1000000
    num_categories: int = 10000

    # FFN 参数
    ffn_multiplier: float = 2.667  # ≈ 8/3
    dropout: float = 0.1

    # UI-MixFormer 解耦参数
    user_heads: int = 4
    item_heads: int = 4

    # MoE (Mixture of Experts) 参数
    use_moe: bool = True           # 是否在 OutputFusion 中使用 Sparse MoE 替代 FFN
    num_experts: int = 4           # 专家数量
    num_active_experts: int = 2    # Top-K: 每次激活的专家数量
    moe_aux_loss_weight: float = 0.01  # 负载均衡辅助损失权重

    # 任务头参数
    task_head_hidden_dims: Optional[list] = None

    # 稀疏特征嵌入相关
    sparse_feature_names: Optional[List[str]] = None
    sparse_vocab_sizes: Optional[List[int]] = None
    sparse_embed_dim: int = 64
    use_torchrec: bool = True
    target_item_mlp_dims: Optional[List[int]] = None

    def __post_init__(self):
        """参数校验和自动计算。"""
        # 自动计算序列特征嵌入维度
        if self.seq_feature_embed_dim is None:
            self.seq_feature_embed_dim = self.num_heads * self.hidden_dim

        # 设置默认任务头隐藏层
        if self.task_head_hidden_dims is None:
            self.task_head_hidden_dims = [256, 128]

        # 设置默认稀疏特征配置
        if self.sparse_feature_names is None:
            self.sparse_feature_names = ["item_id", "category_id"]
        if self.sparse_vocab_sizes is None:
            self.sparse_vocab_sizes = [self.num_items, self.num_categories]

        # 设置默认目标物品 MLP 维度
        if self.target_item_mlp_dims is None:
            self.target_item_mlp_dims = [256]

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
    def default(cls, num_items: int = 4200000, num_categories: int = 10000,
                num_users: int = 1000000) -> MixFormerConfig:
        """默认配置（适配 Alibaba UserBehavior 数据集）。

        数据集:
            - 来源: 天池 Alibaba UserBehavior (DIN 论文 1706.06978)
            - ~1亿条用户行为记录
            - 字段: user_id, item_id, category_id, behavior_type, timestamp
            - 用户行为序列 → 序列特征 (CrossAttention 的 KV)
            - 目标物品 (item_id + category_id) → 非序列 Query 特征

        架构适配:
            - 目标物品的 item_embedding + category_embedding 拼接 → MLP 投影到 (N, D) 作为 Query
            - 历史行为序列中每个物品的 item_embedding + category_embedding 拼接 → 投影到 N*D 作为序列
            - 使用 TorchRec EmbeddingBagCollection 管理 item_id 和 category_id 嵌入表
        """
        N = 8      # 头数（适中，适合中等规模数据集）
        D = 64     # 每头维度
        embed_dim = 64  # 稀疏特征嵌入维度

        # 非序列特征数 M: 需要满足 M * feature_embed_dim % N == 0
        M = 8  # 虚拟非序列特征数，确保 M * feature_embed_dim 能被 N 整除
        feat_embed_dim = 16  # M * 16 = 128, 128 / 8 = 16

        return cls(
            num_heads=N,
            num_layers=3,
            hidden_dim=D,
            seq_length=50,
            num_non_seq_features=M,
            feature_embed_dim=feat_embed_dim,
            num_items=num_items,
            num_users=num_users,
            ffn_multiplier=2.667,
            dropout=0.1,
            user_heads=N // 2,
            item_heads=N // 2,
            task_head_hidden_dims=[256, 128],
            num_categories=num_categories,
            sparse_feature_names=["item_id", "category_id"],
            sparse_vocab_sizes=[num_items, num_categories],
            sparse_embed_dim=embed_dim,
            use_torchrec=True,
            target_item_mlp_dims=[256],
        )

    @classmethod
    def medium(cls, num_items: int = 4200000, num_categories: int = 10000,
               num_users: int = 1000000) -> MixFormerConfig:
        """Medium 配置: N=8, L=4, D=128。"""
        N = 8
        D = 128
        embed_dim = 64

        M = 8
        feat_embed_dim = 16

        return cls(
            num_heads=N,
            num_layers=4,
            hidden_dim=D,
            seq_length=50,
            num_non_seq_features=M,
            feature_embed_dim=feat_embed_dim,
            num_items=num_items,
            num_users=num_users,
            ffn_multiplier=2.667,
            dropout=0.1,
            user_heads=N // 2,
            item_heads=N // 2,
            task_head_hidden_dims=[512, 256],
            num_categories=num_categories,
            sparse_feature_names=["item_id", "category_id"],
            sparse_vocab_sizes=[num_items, num_categories],
            sparse_embed_dim=embed_dim,
            use_torchrec=True,
            target_item_mlp_dims=[256],
        )

    def __repr__(self) -> str:
        moe_str = ""
        if self.use_moe:
            moe_str = (
                f"  use_moe={self.use_moe}, num_experts={self.num_experts}, "
                f"num_active_experts={self.num_active_experts},\n"
            )
        return (
            f"MixFormerConfig(\n"
            f"  num_heads={self.num_heads}, num_layers={self.num_layers}, "
            f"hidden_dim={self.hidden_dim},\n"
            f"  seq_length={self.seq_length}, model_dim={self.model_dim},\n"
            f"  num_items={self.num_items}, num_categories={self.num_categories},\n"
            f"  sparse_embed_dim={self.sparse_embed_dim}, "
            f"use_torchrec={self.use_torchrec},\n"
            f"  ffn_hidden_dim={self.ffn_hidden_dim}, dropout={self.dropout},\n"
            f"{moe_str}"
            f"  user_heads={self.user_heads}, item_heads={self.item_heads}\n"
            f")"
        )
