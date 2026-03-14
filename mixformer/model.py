"""
MixFormer 完整模型实现。

实现 FeatureEncoder（特征嵌入/拆分/投影）、TaskHead（MLP+Sigmoid）、
MixFormer 完整模型以及 UIMixFormer 用户-物品解耦变体。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import MixFormerConfig
from .layers import MixFormerBlock


class FeatureEncoder(nn.Module):
    """特征编码器。

    负责将原始特征编码为模型输入：
    1. 非序列特征: 嵌入 -> 拼接 -> 拆分为 N 个子向量 -> 投影到 D 维
    2. 序列特征: 物品 ID 嵌入 -> 投影到 N*D 维

    Args:
        config: MixFormer 模型配置
    """

    def __init__(self, config: MixFormerConfig):
        super().__init__()
        self.config = config
        N = config.num_heads
        D = config.hidden_dim

        # 非序列特征嵌入层: 每个特征一个 Embedding
        self.feature_embeddings = nn.ModuleDict()
        for feat_name, vocab_size in config.vocab_sizes.items():
            self.feature_embeddings[feat_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=config.feature_embed_dim,
            )

        # 非序列特征: 拆分后的投影层 (N 个线性层, 每个将 d=D_ns/N 投影到 D)
        head_input_dim = config.head_input_dim  # d = D_ns / N
        self.head_projections = nn.ModuleList(
            [nn.Linear(head_input_dim, D, bias=False) for _ in range(N)]
        )

        # 序列特征嵌入: 物品 ID -> 嵌入
        self.seq_item_embedding = nn.Embedding(
            num_embeddings=config.num_items,
            embedding_dim=config.seq_feature_embed_dim,  # N * D
        )

        # 序列位置编码 (可学习)
        self.seq_position_embedding = nn.Embedding(
            num_embeddings=config.seq_length,
            embedding_dim=config.seq_feature_embed_dim,  # N * D
        )

        self._init_weights()

    def _init_weights(self):
        """初始化嵌入层权重。"""
        for emb in self.feature_embeddings.values():
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.seq_item_embedding.weight, std=0.02)
        nn.init.normal_(self.seq_position_embedding.weight, std=0.02)
        for proj in self.head_projections:
            nn.init.xavier_uniform_(proj.weight)

    def encode_non_seq_features(
        self, non_seq_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """编码非序列特征。

        Args:
            non_seq_features: {feature_name: (batch,)} 特征字典

        Returns:
            x: (batch, N, D) — N 个头的特征向量
        """
        config = self.config
        N = config.num_heads
        d = config.head_input_dim  # D_ns / N

        # 1. 嵌入所有特征
        embeddings = []
        for feat_name in sorted(config.vocab_sizes.keys()):
            if feat_name in non_seq_features:
                emb = self.feature_embeddings[feat_name](non_seq_features[feat_name])
                embeddings.append(emb)  # (batch, feature_embed_dim)

        # 2. 拼接: (batch, D_ns) where D_ns = M * feature_embed_dim
        e_ns = torch.cat(embeddings, dim=-1)  # (batch, D_ns)

        # 3. 拆分为 N 个子向量并投影
        heads = []
        for j in range(N):
            # 取第 j 个子向量: (batch, d)
            sub = e_ns[:, j * d : (j + 1) * d]
            # 投影到 D 维: (batch, D)
            heads.append(self.head_projections[j](sub))

        # 4. 堆叠: (batch, N, D)
        x = torch.stack(heads, dim=1)
        return x

    def encode_seq_features(self, seq_features: torch.Tensor) -> torch.Tensor:
        """编码序列特征。

        Args:
            seq_features: (batch, T) — 用户行为序列中的物品 ID

        Returns:
            seq: (batch, T, N*D) — 序列特征嵌入
        """
        batch_size, T = seq_features.shape

        # 物品嵌入: (batch, T, N*D)
        seq_emb = self.seq_item_embedding(seq_features)

        # 位置编码
        positions = torch.arange(T, device=seq_features.device).unsqueeze(0)  # (1, T)
        pos_emb = self.seq_position_embedding(positions)  # (1, T, N*D)

        seq = seq_emb + pos_emb  # (batch, T, N*D)
        return seq

    def forward(
        self,
        non_seq_features: Dict[str, torch.Tensor],
        seq_features: torch.Tensor,
    ) -> tuple:
        """
        Args:
            non_seq_features: {feature_name: (batch,)} 非序列特征
            seq_features: (batch, T) 序列物品 ID

        Returns:
            x: (batch, N, D) — 非序列特征头
            seq: (batch, T, N*D) — 序列特征嵌入
        """
        x = self.encode_non_seq_features(non_seq_features)
        seq = self.encode_seq_features(seq_features)
        return x, seq


class TaskHead(nn.Module):
    """任务头: MLP + Sigmoid 输出 CTR 预测概率。

    Args:
        input_dim: 输入维度 (N * D)
        hidden_dims: MLP 隐藏层维度列表
        dropout: Dropout 概率
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                ]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N*D) — 特征向量

        Returns:
            pred: (batch, 1) — CTR 预测概率 [0, 1]
        """
        logits = self.mlp(x)  # (batch, 1)
        return torch.sigmoid(logits)


class MixFormer(nn.Module):
    """MixFormer 完整推荐模型。

    在单个骨干网络中联合建模用户行为序列和稠密特征交互。

    架构: FeatureEncoder -> L × MixFormerBlock -> TaskHead

    Args:
        config: MixFormer 模型配置
    """

    def __init__(self, config: MixFormerConfig):
        super().__init__()
        self.config = config

        # 特征编码器
        self.feature_encoder = FeatureEncoder(config)

        # L 层 MixFormer Block
        self.blocks = nn.ModuleList(
            [MixFormerBlock(config) for _ in range(config.num_layers)]
        )

        # 最终层归一化
        self.final_norm = nn.RMSNorm(config.hidden_dim)

        # 任务头
        self.task_head = TaskHead(
            input_dim=config.num_heads * config.hidden_dim,
            hidden_dims=config.task_head_hidden_dims,
            dropout=config.dropout,
        )

    def forward(
        self,
        non_seq_features: Dict[str, torch.Tensor],
        seq_features: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            non_seq_features: {feature_name: (batch,)} — 非序列特征字典
            seq_features: (batch, T) — 用户行为序列物品 ID
            seq_mask: (batch, T) — 序列 padding mask, True 表示有效位置

        Returns:
            pred: (batch, 1) — CTR 预测概率
        """
        # 1. 特征编码
        x, seq = self.feature_encoder(non_seq_features, seq_features)
        # x: (batch, N, D), seq: (batch, T, N*D)

        # 2. L 层 MixFormer Block
        for block in self.blocks:
            x = block(x, seq, seq_mask=seq_mask)
        # x: (batch, N, D)

        # 3. 最终归一化并展平
        x = self.final_norm(x)  # (batch, N, D)
        x = x.reshape(x.size(0), -1)  # (batch, N*D)

        # 4. 任务头
        pred = self.task_head(x)  # (batch, 1)

        return pred

    def get_num_params(self) -> int:
        """返回模型总参数量。"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """返回可训练参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UIMixFormer(MixFormer):
    """User-Item 解耦变体 (UI-MixFormer)。

    通过掩码矩阵实现用户侧计算可复用，支持 Request Level Batching，
    显著降低推理延迟。

    解耦策略:
    1. 将非序列特征分为用户侧 (N_U 个头) 和物品侧 (N_G 个头)
    2. 在 Query Mixer 的 HeadMixing 中引入掩码，防止物品信息污染用户头
    3. 推理时用户侧计算可在多个候选物品间复用

    Args:
        config: MixFormer 模型配置
    """

    def __init__(self, config: MixFormerConfig):
        super().__init__(config)
        self.user_heads = config.user_heads
        self.item_heads = config.item_heads

        # 构建解耦掩码矩阵: (N, D)
        self.register_buffer("decouple_mask", self._build_decouple_mask(config))

    @staticmethod
    def _build_decouple_mask(config: MixFormerConfig) -> torch.Tensor:
        """构建解耦掩码矩阵。

        M[i, j] = 0 if i < N_U and j >= N_U * (D/N)
                 = 1 otherwise

        用于阻断物品侧信息到用户侧头的信息流。
        """
        N = config.num_heads
        D = config.hidden_dim
        N_U = config.user_heads

        mask = torch.ones(N, D)

        # 用户侧头 (i < N_U) 不接收物品侧信息
        # HeadMixing 后每个位置包含所有头的混合信息
        # 掩码确保用户侧头的输出中物品相关的部分被置零
        boundary = N_U * (D // N)
        for i in range(N_U):
            mask[i, boundary:] = 0.0

        return mask

    def forward(
        self,
        non_seq_features: Dict[str, torch.Tensor],
        seq_features: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """标准前向传播（带解耦掩码）。"""
        # 1. 特征编码
        x, seq = self.feature_encoder(non_seq_features, seq_features)

        # 2. L 层 MixFormer Block（带解耦掩码）
        for block in self.blocks:
            x = block(x, seq, seq_mask=seq_mask, decouple_mask=self.decouple_mask)

        # 3. 最终归一化并展平
        x = self.final_norm(x)
        x = x.reshape(x.size(0), -1)

        # 4. 任务头
        pred = self.task_head(x)
        return pred

    def encode_user(
        self,
        non_seq_features: Dict[str, torch.Tensor],
        seq_features: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """编码用户侧表示（可缓存复用）。

        Args:
            non_seq_features: 用户侧非序列特征
            seq_features: 用户行为序列
            seq_mask: 序列 padding mask

        Returns:
            user_repr: (batch, N_U, D) — 用户侧头表示
        """
        # 特征编码
        x, seq = self.feature_encoder(non_seq_features, seq_features)

        # L 层 MixFormer Block
        for block in self.blocks:
            x = block(x, seq, seq_mask=seq_mask, decouple_mask=self.decouple_mask)

        x = self.final_norm(x)

        # 提取用户侧头
        user_repr = x[:, : self.user_heads, :]  # (batch, N_U, D)
        return user_repr

    def predict_with_user_cache(
        self,
        user_repr: torch.Tensor,
        item_non_seq_features: Dict[str, torch.Tensor],
        seq_features: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """使用缓存的用户表示和物品特征进行预测。

        Args:
            user_repr: (batch, N_U, D) — 缓存的用户侧表示
            item_non_seq_features: 物品侧非序列特征
            seq_features: 用于物品侧头计算的序列
            seq_mask: 序列 mask

        Returns:
            pred: (batch, 1) — CTR 预测概率
        """
        # 编码物品侧特征
        x, seq = self.feature_encoder(item_non_seq_features, seq_features)

        # 使用完整的 x 进行 block 计算
        for block in self.blocks:
            x = block(x, seq, seq_mask=seq_mask, decouple_mask=self.decouple_mask)

        x = self.final_norm(x)

        # 替换用户侧头为缓存的用户表示
        x = torch.cat(
            [user_repr, x[:, self.user_heads :, :]], dim=1
        )  # (batch, N, D)

        # 展平并通过任务头
        x = x.reshape(x.size(0), -1)
        pred = self.task_head(x)
        return pred
