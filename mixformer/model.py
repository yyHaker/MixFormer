"""
MixFormer 完整模型实现。

实现:
1. FeatureEncoder（基于 TorchRec/EmbeddingBag 的特征编码器）
2. TaskHead（MLP+Sigmoid）
3. MixFormer 完整模型 — 适配 Alibaba UserBehavior 数据集
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import MixFormerConfig
from .layers import MixFormerBlock


# ============================================================================
# TorchRec 嵌入管理 (尝试导入，提供 fallback)
# ============================================================================

_TORCHREC_AVAILABLE = False
try:
    import torchrec
    from torchrec import EmbeddingBagCollection, EmbeddingBagConfig
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    _TORCHREC_AVAILABLE = True
except ImportError:
    pass


class EmbeddingBagFallback(nn.Module):
    """nn.Embedding 的 fallback 实现 (兼容 MPS 设备)。

    当 TorchRec (fbgemm-gpu) 不可用时，使用标准 PyTorch 的 nn.Embedding + mean pooling
    提供相同的接口。注意：nn.EmbeddingBag 在 MPS 设备上不可用，因此使用 nn.Embedding。

    Args:
        feature_names: 特征名列表
        vocab_sizes: 每个特征的词表大小
        embed_dim: 嵌入维度
    """

    def __init__(
        self,
        feature_names: list[str],
        vocab_sizes: list[int],
        embed_dim: int,
    ):
        super().__init__()
        self.feature_names = feature_names
        self.embed_dim = embed_dim

        self.embeddings = nn.ModuleDict()
        for name, vocab_size in zip(feature_names, vocab_sizes):
            # +1 for padding index 0
            self.embeddings[name] = nn.Embedding(
                num_embeddings=vocab_size + 1,
                embedding_dim=embed_dim,
                padding_idx=0,
            )

        self._init_weights()

    def _init_weights(self):
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, std=0.02)
            # padding_idx 的权重保持为 0
            with torch.no_grad():
                emb.weight[0].zero_()

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: {feature_name: (batch,)} 或 {feature_name: (batch, L)} 的特征 ID

        Returns:
            {feature_name: (batch, embed_dim)} 嵌入向量字典
        """
        result = {}
        for name in self.feature_names:
            ids = features[name]
            if ids.dim() == 1:
                # 单个 ID: (batch,) -> (batch, embed_dim)
                result[name] = self.embeddings[name](ids)
            else:
                # 多个 ID: (batch, L) -> mean pooling -> (batch, embed_dim)
                emb = self.embeddings[name](ids)  # (batch, L, embed_dim)
                # 对非 padding 位置做 mean pooling
                mask = (ids != 0).unsqueeze(-1).float()  # (batch, L, 1)
                count = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
                result[name] = (emb * mask).sum(dim=1) / count  # (batch, embed_dim)
        return result


def create_embedding_collection(
    feature_names: list[str],
    vocab_sizes: list[int],
    embed_dim: int,
    use_torchrec: bool = True,
) -> nn.Module:
    """创建嵌入集合的工厂函数。

    优先使用 TorchRec EmbeddingBagCollection，如果不可用则 fallback 到 nn.EmbeddingBag。

    Args:
        feature_names: 特征名列表
        vocab_sizes: 词表大小列表
        embed_dim: 嵌入维度
        use_torchrec: 是否尝试使用 TorchRec

    Returns:
        嵌入集合模块
    """
    if use_torchrec and _TORCHREC_AVAILABLE:
        configs = [
            EmbeddingBagConfig(
                name=name,
                embedding_dim=embed_dim,
                num_embeddings=vocab_size + 1,  # +1 for padding
                feature_names=[name],
            )
            for name, vocab_size in zip(feature_names, vocab_sizes)
        ]
        return EmbeddingBagCollection(tables=configs)
    else:
        if use_torchrec and not _TORCHREC_AVAILABLE:
            import warnings
            warnings.warn(
                "TorchRec not available, falling back to nn.EmbeddingBag. "
                "Install torchrec: pip install torchrec",
                RuntimeWarning,
            )
        return EmbeddingBagFallback(feature_names, vocab_sizes, embed_dim)


# ============================================================================
# 1. FeatureEncoder (特征编码器)
# ============================================================================


class FeatureEncoder(nn.Module):
    """特征编码器 — 适配 Alibaba UserBehavior 数据集。

    数据集特征:
    - 目标物品: target_item_id + target_cate_id → 嵌入拼接 → MLP 投影 → (batch, N, D) Query
    - 历史序列: hist_item_ids + hist_cate_ids → 嵌入拼接 → 投影 → (batch, T, N*D) 序列

    使用 TorchRec EmbeddingBagCollection 管理 item_id 和 category_id 的嵌入表，
    如果 TorchRec 不可用则自动 fallback 到 nn.EmbeddingBag。

    Args:
        config: MixFormer 模型配置
    """

    def __init__(self, config: MixFormerConfig):
        super().__init__()
        self.config = config
        N = config.num_heads
        D = config.hidden_dim
        embed_dim = config.sparse_embed_dim

        # ---- 稀疏特征嵌入 (TorchRec / EmbeddingBag fallback) ----
        # 用于目标物品: item_id embedding + cate_id embedding
        self.target_embedding = create_embedding_collection(
            feature_names=config.sparse_feature_names,
            vocab_sizes=config.sparse_vocab_sizes,
            embed_dim=embed_dim,
            use_torchrec=config.use_torchrec,
        )

        # 用于序列物品: item_id embedding + cate_id embedding
        # 序列中的每个物品需要保持独立的嵌入 (不做 bag pooling)
        # 使用标准 nn.Embedding (不是 EmbeddingBag)
        self.seq_item_embedding = nn.Embedding(
            num_embeddings=config.num_items + 1,  # +1 for padding
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.seq_cate_embedding = nn.Embedding(
            num_embeddings=config.num_categories + 1,  # +1 for padding
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # ---- 目标物品投影: 2*embed_dim → N*D (拆分为 N 个头) ----
        target_concat_dim = 2 * embed_dim  # item + cate 嵌入拼接
        mlp_dims = config.target_item_mlp_dims or [256]

        target_mlp_layers = []
        prev_dim = target_concat_dim
        for h_dim in mlp_dims:
            target_mlp_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            ])
            prev_dim = h_dim
        target_mlp_layers.append(nn.Linear(prev_dim, N * D))
        self.target_mlp = nn.Sequential(*target_mlp_layers)

        # ---- 序列物品投影: 2*embed_dim → N*D ----
        self.seq_projection = nn.Linear(2 * embed_dim, N * D, bias=False)

        # ---- 序列位置编码 ----
        self.seq_position_embedding = nn.Embedding(
            num_embeddings=config.seq_length,
            embedding_dim=N * D,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.seq_item_embedding.weight, std=0.02)
        nn.init.normal_(self.seq_cate_embedding.weight, std=0.02)
        nn.init.normal_(self.seq_position_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.seq_projection.weight)
        for module in self.target_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_target(
        self,
        target_item_id: torch.Tensor,
        target_cate_id: torch.Tensor,
    ) -> torch.Tensor:
        """编码目标物品特征为 Query 头。

        Args:
            target_item_id: (batch,) 目标物品 ID
            target_cate_id: (batch,) 目标物品类目 ID

        Returns:
            query: (batch, N, D) — N 个头的 Query 向量
        """
        N = self.config.num_heads
        D = self.config.hidden_dim

        # 获取嵌入
        if isinstance(self.target_embedding, EmbeddingBagFallback):
            emb_dict = self.target_embedding({
                "item_id": target_item_id,
                "category_id": target_cate_id,
            })
            item_emb = emb_dict["item_id"]  # (batch, embed_dim)
            cate_emb = emb_dict["category_id"]  # (batch, embed_dim)
        else:
            # TorchRec EmbeddingBagCollection 路径
            # 构建 KeyedJaggedTensor
            batch_size = target_item_id.size(0)
            keys = self.config.sparse_feature_names
            values = torch.cat([target_item_id, target_cate_id])
            lengths = torch.ones(2 * batch_size, dtype=torch.int32,
                                 device=target_item_id.device)
            kjt = KeyedJaggedTensor(
                keys=keys,
                values=values,
                lengths=lengths,
            )
            ebc_out = self.target_embedding(kjt)
            item_emb = ebc_out["item_id"]
            cate_emb = ebc_out["category_id"]

        # 拼接 + MLP 投影 → (batch, N*D) → (batch, N, D)
        concat = torch.cat([item_emb, cate_emb], dim=-1)  # (batch, 2*embed_dim)
        projected = self.target_mlp(concat)  # (batch, N*D)
        query = projected.view(-1, N, D)  # (batch, N, D)

        return query

    def encode_sequence(
        self,
        hist_item_ids: torch.Tensor,
        hist_cate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """编码历史行为序列特征。

        Args:
            hist_item_ids: (batch, T) 历史物品 ID 序列
            hist_cate_ids: (batch, T) 历史类目 ID 序列

        Returns:
            seq: (batch, T, N*D) — 序列特征嵌入
        """
        batch_size, T = hist_item_ids.shape

        # 获取嵌入
        item_emb = self.seq_item_embedding(hist_item_ids)  # (batch, T, embed_dim)
        cate_emb = self.seq_cate_embedding(hist_cate_ids)  # (batch, T, embed_dim)

        # 拼接 + 投影
        concat = torch.cat([item_emb, cate_emb], dim=-1)  # (batch, T, 2*embed_dim)
        seq = self.seq_projection(concat)  # (batch, T, N*D)

        # 位置编码
        positions = torch.arange(T, device=hist_item_ids.device).unsqueeze(0)
        pos_emb = self.seq_position_embedding(positions)  # (1, T, N*D)
        seq = seq + pos_emb

        return seq

    def forward(
        self,
        target_item_id: torch.Tensor,
        target_cate_id: torch.Tensor,
        hist_item_ids: torch.Tensor,
        hist_cate_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target_item_id: (batch,) 目标物品 ID
            target_cate_id: (batch,) 目标物品类目 ID
            hist_item_ids: (batch, T) 历史物品 ID 序列
            hist_cate_ids: (batch, T) 历史类目 ID 序列

        Returns:
            query: (batch, N, D) — 目标物品 Query 头
            seq: (batch, T, N*D) — 历史行为序列嵌入
        """
        query = self.encode_target(target_item_id, target_cate_id)
        seq = self.encode_sequence(hist_item_ids, hist_cate_ids)
        return query, seq


# ============================================================================
# 2. TaskHead
# ============================================================================


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


# ============================================================================
# 3. MixFormer 主模型
# ============================================================================


class MixFormer(nn.Module):
    """MixFormer 推荐模型。

    适配 Alibaba UserBehavior 数据集，使用 TorchRec EmbeddingBagCollection 管理
    稀疏特征嵌入 (item_id + category_id)。
    数据集中的用户历史行为序列天然适配 MixFormer 的 CrossAttention 序列建模。

    架构:
        目标物品 (item_id + cate_id) → FeatureEncoder.encode_target → Query (batch, N, D)
        历史序列 (item_ids + cate_ids) → FeatureEncoder.encode_sequence → (batch, T, N*D)
        → L × MixFormerBlock(Query, Sequence) → TaskHead → CTR 预测

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
        target_item_id: torch.Tensor,
        target_cate_id: torch.Tensor,
        hist_item_ids: torch.Tensor,
        hist_cate_ids: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            target_item_id: (batch,) 目标物品 ID
            target_cate_id: (batch,) 目标物品类目 ID
            hist_item_ids: (batch, T) 历史物品 ID 序列
            hist_cate_ids: (batch, T) 历史类目 ID 序列
            seq_mask: (batch, T) 序列 padding mask, True 表示有效位置

        Returns:
            pred: (batch, 1) — CTR 预测概率
        """
        # 1. 特征编码
        query, seq = self.feature_encoder(
            target_item_id, target_cate_id, hist_item_ids, hist_cate_ids
        )
        # query: (batch, N, D), seq: (batch, T, N*D)

        # 2. L 层 MixFormer Block
        for block in self.blocks:
            query = block(query, seq, seq_mask=seq_mask)
        # query: (batch, N, D)

        # 3. 最终归一化并展平
        x = self.final_norm(query)  # (batch, N, D)
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
