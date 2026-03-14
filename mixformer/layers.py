"""
MixFormer Block 层实现。

实现 QueryMixer、CrossAttention、OutputFusion 三大核心模块，
以及组装后的 MixFormerBlock。严格按论文公式实现残差连接和注意力计算。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MixFormerConfig
from .modules import HeadMixing, PerHeadSwiGLUFFN, SwiGLUFFN


class QueryMixer(nn.Module):
    """Query Mixer (QM) 模块 — 替代标准 Self-Attention。

    包含两个子操作：
    1. HeadMixing: 无参数的跨头信息交换 + 残差连接
       P = HeadMixing(Norm(X)) + X
    2. Per-Head FFN: 每个头独立应用 SwiGLU FFN + 残差连接
       q_i = SwiGLUFFN_i(Norm(p_i)) + p_i

    Args:
        config: MixFormer 模型配置
    """

    def __init__(self, config: MixFormerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim

        # HeadMixing 前的 RMSNorm
        self.norm1 = nn.RMSNorm(config.hidden_dim)
        self.head_mixing = HeadMixing(config.num_heads, config.hidden_dim)

        # Per-Head FFN 前的 RMSNorm
        self.norm2 = nn.RMSNorm(config.hidden_dim)
        self.per_head_ffn = PerHeadSwiGLUFFN(
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, N, D) — 非序列特征头
            mask: (N, D) — 可选的解耦掩码矩阵（UI-MixFormer 使用）

        Returns:
            q: (batch, N, D) — Query 输出
        """
        # 1. HeadMixing + 残差: P = HeadMixing(Norm(X)) + X
        normed = self.norm1(x)  # 对每个头独立做 RMSNorm: (batch, N, D)
        mixed = self.head_mixing(normed)  # (batch, N, D)

        # 如果有解耦掩码，应用掩码
        if mask is not None:
            mixed = mixed * mask.unsqueeze(0)  # (batch, N, D) * (1, N, D)

        p = mixed + x  # 残差连接

        # 2. Per-Head FFN + 残差: q_i = SwiGLUFFN_i(Norm(p_i)) + p_i
        normed = self.norm2(p)  # (batch, N, D)
        q = self.per_head_ffn(normed) + p  # (batch, N, D)

        return q


class CrossAttention(nn.Module):
    """Cross Attention (CA) 模块 — 序列建模核心。

    用非序列特征生成的 Query 来聚合用户历史序列。

    流程：
    1. 序列预处理: h_t = SwiGLUFFN(Norm(s_t)) + s_t
    2. 拆分为 N 个头: h_t^i = h_t[iD:(i+1)D]
    3. K/V 投影: k_t^i = W_k^i · h_t^i; v_t^i = W_v^i · h_t^i
    4. Scaled Dot-Product Attention:
       z_i = Σ softmax(q_i^T · k_t^i / √D) · v_t^i + q_i

    Args:
        config: MixFormer 模型配置
    """

    def __init__(self, config: MixFormerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.scale = config.hidden_dim ** -0.5

        # 序列预处理: SwiGLU FFN (对整个 N*D 维度操作)
        self.seq_norm = nn.RMSNorm(config.num_heads * config.hidden_dim)
        self.seq_ffn = SwiGLUFFN(
            in_dim=config.num_heads * config.hidden_dim,
            hidden_dim=config.ffn_hidden_dim * config.num_heads // 4,  # 序列 FFN 适当缩小
            out_dim=config.num_heads * config.hidden_dim,
            dropout=config.dropout,
        )

        # K/V 投影: 每个头独立的投影矩阵
        # 使用分组线性层实现: (N, D, D)
        self.w_k = nn.Parameter(
            torch.empty(config.num_heads, config.hidden_dim, config.hidden_dim)
        )
        self.w_v = nn.Parameter(
            torch.empty(config.num_heads, config.hidden_dim, config.hidden_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化 K/V 投影矩阵。"""
        for i in range(self.w_k.size(0)):
            nn.init.xavier_uniform_(self.w_k[i])
            nn.init.xavier_uniform_(self.w_v[i])

    def forward(
        self,
        q: torch.Tensor,
        seq: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: (batch, N, D) — Query Mixer 输出的 query
            seq: (batch, T, N*D) — 用户行为序列特征
            seq_mask: (batch, T) — 序列 padding mask, True 表示有效位置

        Returns:
            z: (batch, N, D) — Cross Attention 输出
        """
        batch_size, T, _ = seq.shape
        N, D = self.num_heads, self.hidden_dim

        # 1. 序列预处理: h_t = SwiGLUFFN(Norm(s_t)) + s_t
        seq_normed = self.seq_norm(seq)  # (batch, T, N*D)
        seq_h = self.seq_ffn(seq_normed) + seq  # (batch, T, N*D)

        # 2. 拆分为 N 个头: (batch, T, N*D) -> (batch, T, N, D)
        seq_h = seq_h.view(batch_size, T, N, D)

        # 3. K/V 投影
        # seq_h: (batch, T, N, D), w_k: (N, D, D)
        # -> keys: (batch, T, N, D)
        keys = torch.einsum("btnd,nde->btne", seq_h, self.w_k)  # (batch, T, N, D)
        values = torch.einsum("btnd,nde->btne", seq_h, self.w_v)  # (batch, T, N, D)

        # 4. Scaled Dot-Product Attention
        # q: (batch, N, D) -> (batch, N, 1, D) for broadcasting
        # keys: (batch, T, N, D) -> (batch, N, T, D)
        keys = keys.permute(0, 2, 1, 3)  # (batch, N, T, D)
        values = values.permute(0, 2, 1, 3)  # (batch, N, T, D)
        q_expanded = q.unsqueeze(2)  # (batch, N, 1, D)

        # Attention scores: (batch, N, 1, D) @ (batch, N, D, T) -> (batch, N, 1, T)
        attn_scores = torch.matmul(q_expanded, keys.transpose(-2, -1)) * self.scale
        # (batch, N, 1, T)

        # 应用 padding mask
        if seq_mask is not None:
            # seq_mask: (batch, T) -> (batch, 1, 1, T)
            mask = seq_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, T)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, N, 1, T)

        # Weighted sum: (batch, N, 1, T) @ (batch, N, T, D) -> (batch, N, 1, D)
        context = torch.matmul(attn_weights, values)  # (batch, N, 1, D)
        context = context.squeeze(2)  # (batch, N, D)

        # 残差连接
        z = context + q  # (batch, N, D)

        return z


class OutputFusion(nn.Module):
    """Output Fusion (OF) 模块。

    对 Cross Attention 的输出进行深度融合：
    o_i = SwiGLUFFN_i(Norm(z_i)) + z_i

    Args:
        config: MixFormer 模型配置
    """

    def __init__(self, config: MixFormerConfig):
        super().__init__()

        self.norm = nn.RMSNorm(config.hidden_dim)
        self.per_head_ffn = PerHeadSwiGLUFFN(
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, N, D) — Cross Attention 输出

        Returns:
            o: (batch, N, D) — 融合后的输出
        """
        normed = self.norm(z)  # (batch, N, D)
        o = self.per_head_ffn(normed) + z  # (batch, N, D)
        return o


class MixFormerBlock(nn.Module):
    """单个 MixFormer Block。

    按顺序执行: QueryMixer -> CrossAttention -> OutputFusion

    Args:
        config: MixFormer 模型配置
    """

    def __init__(self, config: MixFormerConfig):
        super().__init__()
        self.query_mixer = QueryMixer(config)
        self.cross_attention = CrossAttention(config)
        self.output_fusion = OutputFusion(config)

    def forward(
        self,
        x: torch.Tensor,
        seq: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        decouple_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, N, D) — 非序列特征头（或上一层的输出）
            seq: (batch, T, N*D) — 用户行为序列特征
            seq_mask: (batch, T) — 序列 padding mask
            decouple_mask: (N, D) — UI-MixFormer 解耦掩码

        Returns:
            o: (batch, N, D) — 本层输出
        """
        # 1. Query Mixer
        q = self.query_mixer(x, mask=decouple_mask)  # (batch, N, D)

        # 2. Cross Attention
        z = self.cross_attention(q, seq, seq_mask)  # (batch, N, D)

        # 3. Output Fusion
        o = self.output_fusion(z)  # (batch, N, D)

        return o
