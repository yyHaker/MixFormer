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
from .modules import HeadMixing, PerHeadSwiGLUFFN, PerHeadSparseMoE, SwiGLUFFN


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
    """Cross Attention (CA) 模块 — 使用 Flash Attention 加速。

    用非序列特征生成的 Query 来聚合用户历史序列。
    使用 PyTorch 2.0+ 的 scaled_dot_product_attention，
    自动利用 FlashAttention / Memory-Efficient Attention 内核加速。

    流程：
    1. 序列预处理: h_t = SwiGLUFFN(Norm(s_t)) + s_t
    2. 拆分为 N 个头: h_t^i = h_t[iD:(i+1)D]
    3. K/V 投影 (并行): k^i, v^i = W_k^i · h^i, W_v^i · h^i
    4. Flash Attention:
       z_i = SDPA(q_i, k^i, v^i, attn_mask) + q_i

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

        # K/V 投影: 每个头独立的投影矩阵，打包在一起并行计算
        # 使用单个线性层同时计算所有头的 K 和 V，提升并行度
        self.w_kv = nn.Parameter(
            torch.empty(config.num_heads, config.hidden_dim, 2 * config.hidden_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化 K/V 投影矩阵。"""
        for i in range(self.w_kv.size(0)):
            nn.init.xavier_uniform_(self.w_kv[i])

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

        # 2. 拆分为 N 个头: (batch, T, N*D) -> (batch, T, N, D) -> (batch, N, T, D)
        seq_h = seq_h.view(batch_size, T, N, D).transpose(1, 2)  # (batch, N, T, D)

        # 3. K/V 并行投影: 一次 einsum 同时计算所有头的 K 和 V
        # seq_h: (batch, N, T, D), w_kv: (N, D, 2*D)
        kv = torch.einsum("bntd,nde->bnte", seq_h, self.w_kv)  # (batch, N, T, 2*D)
        keys, values = kv.split(D, dim=-1)  # 各 (batch, N, T, D)

        # 4. Flash Attention (PyTorch 2.0+ scaled_dot_product_attention)
        # q: (batch, N, D) -> (batch, N, 1, D) 作为 query
        q_expanded = q.unsqueeze(2)  # (batch, N, 1, D)

        # 构建 attention mask for SDPA
        # seq_mask: (batch, T) -> (batch, 1, 1, T) 布尔掩码
        attn_mask = None
        if seq_mask is not None:
            attn_mask = seq_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, T)
            # 扩展到 (batch, N, 1, T) 以匹配注意力维度
            attn_mask = attn_mask.expand(-1, N, 1, T)

        # SDPA: 自动选择 FlashAttention / Memory-Efficient / Math 后端
        # q_expanded: (batch, N, 1, D), keys: (batch, N, T, D), values: (batch, N, T, D)
        context = F.scaled_dot_product_attention(
            q_expanded, keys, values,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=self.scale,
        )  # (batch, N, 1, D)
        context = context.squeeze(2)  # (batch, N, D)

        # 残差连接
        z = context + q  # (batch, N, D)

        return z


class OutputFusion(nn.Module):
    """Output Fusion (OF) 模块。

    对 Cross Attention 的输出进行深度融合：
    o_i = FFN_i(Norm(z_i)) + z_i

    支持两种模式：
    - 标准模式 (use_moe=False): 使用 PerHeadSwiGLUFFN
    - MoE 模式 (use_moe=True): 使用 PerHeadSparseMoE (Sparse Mixture of Experts)

    Args:
        config: MixFormer 模型配置
    """

    def __init__(self, config: MixFormerConfig):
        super().__init__()
        self.use_moe = config.use_moe

        self.norm = nn.RMSNorm(config.hidden_dim)

        if config.use_moe:
            self.moe_ffn = PerHeadSparseMoE(
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                ffn_hidden_dim=config.ffn_hidden_dim,
                num_experts=config.num_experts,
                num_active_experts=config.num_active_experts,
                dropout=config.dropout,
            )
        else:
            self.per_head_ffn = PerHeadSwiGLUFFN(
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout,
            )

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        """返回 MoE 辅助损失 (仅 MoE 模式下可用)。"""
        if self.use_moe:
            return self.moe_ffn.aux_loss
        return None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, N, D) — Cross Attention 输出

        Returns:
            o: (batch, N, D) — 融合后的输出
        """
        normed = self.norm(z)  # (batch, N, D)
        if self.use_moe:
            o = self.moe_ffn(normed) + z  # (batch, N, D)
        else:
            o = self.per_head_ffn(normed) + z  # (batch, N, D)
        return o


class MixFormerBlock(nn.Module):
    """单个 MixFormer Block。

    按顺序执行: QueryMixer -> CrossAttention (Flash Attention) -> OutputFusion (可选 MoE)

    Args:
        config: MixFormer 模型配置
    """

    def __init__(self, config: MixFormerConfig):
        super().__init__()
        self.query_mixer = QueryMixer(config)
        self.cross_attention = CrossAttention(config)
        self.output_fusion = OutputFusion(config)

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        """返回本层 MoE 辅助损失。"""
        return self.output_fusion.aux_loss

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

        # 2. Cross Attention (Flash Attention)
        z = self.cross_attention(q, seq, seq_mask)  # (batch, N, D)

        # 3. Output Fusion (可选 MoE)
        o = self.output_fusion(z)  # (batch, N, D)

        return o
