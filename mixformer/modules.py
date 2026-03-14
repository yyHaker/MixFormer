"""
MixFormer 基础组件模块。

实现 SwiGLUFFN、HeadMixing、PerHeadSwiGLUFFN 三个核心基础组件。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU 激活的前馈网络。

    SwiGLU(x) = (W1·x) ⊙ SiLU(W_gate·x)
    FFN(x) = W2 · SwiGLU(x)

    Args:
        in_dim: 输入维度
        hidden_dim: 中间隐藏维度
        out_dim: 输出维度（默认等于 in_dim）
        dropout: Dropout 概率
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_dim = out_dim or in_dim

        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化。"""
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim)

        Returns:
            (..., out_dim)
        """
        # SwiGLU: (W1·x) ⊙ SiLU(W_gate·x)
        gate = F.silu(self.w_gate(x))
        x = self.w1(x) * gate
        x = self.dropout(x)
        x = self.w2(x)
        return x


class HeadMixing(nn.Module):
    """无参数的跨头信息交换模块。

    通过转置和重塑操作实现不同头之间的信息流动。
    输入 (batch, N, D) -> 转置为 (batch, D, N) -> 重塑回 (batch, N, D)，
    使得每个"头"位置包含来自所有原始头的混合信息。

    Args:
        num_heads: 头数 N
        hidden_dim: 每个头的维度 D
    """

    def __init__(self, num_heads: int, hidden_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N, D) — N 个头，每个头 D 维

        Returns:
            (batch, N, D) — 跨头混合后的结果
        """
        batch_size = x.size(0)
        # (batch, N, D) -> (batch, D, N) -> (batch, N, D)
        # 转置后 reshape 回原始形状，实现信息在头维度和特征维度之间的交换
        x = x.transpose(1, 2).contiguous()  # (batch, D, N)
        x = x.view(batch_size, self.num_heads, self.hidden_dim)  # (batch, N, D)
        return x


class PerHeadSwiGLUFFN(nn.Module):
    """N 个头并行的 SwiGLU FFN。

    每个头有独立的 SwiGLU FFN 参数。为了效率，使用分组计算——
    将 N 个头的权重打包在一起，通过 einsum 实现批量矩阵乘法。

    Args:
        num_heads: 头数 N
        hidden_dim: 每个头的输入/输出维度 D
        ffn_hidden_dim: SwiGLU FFN 的中间维度
        dropout: Dropout 概率
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        ffn_hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        # 每个头独立的权重: (N, hidden_dim, ffn_hidden_dim)
        self.w1 = nn.Parameter(torch.empty(num_heads, hidden_dim, ffn_hidden_dim))
        self.w_gate = nn.Parameter(torch.empty(num_heads, hidden_dim, ffn_hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_heads, ffn_hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化。"""
        for w in [self.w1, self.w_gate, self.w2]:
            # 对每个头分别做 Xavier 初始化
            for i in range(self.num_heads):
                nn.init.xavier_uniform_(w[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N, D) — N 个头的输入

        Returns:
            (batch, N, D) — 每个头独立经过 SwiGLU FFN 后的输出
        """
        # x: (batch, N, D)
        # w1: (N, D, H), w_gate: (N, D, H), w2: (N, H, D)
        # 使用 einsum 实现批量矩阵乘法，等价于每个头独立做线性变换

        # (batch, N, D) @ (N, D, H) -> (batch, N, H)
        h1 = torch.einsum("bnd,ndh->bnh", x, self.w1)
        gate = torch.einsum("bnd,ndh->bnh", x, self.w_gate)

        # SwiGLU: h1 ⊙ SiLU(gate)
        h = h1 * F.silu(gate)
        h = self.dropout(h)

        # (batch, N, H) @ (N, H, D) -> (batch, N, D)
        out = torch.einsum("bnh,nhd->bnd", h, self.w2)
        return out
