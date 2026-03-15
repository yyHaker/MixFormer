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


class SparseMoE(nn.Module):
    """Sparse Mixture of Experts (MoE) 模块。

    使用 Top-K 路由策略：对每个 token，选择 K 个专家进行计算，
    通过 softmax 路由权重加权求和。包含负载均衡辅助损失以防止专家坍缩。

    每个专家是一个独立的 SwiGLU FFN。

    Args:
        in_dim: 输入/输出维度 D
        hidden_dim: FFN 中间隐藏维度
        num_experts: 专家数量 E
        num_active_experts: Top-K 激活的专家数量
        dropout: Dropout 概率
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        num_active_experts: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts

        # 路由器: 线性层将输入映射到专家分数
        self.router = nn.Linear(in_dim, num_experts, bias=False)

        # E 个 SwiGLU FFN 专家 (参数打包以高效计算)
        # 每个专家: in_dim -> hidden_dim -> in_dim
        self.w1 = nn.Parameter(torch.empty(num_experts, in_dim, hidden_dim))
        self.w_gate = nn.Parameter(torch.empty(num_experts, in_dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, in_dim))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 辅助损失缓存 (训练时使用)
        self._aux_loss: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.router.weight)
        for i in range(self.num_experts):
            nn.init.xavier_uniform_(self.w1[i])
            nn.init.xavier_uniform_(self.w_gate[i])
            nn.init.xavier_uniform_(self.w2[i])

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        """返回负载均衡辅助损失 (训练时非 None)。"""
        return self._aux_loss

    def _compute_aux_loss(
        self, router_logits: torch.Tensor, expert_mask: torch.Tensor
    ) -> torch.Tensor:
        """计算负载均衡辅助损失 (Switch Transformer 风格)。

        L_aux = E * Σ_i (f_i · p_i)
        其中 f_i = 分配到专家 i 的 token 比例
             p_i = 路由到专家 i 的概率均值

        Args:
            router_logits: (num_tokens, E) 路由器原始分数
            expert_mask: (num_tokens, E) 0/1 掩码，表示 token 被分配到哪些专家
        """
        # f_i: 每个专家被分配到的 token 比例
        tokens_per_expert = expert_mask.float().mean(dim=0)  # (E,)

        # p_i: 每个专家的平均路由概率
        router_probs = F.softmax(router_logits, dim=-1)  # (num_tokens, E)
        avg_probs = router_probs.mean(dim=0)  # (E,)

        # Switch Transformer 风格辅助损失
        aux_loss = self.num_experts * (tokens_per_expert * avg_probs).sum()
        return aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., D) — 任意前导维度的输入

        Returns:
            (..., D) — MoE 输出，与输入形状相同
        """
        orig_shape = x.shape
        # 展平为 (num_tokens, D)
        x_flat = x.reshape(-1, self.in_dim)  # (T, D)
        num_tokens = x_flat.size(0)

        # 1. 路由: 计算每个 token 对每个专家的分数
        router_logits = self.router(x_flat)  # (T, E)

        # 2. Top-K 选择
        topk_logits, topk_indices = torch.topk(
            router_logits, self.num_active_experts, dim=-1
        )  # (T, K), (T, K)
        topk_weights = F.softmax(topk_logits, dim=-1)  # (T, K) 归一化权重

        # 3. 计算辅助损失 (仅训练时)
        if self.training:
            expert_mask = torch.zeros_like(router_logits)
            expert_mask.scatter_(1, topk_indices, 1.0)
            self._aux_loss = self._compute_aux_loss(router_logits, expert_mask)
        else:
            self._aux_loss = None

        # 4. 对每个专家计算输出 (批量方式，利用 einsum 并行)
        # 策略: 为所有专家计算，然后用稀疏掩码选择
        # 为了效率，对每个专家分别选择其 token 子集，计算后合并

        output = torch.zeros_like(x_flat)  # (T, D)

        for k in range(self.num_active_experts):
            # 第 k 个被选中的专家 (每个 token 不同)
            expert_idx = topk_indices[:, k]  # (T,)
            weight = topk_weights[:, k]  # (T,)

            # 按专家分组处理
            for e in range(self.num_experts):
                mask = (expert_idx == e)  # (T,) bool
                if not mask.any():
                    continue

                x_e = x_flat[mask]  # (T_e, D)

                # SwiGLU: (x @ w1) * silu(x @ w_gate) -> @ w2
                h1 = x_e @ self.w1[e]  # (T_e, H)
                gate = F.silu(x_e @ self.w_gate[e])  # (T_e, H)
                h = self.dropout(h1 * gate)
                out_e = h @ self.w2[e]  # (T_e, D)

                # 加权累加
                output[mask] += weight[mask].unsqueeze(-1) * out_e

        return output.reshape(orig_shape)


class PerHeadSparseMoE(nn.Module):
    """N 个头共享路由器的 Sparse MoE。

    每个头独立应用 MoE：所有头共享同一路由器，但每个专家有独立的
    per-head 权重。这保持了 per-head 独立性的同时引入了 MoE 的稀疏激活。

    Args:
        num_heads: 头数 N
        hidden_dim: 每个头的输入/输出维度 D
        ffn_hidden_dim: SwiGLU FFN 中间维度
        num_experts: 专家数量 E
        num_active_experts: Top-K 激活专家数
        dropout: Dropout 概率
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        ffn_hidden_dim: int,
        num_experts: int = 4,
        num_active_experts: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts

        # 共享路由器: 基于所有头的拼接输入做路由决策
        self.router = nn.Linear(num_heads * hidden_dim, num_experts, bias=False)

        # E 个专家，每个专家有独立的 per-head 权重
        # 权重形状: (E, N, D_in, D_out)
        self.w1 = nn.Parameter(
            torch.empty(num_experts, num_heads, hidden_dim, ffn_hidden_dim)
        )
        self.w_gate = nn.Parameter(
            torch.empty(num_experts, num_heads, hidden_dim, ffn_hidden_dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(num_experts, num_heads, ffn_hidden_dim, hidden_dim)
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._aux_loss: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.router.weight)
        for e in range(self.num_experts):
            for n in range(self.num_heads):
                nn.init.xavier_uniform_(self.w1[e, n])
                nn.init.xavier_uniform_(self.w_gate[e, n])
                nn.init.xavier_uniform_(self.w2[e, n])

    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        return self._aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N, D) — N 个头的输入

        Returns:
            (batch, N, D) — MoE 后的输出
        """
        batch_size, N, D = x.shape

        # 1. 路由: 使用所有头拼接后的特征做决策
        x_flat = x.reshape(batch_size, N * D)  # (B, N*D)
        router_logits = self.router(x_flat)  # (B, E)

        # 2. Top-K 选择
        topk_logits, topk_indices = torch.topk(
            router_logits, self.num_active_experts, dim=-1
        )  # (B, K), (B, K)
        topk_weights = F.softmax(topk_logits, dim=-1)  # (B, K)

        # 3. 辅助损失
        if self.training:
            expert_mask = torch.zeros_like(router_logits)
            expert_mask.scatter_(1, topk_indices, 1.0)
            tokens_per_expert = expert_mask.float().mean(dim=0)
            router_probs = F.softmax(router_logits, dim=-1)
            avg_probs = router_probs.mean(dim=0)
            self._aux_loss = self.num_experts * (tokens_per_expert * avg_probs).sum()
        else:
            self._aux_loss = None

        # 4. 按专家计算并加权合并
        output = torch.zeros_like(x)  # (B, N, D)

        for k in range(self.num_active_experts):
            expert_idx = topk_indices[:, k]  # (B,)
            weight = topk_weights[:, k]  # (B,)

            for e in range(self.num_experts):
                mask = (expert_idx == e)  # (B,) bool
                if not mask.any():
                    continue

                x_e = x[mask]  # (B_e, N, D)

                # Per-head SwiGLU: einsum 批量计算
                # x_e: (B_e, N, D), w1[e]: (N, D, H)
                h1 = torch.einsum("bnd,ndh->bnh", x_e, self.w1[e])
                gate = torch.einsum("bnd,ndh->bnh", x_e, self.w_gate[e])
                h = self.dropout(h1 * F.silu(gate))
                out_e = torch.einsum("bnh,nhd->bnd", h, self.w2[e])

                output[mask] += weight[mask].unsqueeze(-1).unsqueeze(-1) * out_e

        return output
