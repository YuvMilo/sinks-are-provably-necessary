"""Model architectures for the trigger-conditional averaging task."""

from __future__ import annotations

import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    """One-layer single-head self-attention (softmax or ReLU)."""

    def __init__(self, d_model: int, attn_type: str = "softmax"):
        super().__init__()
        assert attn_type in {"softmax", "relu"}
        self.d = d_model
        self.attn_type = attn_type
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        for m in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        q = self.W_Q(x)
        k = self.W_K(x)
        logits = torch.matmul(q, k.transpose(1, 2))

        L = logits.shape[-1]
        mask = torch.ones(L, L, device=logits.device, dtype=torch.bool).triu(1)

        if self.attn_type == "softmax":
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
            attn = torch.softmax(logits, dim=-1)
        else:
            logits = logits.masked_fill(mask, 0.0)
            attn = torch.relu(logits)
            row_counts = torch.arange(L, device=attn.device, dtype=attn.dtype).clamp_min(1)
            attn = attn / row_counts.view(1, L, 1)

        v = self.W_V(x)
        ctx = torch.matmul(attn, v)
        y = self.W_O(ctx)
        if return_attn:
            return y, attn
        return y


class MultiHeadAttentionLayer(nn.Module):
    """Multi-head self-attention layer (softmax or ReLU).

    Each head has full d_model dimension.
    """

    def __init__(self, d_model: int, num_heads: int, attn_type: str = "softmax"):
        super().__init__()
        assert attn_type in {"softmax", "relu"}
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model
        self.attn_type = attn_type

        self.W_Q = nn.Linear(d_model, num_heads * d_model, bias=False)
        self.W_K = nn.Linear(d_model, num_heads * d_model, bias=False)
        self.W_V = nn.Linear(d_model, num_heads * d_model, bias=False)
        self.W_O = nn.Linear(num_heads * d_model, d_model, bias=False)
        for m in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, L, D = x.shape

        q = self.W_Q(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(2, 3))

        mask = torch.ones(L, L, device=logits.device, dtype=torch.bool).triu(1)

        if self.attn_type == "softmax":
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
            attn = torch.softmax(logits, dim=-1)
        else:
            logits = logits.masked_fill(mask, 0.0)
            attn = torch.relu(logits)
            row_counts = torch.arange(L, device=attn.device, dtype=attn.dtype).clamp_min(1)
            attn = attn / row_counts.view(1, 1, L, 1)

        v = self.W_V(x).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.d_head)
        y = self.W_O(ctx)
        if return_attn:
            return y, attn
        return y


class MultiLayerTransformer(nn.Module):
    """Multi-layer transformer with multi-head attention and residual connections."""

    def __init__(self, d_model: int, num_heads: int, num_layers: int,
                 attn_type: str = "softmax"):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(d_model, num_heads, attn_type)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            all_attn = []
            for layer in self.layers:
                residual = x
                x, attn = layer(x, return_attn=True)
                x = x + residual
                all_attn.append(attn)
            return x, all_attn
        else:
            for layer in self.layers:
                x = layer(x) + x
            return x
