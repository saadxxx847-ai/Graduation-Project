"""
RoPE 自注意力：用于在Transformer块中替代可学习绝对位置编码。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_1d(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: (B, n_heads, L, d_h)
    cos, sin: (L, d_h) 或 (1, 1, L, d_h)
    """
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    q_ = (q * cos) + (_rotate_half(q) * sin)
    k_ = (k * cos) + (_rotate_half(k) * sin)
    return q_, k_


def rope_cos_sin(
    seq_len: int,
    d_head: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = d_head
    if d % 2 != 0:
        raise ValueError(f"RoPE 需要 d_head 为偶数，当前 {d_head}")
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
    freqs = torch.einsum("i,j->ij", pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 须能被 n_heads 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if self.d_head % 2 != 0:
            raise ValueError("d_model//n_heads 须为偶数以使用 RoPE")
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, L, d_model)
        """
        b, l, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, l, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = k.view(b, l, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.view(b, l, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        cos, sin = rope_cos_sin(
            l, self.d_head, x.device, x.dtype, base=10000.0
        )
        q, k = apply_rope_1d(q, k, cos, sin)
        # SDPA: (B, h, L, d_h) @ (B, h, d_h, L) -> (B, h, L, L)
        o = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        o = o.permute(0, 2, 1, 3).contiguous().reshape(b, l, d)
        return self.out(o)


class RoPEEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float, norm_first: bool = True):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = RoPESelfAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            h = x + self.dropout1(self.self_attn(self.norm1(x)))
            return h + self.dropout2(self.ffn(self.norm2(h)))
        h = self.norm1(x + self.dropout1(self.self_attn(x)))
        return self.norm2(h + self.dropout2(self.ffn(h)))


class StandardEncoderBlock(nn.Module):
    """可学习/无位置偏置的 Pre-LN Transformer 块；位置由外部与 token 相加。"""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float, norm_first: bool = True):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x_ = self.norm1(x)
            a, _ = self.self_attn(x_, x_, x_, need_weights=False)
            h = x + self.dropout1(a)
            return h + self.dropout2(self.ffn(self.norm2(h)))
        a, _ = self.self_attn(x, x, x, need_weights=False)
        h = self.norm1(x + self.dropout1(a))
        return self.norm2(h + self.dropout2(self.ffn(h)))


def build_encoder_stack(
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    use_rope: bool,
) -> nn.ModuleList:
    dim_ff = d_model * 4
    blocks: list[nn.Module] = []
    for _ in range(n_layers):
        if use_rope:
            blocks.append(RoPEEncoderBlock(d_model, n_heads, dim_ff, dropout, norm_first=True))
        else:
            blocks.append(StandardEncoderBlock(d_model, n_heads, dim_ff, dropout, norm_first=True))
    return nn.ModuleList(blocks)
