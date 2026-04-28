"""
RevIn（可逆实例归一化）与 RMSNorm。用于在 Denoiser 的 token 嵌入后做 RevIn、在 Transformer 内用 RMSNorm 替代 LayerNorm。

与 SimDiff 的 NI（时间窗口级、通道空间）正交：NI 在 IndependentNormalizer 中完成，RevIn 仅在 (B, L, d_model) 潜空间上按序列维统计，二者分层协同。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMS 归一化：仅对最后一维用均方根缩放，无均值去中心化。兼容 torch>=2.0 且无 nn.RMSNorm 时亦可使用。"""

    def __init__(self, dim: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        r = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return r


class RevINPatch(nn.Module):
    """
    对 (B, L, d_model) 在序列维 L 上按实例算 μ,σ 并可逆，仿射参量 shape (1,1,d_model)。

    前向用 norm 模式；在 encoder 后对未来段用「与 norm 时同一前向中保存的」统计量做 denorm。该用法与多数量预测代码库中「骨干输出再反归一」一致，便于在潜空间与 out_proj 衔接。
    """

    def __init__(self, d_model: int, eps: float = 1e-4, affine: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, d_model))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def _stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = x.mean(dim=1, keepdim=True)
        v = (x - m).pow(2).mean(dim=1, keepdim=True) + self.eps
        return m, v.sqrt()

    def forward_norm(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> 在 L 上实例归一后可选仿射；保存 μ,σ 供同一次前向的 denorm。"""
        m, s = self._stats(x)
        self._mean, self._std = m, s
        y = (x - m) / s
        if self.affine and self.affine_weight is not None and self.affine_bias is not None:
            y = y * self.affine_weight + self.affine_bias
        return y

    def forward_denorm(self, x: torch.Tensor) -> torch.Tensor:
        """在 encoder 输出上使用 norm 时缓存的 μ,σ 做可逆反变换。"""
        if self._mean is None or self._std is None:
            raise RuntimeError("须先对同一 batch 的序列调用 forward_norm 再对对应输出调用 forward_denorm")
        y = x
        if self.affine and self.affine_weight is not None and self.affine_bias is not None:
            w = (self.affine_weight + self.eps).clamp_min(1e-5)
            y = (y - self.affine_bias) / w
        y = y * self._std + self._mean
        return y

    def clear_cache(self) -> None:
        self._mean = self._std = None


class DenoiserEncoderLayerRMSPre(nn.Module):
    """与 nn.TransformerEncoderLayer(norm_first=True) 等价的子层，但预归一化使用 RMSNorm 而非 LayerNorm。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.rms1 = RMSNorm(d_model)
        self.rms2 = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation: nn.Module = nn.GELU()

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        # 与 nn.TransformerEncoderLayer 的 FFN dropout 布局一致
        h = self.linear2(self.dropout_ff(self.activation(self.linear1(x))))
        return self.dropout2(h)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # pre-norm 与官方 TransformerEncoderLayer(norm_first=True) 一致
        r1 = self.rms1(x)
        x = x + self.dropout1(
            self.self_attn(
                r1,
                r1,
                r1,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
            )[0]
        )
        x = x + self._ff_block(self.rms2(x))
        return x


class HistoryAdditiveBias(nn.Module):
    """
    仅用历史段 token 均值经小 MLP 得到与 d_model 同维偏置，broadcast 加到整段 concat(hist,fut) 上：
    seq <- seq + scale * MLP(mean(hist_emb))。无乘项，末层零初始化使训练初期近似恒等。
    """

    def __init__(self, d_model: int, scale: float = 0.12) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.scale = float(scale)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, seq: torch.Tensor, hist_tokens: int) -> torch.Tensor:
        ht = int(hist_tokens)
        if ht <= 0:
            return seq
        g = seq[:, :ht, :].mean(dim=1)
        b = self.mlp(g).unsqueeze(1)
        return seq + self.scale * b
