"""
条件去噪网络：历史 + 加噪未来 token 拼接；可选 Patch 分块嵌入与 RoPE。
输出与未来序列同形状 (B, pred_len, C) 的预测噪声 eps。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope_mha import build_encoder_stack


def _pad1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """x: (B, L, C) 右侧零填充到 target_len。"""
    b, l, c = x.shape
    if l >= target_len:
        return x
    return F.pad(x, (0, 0, 0, target_len - l), value=0.0)


def _patch_padded_len(length: int, patch_size: int, stride: int) -> int:
    """为覆盖原序列的 patch 起止，将 L 补到 Lp，使 n = ceil((L-P)/S)+1 个 patch 且最后一块覆盖到 Lp。"""
    p, s = patch_size, stride
    if length <= p:
        return p
    n = int(math.ceil((length - p) / s)) + 1
    return (n - 1) * s + p


def _to_patches(
    x: torch.Tensor, patch_size: int, stride: int
) -> torch.Tensor:
    """
    x: (B, L, C) -> 右侧填充后 -> (B, n, P*C) token 行。
    """
    b, l, c = x.shape
    p, s = patch_size, stride
    lpad = _patch_padded_len(l, p, s)
    if lpad > l:
        x = _pad1d(x, lpad)
    n = 1 + (x.size(1) - p) // s
    if n < 1:
        raise ValueError("无法构造 patch，请调大序列或调小 patch_size")
    out = []
    for k in range(n):
        st = k * s
        seg = x[:, st : st + p, :].reshape(b, 1, p * c)
        out.append(seg)
    return torch.cat(out, dim=1)


class DenoiserTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        use_patch: bool = False,
        patch_size: int = 8,
        patch_stride: int | None = None,
        use_rope: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.d_model = d_model
        self.use_patch = bool(use_patch)
        self.use_rope = bool(use_rope)
        self.ps = int(patch_size)
        self.stride = int(patch_stride) if patch_stride is not None else self.ps
        if self.ps < 1 or self.stride < 1:
            raise ValueError("patch_size/stride 须 >= 1")

        self.in_proj = nn.Linear(channels, d_model)
        self.patch_in = (
            nn.Linear(self.ps * channels, d_model) if self.use_patch else None
        )

        half = d_model // 2
        t_scale = torch.exp(
            torch.arange(0, half, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1))
        )
        self.register_buffer("t_emb_scale", t_scale)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        if self.use_patch:
            n_h, n_f = self._count_patch_tokens(self.seq_len, self.pred_len)
            n_tok = n_h + n_f
            if not use_rope:
                self.pos_patch: nn.Parameter | None = nn.Parameter(
                    torch.zeros(1, n_tok, d_model)
                )
            else:
                self.pos_patch = None
        else:
            self.pos_patch = None
            if not use_rope:
                self.pos_h = nn.Parameter(torch.zeros(1, seq_len, d_model))
                self.pos_f = nn.Parameter(torch.zeros(1, pred_len, d_model))
            else:
                self.pos_h = None
                self.pos_f = None

        self.encoder_blocks = build_encoder_stack(
            d_model, n_heads, n_layers, dropout, use_rope=use_rope
        )
        if self.use_patch:
            n_h, n_f = self._count_patch_tokens(self.seq_len, self.pred_len)
            self.patch_decode = nn.Linear(d_model, self.ps * channels)
        else:
            n_h, n_f = 0, 0
        self.n_hist_tokens = n_h
        self.n_fut_tokens = n_f
        self.out_proj = (
            nn.Linear(d_model, channels) if not self.use_patch else None
        )  # type: ignore[assignment]
        if self.use_patch:
            f_pad = _patch_padded_len(self.pred_len, self.ps, self.stride)
            self.fut_pad_len = f_pad
            self.fut_n_tokens = 1 + (f_pad - self.ps) // self.stride
        else:
            self.fut_pad_len = pred_len
            self.fut_n_tokens = 0

    def _count_patch_tokens(self, lh: int, lf: int) -> tuple[int, int]:
        p, s = self.ps, self.stride
        h_pad = _patch_padded_len(lh, p, s)
        f_pad = _patch_padded_len(lf, p, s)
        n_h = 1 + (h_pad - p) // s
        n_f = 1 + (f_pad - p) // s
        return n_h, n_f

    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        b = t.shape[0]
        device = t.device
        half = self.t_emb_scale.shape[0]
        ang = t.float().unsqueeze(1) * self.t_emb_scale.unsqueeze(0).to(device)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if emb.shape[1] < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.shape[1]))
        elif emb.shape[1] > self.d_model:
            emb = emb[:, : self.d_model]
        return self.time_mlp(emb)

    def _forward_patch(self, x_t: torch.Tensor, t: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        b = hist.shape[0]
        p, s = self.ps, self.stride
        h_tok = _to_patches(hist, p, s)
        f_tok = _to_patches(x_t, p, s)
        te = self._timestep_embedding(t)[:, None, :]
        h_e = self.patch_in(h_tok) + te  # type: ignore[operator]
        f_e = self.patch_in(f_tok) + te
        if self.pos_patch is not None:
            n_tok = h_e.size(1) + f_e.size(1)
            h_e = h_e + self.pos_patch[:, : h_e.size(1), :]
            f_e = f_e + self.pos_patch[:, h_e.size(1) : n_tok, :]
        seq = torch.cat([h_e, f_e], dim=1)
        for blk in self.encoder_blocks:
            seq = blk(seq)
        n_h = h_e.size(1)
        out_tok = seq[:, n_h:, :]
        fl = self.patch_decode(out_tok)
        c = self.channels
        segs: list[torch.Tensor] = []
        for j in range(out_tok.size(1)):
            segs.append(fl[:, j, :].view(b, p, c))
        # 重叠时按起址叠加平均
        f_pad = _patch_padded_len(self.pred_len, p, s)
        buf = torch.zeros(b, f_pad, c, device=hist.device, dtype=hist.dtype)
        w = torch.zeros(b, f_pad, 1, device=hist.device, dtype=hist.dtype)
        for j in range(len(segs)):
            st = j * s
            buf[:, st : st + p, :] += segs[j]
            w[:, st : st + p, 0] += 1.0
        w = w.clamp(min=1e-6)
        out = buf / w
        return out[:, : self.pred_len, :]

    def _forward_token(self, x_t: torch.Tensor, t: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        te = self._timestep_embedding(t)[:, None, :]
        h = self.in_proj(hist) + te
        f = self.in_proj(x_t) + te
        if not self.use_rope:
            h = h + self.pos_h
            f = f + self.pos_f
        seq = torch.cat([h, f], dim=1)
        for blk in self.encoder_blocks:
            seq = blk(seq)
        out = seq[:, self.seq_len :, :]
        assert self.out_proj is not None
        return self.out_proj(out)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        if self.use_patch:
            return self._forward_patch(x_t, t, hist)
        return self._forward_token(x_t, t, hist)
