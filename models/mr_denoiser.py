"""
轻量多尺度风格 1D 卷积去噪网（作 mr-Diff 对照；与训练日程、NI/MoM 等外围一致）。
前向: (x_t, t, hist) -> 与 x_t 同形状 eps 预测。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MrDenoiser(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        d_model: int,
        n_heads: int,  # 接口对齐，本模型未用
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        _ = n_heads, n_layers
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = c = channels
        h = 128
        self.h = h
        half = d_model // 2
        t_scale = torch.exp(
            torch.arange(0, half, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1))
        )
        self.register_buffer("t_emb_scale", t_scale)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, h),
        )
        self.hist_fc = nn.Linear(seq_len * c, h)
        self.in_conv = nn.Conv1d(c, h, kernel_size=3, padding=1)
        def _dilated_block(d1: int, d2: int) -> nn.Sequential:
            # padding = dilation * (k-1)//2 保持长度与输入一致（k=3）
            p1, p2 = d1 * (3 - 1) // 2, d2 * (3 - 1) // 2
            return nn.Sequential(
                nn.Conv1d(h, h, 3, padding=p1, dilation=d1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(h, h, 3, padding=p2, dilation=d2),
                nn.GELU(),
            )

        self.blocks = nn.ModuleList([_dilated_block(1, 2) for _ in range(2)])
        self.out = nn.Conv1d(h, c, 1)

    def _temb(self, t: torch.Tensor) -> torch.Tensor:
        half = self.t_emb_scale.shape[0]
        device = t.device
        d_model = 2 * half
        ang = t.float().unsqueeze(1) * self.t_emb_scale.unsqueeze(0).to(device)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if emb.shape[1] < d_model:
            emb = F.pad(emb, (0, d_model - emb.shape[1]))
        elif emb.shape[1] > d_model:
            emb = emb[:, :d_model]
        return self.time_mlp(emb)[:, :, None]  # (B, h, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        b, lf, c = x_t.shape
        te = self._temb(t)  # (B,h,1)
        hvec = F.gelu(self.hist_fc(hist.reshape(b, -1)))[:, :, None]  # (B,h,1)
        x = self.in_conv(x_t.permute(0, 2, 1)) + te + hvec
        for bl in self.blocks:
            y = bl(x)
            x = x + y
        return self.out(x).permute(0, 2, 1)
