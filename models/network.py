"""
条件去噪网络：历史与未来 token 拼接，时间步正弦嵌入 + Transformer 编码，
输出与未来序列同形状的预测噪声 eps。

默认：在 token 嵌入后插入 RevIn（可逆）、Transformer 子层中 LayerNorm 换为 RMSNorm；数据层的 NI 仍在 IndependentNormalizer 中独立执行。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.revin_rms import DenoiserEncoderLayerRMSPre, RevINPatch


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
        use_revin: bool = True,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.d_model = d_model
        self.use_revin = use_revin
        self.use_rmsnorm = use_rmsnorm

        self.in_proj = nn.Linear(channels, d_model)
        half = d_model // 2
        t_scale = torch.exp(torch.arange(0, half, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
        self.register_buffer("t_emb_scale", t_scale)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.pos_h = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.pos_f = nn.Parameter(torch.zeros(1, pred_len, d_model))

        if use_revin:
            self.revin = RevINPatch(d_model, eps=1e-5, affine=True)
        else:
            self.revin = None  # type: ignore[assignment]

        if use_rmsnorm:
            self.encoder = self._build_rms_encoder(
                d_model, n_heads, n_layers, dropout, dim_feedforward=d_model * 4
            )
        else:
            enc = nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.out_proj = nn.Linear(d_model, channels)

    @staticmethod
    def _build_rms_encoder(
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        dim_feedforward: int,
    ) -> nn.ModuleList[nn.Module]:
        return nn.ModuleList(
            [
                DenoiserEncoderLayerRMSPre(
                    d_model,
                    n_heads,
                    dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def _timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,)整数时间步 -> (B, d_model)"""
        b = t.shape[0]
        device = t.device
        half = self.t_emb_scale.shape[0]
        ang = t.float().unsqueeze(1) * self.t_emb_scale.unsqueeze(0).to(device)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        if emb.shape[1] < self.d_model:
            emb = nn.functional.pad(emb, (0, self.d_model - emb.shape[1]))
        elif emb.shape[1] > self.d_model:
            emb = emb[:, : self.d_model]
        return self.time_mlp(emb)

    def _encode(self, seq: torch.Tensor) -> torch.Tensor:
        if self.use_rmsnorm:
            assert isinstance(self.encoder, nn.ModuleList)
            h = seq
            for layer in self.encoder:
                h = layer(h, src_mask=None, src_key_padding_mask=None)
            return h
        return self.encoder(seq)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, Lf, C) 加噪的未来
        hist: (B, Lh, C) 已归一化的历史
        """
        te = self._timestep_embedding(t)[:, None, :]
        h = self.in_proj(hist) + self.pos_h + te
        f = self.in_proj(x_t) + self.pos_f + te
        seq = torch.cat([h, f], dim=1)
        if self.use_revin and self.revin is not None:
            seq = self.revin.forward_norm(seq)
        out = self._encode(seq)
        out_f = out[:, self.seq_len :, :]
        if self.use_revin and self.revin is not None:
            out_f = self.revin.forward_denorm(out_f)
        return self.out_proj(out_f)
