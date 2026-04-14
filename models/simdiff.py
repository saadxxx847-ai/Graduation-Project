"""
SimDiff-Weather：条件扩散训练与采样预测。

默认在训练集上拟合「全局 per-channel」均值/方差做标准化，避免滑动窗口内近常数特征导致 σ→0、归一化数值爆炸（表现为 train loss 数万、测试 MSE 1e12 量级）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config.config import Config
from models.diffusion import GaussianDiffusion
from models.network import DenoiserTransformer
from utils.normalizer import Normalizer


class SimDiffWeather(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.input_dim <= 0:
            raise ValueError("请先加载数据并设置 cfg.input_dim")
        self.cfg = cfg
        self.net = DenoiserTransformer(
            cfg.seq_len,
            cfg.pred_len,
            cfg.input_dim,
            cfg.d_model,
            cfg.n_heads,
            cfg.n_layers,
            cfg.dropout,
        )
        self.diffusion = GaussianDiffusion(cfg.timesteps, cfg.cosine_s)
        self._sample_clamp_abs = float(cfg.z_clip) + 2.0

        if cfg.use_global_standardization and cfg.global_mean is not None and cfg.global_std is not None:
            gm = torch.as_tensor(cfg.global_mean, dtype=torch.float32).reshape(1, 1, -1)
            gs = torch.as_tensor(cfg.global_std, dtype=torch.float32).reshape(1, 1, -1)
            self.register_buffer("_g_mean", gm)
            self.register_buffer("_g_std", gs)
            self._use_global = True
        else:
            self.register_buffer("_g_mean", torch.zeros(1, 1, cfg.input_dim))
            self.register_buffer("_g_std", torch.ones(1, 1, cfg.input_dim))
            self._use_global = False

    def _indep_fut(self) -> bool:
        return self.cfg.independent_future_normalization

    def _normalize_training_pair(
        self, hist: torch.Tensor, future: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self._use_global:
            gm = self._g_mean.to(hist.device)
            gs = self._g_std.to(hist.device)
            hist_n = (hist - gm) / gs
            if self._indep_fut():
                f_mean = future.mean(dim=1, keepdim=True)
                f_std = future.std(dim=1, keepdim=True).clamp(min=1e-5)
                fut_n = (future - f_mean) / f_std
                stats = {"f_mean": f_mean, "f_std": f_std, "mode": "global_hist_indep_fut"}
            else:
                fut_n = (future - gm) / gs
                stats = {"f_mean": gm, "f_std": gs, "mode": "global_both"}
            return hist_n, fut_n, stats
        return Normalizer.normalize_pair(hist, future, independent_future=self._indep_fut())

    def _clip_z(self, x: torch.Tensor) -> torch.Tensor:
        z = self.cfg.z_clip
        if z is None or z <= 0:
            return x
        return x.clamp(-z, z)

    def training_loss(self, hist: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        hist_n, fut_n, _ = self._normalize_training_pair(hist, future)
        hist_n = self._clip_z(hist_n)
        fut_n = self._clip_z(fut_n)
        b = hist.shape[0]
        device = hist.device
        t = torch.randint(0, self.diffusion.timesteps, (b,), device=device, dtype=torch.long)
        return self.diffusion.training_losses(
            self.net,
            fut_n,
            t,
            hist_n,
            l1_weight=float(self.cfg.training_noise_l1_weight),
            temporal_diff_weight=float(self.cfg.training_noise_temporal_diff_weight),
        )

    @torch.no_grad()
    def forecast(self, hist: torch.Tensor) -> torch.Tensor:
        """原始尺度下的未来预测 (B, pred_len, C)。"""
        device = hist.device
        if self._use_global:
            gm = self._g_mean.to(device)
            gs = self._g_std.to(device)
            hist_n = self._clip_z((hist - gm) / gs)
            fut_n = self.diffusion.sample(
                self.net,
                hist_n,
                self.cfg.pred_len,
                self.cfg.input_dim,
                device,
                clamp_abs=self._sample_clamp_abs,
                sampling_mode=self.cfg.sampling_mode,
                sampling_steps=self.cfg.sampling_steps,
                ddim_eta=float(self.cfg.ddim_eta),
                clip_pred_x0=bool(self.cfg.sample_clip_pred_x0),
                sample_debug=bool(self.cfg.sample_debug),
                sample_debug_every=int(self.cfg.sample_debug_every),
            )
            b, lf, _ = fut_n.shape
            return fut_n * gs.expand(b, lf, -1) + gm.expand(b, lf, -1)

        h_mean = hist.mean(dim=1, keepdim=True)
        h_std = hist.std(dim=1, keepdim=True).clamp(min=1e-5)
        hist_n = (hist - h_mean) / h_std
        fut_n = self.diffusion.sample(
            self.net,
            hist_n,
            self.cfg.pred_len,
            self.cfg.input_dim,
            device,
            clamp_abs=self._sample_clamp_abs,
            sampling_mode=self.cfg.sampling_mode,
            sampling_steps=self.cfg.sampling_steps,
            ddim_eta=float(self.cfg.ddim_eta),
            clip_pred_x0=bool(self.cfg.sample_clip_pred_x0),
            sample_debug=bool(self.cfg.sample_debug),
            sample_debug_every=int(self.cfg.sample_debug_every),
        )
        f_mean, f_std = Normalizer.infer_future_stats_from_hist(hist)
        b, lf, c = fut_n.shape
        mu = f_mean.expand(-1, lf, -1)
        sig = f_std.expand(-1, lf, -1)
        return Normalizer.denormalize_future(fut_n, mu, sig)

    @torch.no_grad()
    def validation_mse(
        self,
        hist: torch.Tensor,
        future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hist_n, _, stats = self._normalize_training_pair(hist, future)
        hist_n = self._clip_z(hist_n)
        device = hist.device
        fut_n = self.diffusion.sample(
            self.net,
            hist_n,
            self.cfg.pred_len,
            self.cfg.input_dim,
            device,
            clamp_abs=self._sample_clamp_abs,
            sampling_mode=self.cfg.sampling_mode,
            sampling_steps=self.cfg.sampling_steps,
            ddim_eta=float(self.cfg.ddim_eta),
            clip_pred_x0=bool(self.cfg.sample_clip_pred_x0),
            sample_debug=bool(self.cfg.sample_debug),
            sample_debug_every=int(self.cfg.sample_debug_every),
        )
        lf = fut_n.shape[1]
        mu = stats["f_mean"].expand(-1, lf, -1)
        sig = stats["f_std"].expand(-1, lf, -1)
        pred = Normalizer.denormalize_future(fut_n, mu, sig)
        mse = torch.mean((pred - future) ** 2)
        mae = torch.mean(torch.abs(pred - future))
        return mse, mae
