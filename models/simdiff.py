"""
SimDiff-Weather：Normalization Independence + Median-of-Means 集成预测。

* 历史与未来分别在各自时间维上估计 μ,σ，互不混用；网络条件为 normalize_history(hist)。
* 扩散目标在 normalize_future(future) 空间；评估时有真值则用 batch 的 μ_f,σ_f 反变换，无真值则用训练集未来边际统计量。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from config.config import Config
from models.diffusion import GaussianDiffusion
from models.network import DenoiserTransformer
from utils.independent_normalizer import IndependentNormalizer, mom_aggregate_normalized


@dataclass
class ForecastOutput:
    """单次采样、K 次均值、MoM 三种点预测（原始尺度）。"""

    single: torch.Tensor
    sample_mean: torch.Tensor
    mom: torch.Tensor


class SimDiffWeather(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.input_dim <= 0:
            raise ValueError("请先加载数据并设置 cfg.input_dim")
        cfg.validate_mom_config()
        if cfg.train_future_marginal_mean is None or cfg.train_future_marginal_std is None:
            raise ValueError("请先运行 make_loaders 以写入 train_future_marginal_mean/std")

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

        fm = torch.as_tensor(cfg.train_future_marginal_mean, dtype=torch.float32).reshape(1, 1, -1)
        fs = torch.as_tensor(cfg.train_future_marginal_std, dtype=torch.float32).reshape(1, 1, -1)
        self.register_buffer("_fut_mu_marginal", fm)
        self.register_buffer("_fut_sig_marginal", fs)

    def _clip_z(self, x: torch.Tensor) -> torch.Tensor:
        z = self.cfg.z_clip
        if z is None or z <= 0:
            return x
        return x.clamp(-z, z)

    def _future_mu_sig_for_inverse(
        self,
        future: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """有真值 future 时用本 batch 的 μ_f,σ_f；否则用训练集未来边际（1,1,C）广播。"""
        if future is not None:
            _, st = IndependentNormalizer.normalize_future(future)
            return st["mu_f"], st["sig_f"]
        mu = self._fut_mu_marginal.to(device).expand(batch_size,1, -1)
        sig = self._fut_sig_marginal.to(device).expand(batch_size, 1, -1)
        return mu, sig

    def training_loss(self, hist: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        if self.cfg.debug_norm_assert:
            IndependentNormalizer.debug_assert_shapes_and_idempotent_history(
                hist, future, self.cfg.seq_len, self.cfg.pred_len
            )
        hist_n, _ = IndependentNormalizer.normalize_history(hist)
        fut_n, _ = IndependentNormalizer.normalize_future(future)
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
    def _sample_k_trajectories_norm(
        self,
        hist_n: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """同一 hist 条件重复 K 次独立扩散采样，返回 (B, K, L, C) 归一化空间。"""
        device = hist_n.device
        b = hist_n.shape[0]
        hist_rep = hist_n.repeat_interleave(k, dim=0)
        fut_flat = self.diffusion.sample(
            self.net,
            hist_rep,
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
        lf, c = self.cfg.pred_len, self.cfg.input_dim
        stacked = fut_flat.reshape(b, k, lf, c)
        return stacked

    @torch.no_grad()
    def forecast(
        self,
        hist: torch.Tensor,
        future: torch.Tensor | None = None,
        num_samples: int | None = None,
        num_groups: int | None = None,
    ) -> ForecastOutput:
        """
        推理：K 次采样 → MoM（M 组组均值的中位数）+ 单次与全均值。
        future 可选；评估时传入真值以便用本窗 μ_f,σ_f 反变换（模型仍只消费 hist）。
        """
        device = hist.device
        b = hist.shape[0]
        k = int(num_samples if num_samples is not None else self.cfg.forecast_num_samples)
        m = int(num_groups if num_groups is not None else self.cfg.mom_num_groups)
        if k == 1:
            m = 1
        elif k % m != 0:
            raise ValueError(f"K={k} 必须能被 M={m} 整除")

        hist_n, _ = IndependentNormalizer.normalize_history(hist)
        hist_n = self._clip_z(hist_n)

        stacked = self._sample_k_trajectories_norm(hist_n, k)
        single_n, mean_n, mom_n = mom_aggregate_normalized(stacked, m)

        mu_f, sig_f = self._future_mu_sig_for_inverse(future, b, device)
        single = IndependentNormalizer.inverse_transform_future(single_n, mu_f, sig_f)
        sample_mean = IndependentNormalizer.inverse_transform_future(mean_n, mu_f, sig_f)
        mom = IndependentNormalizer.inverse_transform_future(mom_n, mu_f, sig_f)
        return ForecastOutput(single=single, sample_mean=sample_mean, mom=mom)

    @torch.no_grad()
    def validation_mse(
        self,
        hist: torch.Tensor,
        future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forecast(hist, future=future)
        mse = torch.mean((out.mom - future) ** 2)
        mae = torch.mean(torch.abs(out.mom - future))
        return mse, mae
