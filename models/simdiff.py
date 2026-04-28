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


def point_prediction_from_forecast(out: ForecastOutput, cfg: Config) -> torch.Tensor:
    """根据消融配置选择点预测：ni_only 用 K 次采样算术均值；full / mom_only 用 MoM。"""
    if cfg.simdiff_ablation == "ni_only":
        return out.sample_mean
    return out.mom


@dataclass
class ForecastOutput:
    """单次采样、K 次均值、MoM 三种点预测（原始尺度）；可选 K 条轨迹供 CRPS / 区间图。"""

    single: torch.Tensor
    sample_mean: torch.Tensor
    mom: torch.Tensor
    samples: torch.Tensor | None = None


class SimDiffWeather(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.input_dim <= 0:
            raise ValueError("请先加载数据并设置 cfg.input_dim")
        cfg.validate_mom_config()
        cfg.validate_simdiff_ablation()
        cfg.validate_denoiser_embedding_options()
        if cfg.train_future_marginal_mean is None or cfg.train_future_marginal_std is None:
            raise ValueError("请先运行 make_loaders 以写入 train_future_marginal_mean/std")

        self.cfg = cfg
        _abd = float(getattr(cfg, "hist_add_bias_scale", 0.12))
        if bool(getattr(cfg, "use_hist_add_bias", False)) and bool(cfg.use_rmsnorm):
            _abd = float(getattr(cfg, "hist_add_bias_scale_with_rmsnorm", 0.08))
        self.net = DenoiserTransformer(
            cfg.effective_hist_len(),
            cfg.pred_len,
            cfg.input_dim,
            cfg.d_model,
            cfg.n_heads,
            cfg.n_layers,
            cfg.dropout,
            use_revin=bool(cfg.use_revin),
            use_rmsnorm=bool(cfg.use_rmsnorm),
            use_hist_add_bias=bool(getattr(cfg, "use_hist_add_bias", False)),
            hist_add_bias_scale=_abd,
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
                hist, future, self.cfg.effective_hist_len(), self.cfg.pred_len
            )
        hist_n, st_h = IndependentNormalizer.normalize_history(hist)
        hist_n = self._clip_z(hist_n)
        if self.cfg.simdiff_ablation == "mom_only":
            mu_h, sig_h = st_h["mu_h"], st_h["sig_h"]
            fut_n = (future - mu_h) / sig_h
        else:
            fut_n, _ = IndependentNormalizer.normalize_future(future)
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
            mse_huber_alpha=float(self.cfg.training_noise_mse_huber_alpha),
            huber_beta=float(self.cfg.training_noise_huber_beta),
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
            inference_fp16=bool(self.cfg.forecast_amp),
        )
        lf, c = self.cfg.pred_len, self.cfg.input_dim
        stacked = fut_flat.reshape(b, k, lf, c)
        return stacked

    @torch.no_grad()
    def get_denoise_trajectory_physical(
        self,
        hist: torch.Tensor,
        future: torch.Tensor | None,
        max_points: int | None = None,
    ) -> list[np.ndarray]:
        """
        单条反向扩散链在原始尺度的快照 [ (Lf, C), ... ]；使用 hist的第 0 条样本。
         评估时传入 future[:1] 以便与 forecast 一致地做 NI 反变换。
        """
        mp = int(max_points if max_points is not None else self.cfg.denoise_trajectory_max_points)
        device = hist.device
        b0 = hist[:1]
        hist_n, st_h = IndependentNormalizer.normalize_history(b0)
        hist_n = self._clip_z(hist_n)
        _out = self.diffusion.sample(
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
            sample_debug=False,
            sample_debug_every=int(self.cfg.sample_debug_every),
            return_trajectory=True,
            trajectory_max_points=mp,
            inference_fp16=bool(self.cfg.forecast_amp),
        )
        _, traj_norm = _out
        if self.cfg.simdiff_ablation == "mom_only":
            mu_inv, sig_inv = st_h["mu_h"], st_h["sig_h"]
        else:
            fut0 = future[:1] if future is not None else None
            mu_inv, sig_inv = self._future_mu_sig_for_inverse(fut0, 1, device)
        phys: list[np.ndarray] = []
        for tn in traj_norm:
            x = IndependentNormalizer.inverse_transform_future(tn, mu_inv, sig_inv)
            phys.append(x[0].detach().cpu().numpy())
        return phys

    @torch.no_grad()
    def forecast(
        self,
        hist: torch.Tensor,
        future: torch.Tensor | None = None,
        num_samples: int | None = None,
        num_groups: int | None = None,
        return_samples: bool = False,
    ) -> ForecastOutput:
        """
        推理：K 次采样 → MoM（M 组组均值的中位数，可与低温加权凸组合）+ 单次与全均值。
        future 可选；评估时传入真值以便用本窗 μ_f,σ_f 反变换（模型仍只消费 hist）。
        return_samples=True 时额外返回 samples (B,K,Lf,C) 原始尺度，用于 CRPS / 预测区间。
        """
        device = hist.device
        b = hist.shape[0]
        k = int(num_samples if num_samples is not None else self.cfg.forecast_num_samples)
        m = int(num_groups if num_groups is not None else self.cfg.mom_num_groups)
        if k == 1:
            m = 1
        elif k % m != 0:
            raise ValueError(f"K={k} 必须能被 M={m} 整除")

        hist_n, st_h = IndependentNormalizer.normalize_history(hist)
        hist_n = self._clip_z(hist_n)

        stacked = self._sample_k_trajectories_norm(hist_n, k)
        single_n, mean_n, mom_n = mom_aggregate_normalized(
            stacked,
            m,
            cold_bias_blend=float(self.cfg.mom_cold_bias_blend),
            cold_sharpness=float(self.cfg.mom_cold_sharpness),
        )

        if self.cfg.simdiff_ablation == "mom_only":
            mu_inv, sig_inv = st_h["mu_h"], st_h["sig_h"]
        else:
            mu_inv, sig_inv = self._future_mu_sig_for_inverse(future, b, device)
        single = IndependentNormalizer.inverse_transform_future(single_n, mu_inv, sig_inv)
        sample_mean = IndependentNormalizer.inverse_transform_future(mean_n, mu_inv, sig_inv)
        mom = IndependentNormalizer.inverse_transform_future(mom_n, mu_inv, sig_inv)

        samples_phys: torch.Tensor | None = None
        if return_samples:
            bk = b * k
            _, _, lf, c = stacked.shape
            sk = stacked.reshape(bk, lf, c)
            mu_bk = mu_inv.repeat_interleave(k, dim=0)
            sig_bk = sig_inv.repeat_interleave(k, dim=0)
            samples_phys = IndependentNormalizer.inverse_transform_future(
                sk, mu_bk, sig_bk
            ).reshape(b, k, lf, c)

        return ForecastOutput(
            single=single, sample_mean=sample_mean, mom=mom, samples=samples_phys
        )

    @torch.no_grad()
    def validation_mse(
        self,
        hist: torch.Tensor,
        future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forecast(hist, future=future)
        pred = point_prediction_from_forecast(out, self.cfg)
        mse = torch.mean((pred - future) ** 2)
        mae = torch.mean(torch.abs(pred - future))
        return mse, mae
