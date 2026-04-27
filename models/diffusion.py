"""
高斯扩散：余弦噪声日程、前向 q_sample、训练用噪声 MSE（预测 eps）。
"""
from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inference_ctx(device: torch.device, use_fp16: bool):
    if use_fp16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return nullcontext()


def _trajectory_save_indices(n_inner_steps: int, max_points: int) -> set[int]:
    """在0..n_inner_steps-1 内均匀取点，记录「完成该步反向更新后」的 x。"""
    if n_inner_steps <= 0:
        return set()
    m = min(max(max_points, 2) - 1, n_inner_steps)
    idx = np.linspace(0, n_inner_steps - 1, m, dtype=np.int64)
    return set(int(x) for x in idx.tolist())


def build_ddim_time_pairs(t_total: int, sampling_steps: int | None) -> list[tuple[int, int]]:
    """
    构造 DDIM 的 (t, t_prev) 序列，保证时间严格递减、终点为 0。
    当 sampling_steps > t_total 时，离散日程上无法对应更多「独立」时间索引，
    旧实现用 linspace 会产生大量 (t,t) 重复并被跳过，有效去噪步数远少于预期，易发散。
    """
    s = t_total if sampling_steps is None else int(sampling_steps)
    if s <= 0:
        s = t_total
    if s > t_total:
        s = t_total
    idx = np.linspace(0, t_total - 1, s + 1).astype(np.float64)
    tim = np.round(t_total - 1 - idx).astype(np.int64)
    tim = np.clip(tim, 0, t_total - 1)
    clean: list[int] = [int(tim[0])]
    for t in tim[1:]:
        t = int(t)
        if t != clean[-1]:
            clean.append(t)
    if clean[-1] != 0:
        clean.append(0)
    pairs: list[tuple[int, int]] = []
    for i in range(len(clean) - 1):
        a, b = clean[i], clean[i + 1]
        if a > b:
            pairs.append((a, b))
    return pairs


def cosine_beta_schedule(timesteps: int, s: float = 5.0) -> np.ndarray:
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int, cosine_s: float = 5.0):
        super().__init__()
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps, cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / np.maximum(1e-20, 1.0 - alphas_cumprod)

        to_torch = lambda arr: torch.tensor(arr, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer("posterior_variance", to_torch(posterior_variance))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def training_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        hist: torch.Tensor,
        l1_weight: float = 0.0,
        temporal_diff_weight: float = 0.0,
        mse_huber_alpha: float = 1.0,
        huber_beta: float = 1.0,
    ) -> torch.Tensor:
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t, hist)
        a = float(mse_huber_alpha)
        b = float(huber_beta)
        if a >= 1.0 - 1e-8:
            main = F.mse_loss(pred, noise)
        elif a <= 1e-8:
            main = F.smooth_l1_loss(pred, noise, beta=b)
        else:
            main = a * F.mse_loss(pred, noise) + (1.0 - a) * F.smooth_l1_loss(
                pred, noise, beta=b
            )
        loss = main
        if l1_weight > 0:
            loss = loss + l1_weight * F.l1_loss(pred, noise)
        if temporal_diff_weight > 0 and x0.shape[1] >= 2:
            pd = pred[:, 1:] - pred[:, :-1]
            nd = noise[:, 1:] - noise[:, :-1]
            loss = loss + temporal_diff_weight * F.mse_loss(pd, nd)
        return loss

    @torch.no_grad()
    def p_sample_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: int,
        hist: torch.Tensor,
        clamp_abs: float = 6.0,
        clip_pred_x0: bool = True,
        inference_fp16: bool = False,
    ) -> torch.Tensor:
        """DDPM 单步反向采样；对 pred_x0 裁剪再算后验均值，减轻与 DDIM 类似的数值发散。"""
        b = x.shape[0]
        device = x.device
        dtype = x.dtype
        t_b = torch.full((b,), t, device=device, dtype=torch.long)
        beta_t = self.betas[t]
        alpha_t = 1.0 - beta_t
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        with _inference_ctx(device, inference_fp16):
            eps_theta = model(x, t_b, hist)

        pred_x0 = (x - sqrt_one_minus_ab * eps_theta) / sqrt_alpha_bar.clamp(min=1e-8)
        if clip_pred_x0:
            pred_x0 = pred_x0.clamp(-clamp_abs, clamp_abs)

        alpha_bar = self.alphas_cumprod[t].to(device=device, dtype=dtype)
        alpha_bar_prev = self.alphas_cumprod_prev[t].to(device=device, dtype=dtype)
        coef_x0 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1.0 - alpha_bar).clamp(min=1e-8)
        coef_xt = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar).clamp(min=1e-8)
        mean = coef_x0 * pred_x0 + coef_xt * x

        if t == 0:
            return mean
        noise = torch.randn_like(x)
        sigma = torch.sqrt(self.posterior_variance[t])
        return mean + sigma * noise

    @torch.no_grad()
    def _ddim_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: int,
        t_prev: int,
        hist: torch.Tensor,
        eta: float,
        clamp_abs: float,
        clip_pred_x0: bool,
        inference_fp16: bool = False,
    ) -> torch.Tensor:
        """单步 DDIM：从离散时间 t 到更小的 t_prev（含 t_prev=0）。"""
        b = x.shape[0]
        device = x.device
        dtype = x.dtype
        t_b = torch.full((b,), t, device=device, dtype=torch.long)
        with _inference_ctx(device, inference_fp16):
            eps = model(x, t_b, hist)

        alpha_t = self.alphas_cumprod[t].to(device=device, dtype=dtype)
        alpha_prev = self.alphas_cumprod[t_prev].to(device=device, dtype=dtype)

        if t == t_prev:
            return x.clamp(-clamp_abs, clamp_abs)

        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()
        pred_x0 = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t.clamp(min=1e-8)
        if clip_pred_x0:
            pred_x0 = pred_x0.clamp(-clamp_abs, clamp_abs)

        if eta > 0 and t_prev >= 0:
            sigma = eta * (
                (1.0 - alpha_prev)
                / (1.0 - alpha_t).clamp(min=1e-8)
                * (1.0 - alpha_t / alpha_prev.clamp(min=1e-8))
            ).sqrt()
        else:
            sigma = torch.zeros((), device=device, dtype=dtype)

        c = (1.0 - alpha_prev - sigma * sigma).clamp(min=0.0).sqrt()
        noise = torch.randn_like(x) if float(eta) > 0 else torch.zeros_like(x)
        out = alpha_prev.sqrt() * pred_x0 + c * eps + sigma * noise
        return out.clamp(-clamp_abs, clamp_abs)

    def _log_sample_tensor(self, tag: str, x: torch.Tensor) -> None:
        print(
            f"  [sample] {tag}  min={x.min().item():.5f}  max={x.max().item():.5f}  "
            f"mean={x.mean().item():.5f}"
        )

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        hist: torch.Tensor,
        pred_len: int,
        channels: int,
        device: torch.device,
        clamp_abs: float = 6.0,
        sampling_mode: str = "ddpm",
        sampling_steps: int | None = None,
        ddim_eta: float = 0.0,
        clip_pred_x0: bool = True,
        sample_debug: bool = False,
        sample_debug_every: int = 20,
        return_trajectory: bool = False,
        trajectory_max_points: int = 36,
        inference_fp16: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """从纯高斯噪声反向采样得到归一化空间中的未来序列；末尾 clamp 与训练时 z_clip 同量级，减轻未收敛时的爆炸。
        return_trajectory=True 时返回 (x_final, traj)，traj[0] 为初始噪声，其后为沿反向过程子采样快照（同 x 形状）。"""
        b = hist.shape[0]
        x = torch.randn(b, pred_len, channels, device=device)
        model.eval()
        mode = (sampling_mode or "ddpm").lower()
        trajectory: list[torch.Tensor] = []
        if return_trajectory:
            trajectory.append(x.detach().clone())
        if sample_debug:
            self._log_sample_tensor("init noise x_T", x)

        if mode == "ddim":
            t_total = self.timesteps
            pairs = build_ddim_time_pairs(t_total, sampling_steps)
            n_pairs = len(pairs)
            save_idx = (
                _trajectory_save_indices(n_pairs, trajectory_max_points)
                if return_trajectory
                else set()
            )
            for i, (t, t_prev) in enumerate(pairs):
                x = self._ddim_step(
                    model,
                    x,
                    t,
                    t_prev,
                    hist,
                    float(ddim_eta),
                    clamp_abs,
                    clip_pred_x0,
                    inference_fp16,
                )
                if sample_debug and (
                    i % max(1, sample_debug_every) == 0 or i == len(pairs) - 1
                ):
                    self._log_sample_tensor(f"ddim {i+1}/{len(pairs)} t={t}->{t_prev} x", x)
                if return_trajectory and i in save_idx:
                    trajectory.append(x.detach().clone())
            x = x.clamp(-clamp_abs, clamp_abs)
            if return_trajectory:
                if len(trajectory) == 0 or not torch.allclose(trajectory[-1], x, atol=1e-5, rtol=1e-4):
                    trajectory.append(x.detach().clone())
                return x, trajectory
            return x

        total = self.timesteps
        save_idx = (
            _trajectory_save_indices(total, trajectory_max_points)
            if return_trajectory
            else set()
        )
        for step_i, t in enumerate(reversed(range(total))):
            x = self.p_sample_step(
                model,
                x,
                t,
                hist,
                clamp_abs=clamp_abs,
                clip_pred_x0=clip_pred_x0,
                inference_fp16=inference_fp16,
            )
            x = x.clamp(-clamp_abs, clamp_abs)
            if sample_debug and (
                step_i % max(1, sample_debug_every) == 0 or t == 0
            ):
                self._log_sample_tensor(f"ddpm t={t} x", x)
            if return_trajectory and step_i in save_idx:
                trajectory.append(x.detach().clone())
        x = x.clamp(-clamp_abs, clamp_abs)
        if return_trajectory:
            if len(trajectory) == 0 or not torch.allclose(trajectory[-1], x, atol=1e-5, rtol=1e-4):
                trajectory.append(x.detach().clone())
            return x, trajectory
        return x
