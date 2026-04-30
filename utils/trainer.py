"""
训练 / 验证循环、早停与最优权重保存；**固定学习率**（无调度器）。
支持训练期 CUDA 混合精度（AMP）与 SimDiff 权重的 EMA（验证与保存/推理用）。
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config.config import Config
from models.simdiff import SimDiffWeather, point_prediction_from_forecast
from utils.baselines import forecast_amp_context


def _config_to_meta(cfg: Config) -> dict:
    meta = {
        "learning_rate": cfg.learning_rate,
        "z_clip": cfg.z_clip,
        "grad_clip_max_norm": cfg.grad_clip_max_norm,
        "timesteps": cfg.timesteps,
        "sampling_mode": cfg.sampling_mode,
        "sampling_steps": cfg.sampling_steps,
        "ddim_eta": cfg.ddim_eta,
        "sample_clip_pred_x0": cfg.sample_clip_pred_x0,
        "training_noise_l1_weight": cfg.training_noise_l1_weight,
        "training_noise_temporal_diff_weight": cfg.training_noise_temporal_diff_weight,
        "training_noise_mse_huber_alpha": cfg.training_noise_mse_huber_alpha,
        "training_noise_huber_beta": cfg.training_noise_huber_beta,
        "training_noise_x0_aux_weight": float(
            getattr(cfg, "training_noise_x0_aux_weight", 0.0)
        ),
        "ni_inverse_hist_frac": float(getattr(cfg, "ni_inverse_hist_frac", 0.0)),
        "forecast_num_samples": cfg.forecast_num_samples,
        "mom_num_groups": cfg.mom_num_groups,
        "mom_cold_bias_blend": cfg.mom_cold_bias_blend,
        "mom_cold_sharpness": cfg.mom_cold_sharpness,
        "simdiff_ablation": cfg.simdiff_ablation,
        "temperature_only": cfg.temperature_only,
        "seq_len": cfg.seq_len,
        "denoiser_hist_len": cfg.effective_hist_len(),
        "pred_len": cfg.pred_len,
        "input_dim": cfg.input_dim,
        "use_multiscale_hist": getattr(cfg, "use_multiscale_hist", False),
        "multiscale_steps_per_hour": getattr(cfg, "multiscale_steps_per_hour", None),
        "hist_window_start_min": getattr(cfg, "hist_window_start_min", 0),
        "use_revin": cfg.use_revin,
        "use_hist_add_bias": getattr(cfg, "use_hist_add_bias", False),
        "hist_add_bias_scale": float(getattr(cfg, "hist_add_bias_scale", 0.12)),
        "hist_add_bias_scale_with_rmsnorm": float(
            getattr(cfg, "hist_add_bias_scale_with_rmsnorm", 0.08)
        ),
        "use_rmsnorm": cfg.use_rmsnorm,
        "ablation_ckpt_suite": getattr(cfg, "ablation_ckpt_suite", None),
        "denoiser_variant": cfg.denoiser_variant,
        "simdiff_checkpoint_extra_suffix": getattr(cfg, "simdiff_checkpoint_extra_suffix", None),
        "result_name_suffix": cfg.result_name_suffix,
        "forecast_amp": cfg.forecast_amp,
        "train_amp": cfg.train_amp,
        "use_ema": cfg.use_ema,
        "ema_decay": cfg.ema_decay,
        "val_forecast_mae_every": int(getattr(cfg, "val_forecast_mae_every", 0)),
        "val_forecast_mae_max_batches": int(getattr(cfg, "val_forecast_mae_max_batches", 4)),
        "val_forecast_mae_num_samples": getattr(cfg, "val_forecast_mae_num_samples", None),
        "checkpoint_select_metric": getattr(cfg, "checkpoint_select_metric", "val_noise"),
        "forecast_point_mode": getattr(cfg, "forecast_point_mode", "mom"),
        "forecast_loss_primary_weight": float(
            getattr(cfg, "forecast_loss_primary_weight", 1.0)
        ),
        "forecast_loss_primary_channel_idx": getattr(
            cfg, "forecast_loss_primary_channel_idx", None
        ),
    }
    if cfg.train_future_marginal_mean is not None:
        meta["train_future_marginal_mean"] = np.asarray(
            cfg.train_future_marginal_mean, dtype=np.float64
        ).tolist()
    if getattr(cfg, "train_metric_z_mu", None) is not None:
        meta["train_metric_z_mu"] = np.asarray(cfg.train_metric_z_mu, dtype=np.float64).tolist()
    if getattr(cfg, "train_metric_z_sigma", None) is not None:
        meta["train_metric_z_sigma"] = np.asarray(cfg.train_metric_z_sigma, dtype=np.float64).tolist()
    return meta


class _ModelEMA:
    """对 state_dict 做指数滑动平均，用于推理/验证（不额外增加推理步数，几乎不占训练时间）。"""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        with torch.no_grad():
            self.shadow: dict[str, torch.Tensor] = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if k in msd:
                msd[k].copy_(v.to(device=msd[k].device, dtype=msd[k].dtype))

    @torch.no_grad()
    def state_dict(self) -> dict[str, Any]:
        return {k: v.clone() for k, v in self.shadow.items()}


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: SimDiffWeather,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        *,
        primary_forecast_channel: int = 0,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.primary_forecast_channel = int(primary_forecast_channel)
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.99),
        )
        self.best_val = float("inf")
        self.best_forecast_mae = float("inf")
        self.bad_epochs = 0
        self.history_train: list[float] = []
        self.history_val: list[float] = []

        self._train_amp = bool(cfg.train_amp) and device.type == "cuda"
        if self._train_amp:
            self._scaler = torch.amp.GradScaler("cuda")
        else:
            self._scaler = None
        if bool(cfg.use_ema) and self.cfg.ema_decay > 0.0 and self.cfg.ema_decay < 1.0:
            self.ema: _ModelEMA | None = _ModelEMA(
                self.model, decay=float(self.cfg.ema_decay)
            )
        else:
            self.ema = None

    @contextmanager
    def _param_snapshot(self):
        """无 EMA 时恒等；有 EMA 时验证用 shadow 权，结束再写回正在训练的权重。"""
        if self.ema is None:
            yield
            return
        backup: dict[str, torch.Tensor] = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }
        try:
            self.ema.copy_to(self.model)
            yield
        finally:
            self.model.load_state_dict(backup, strict=True)

    def train_epoch(self) -> float:
        self.model.train()
        losses = []
        skipped = 0
        clip_norm = self.cfg.grad_clip_max_norm
        for hist, fut in tqdm(self.train_loader, desc="train", leave=False):
            hist = hist.to(self.device)
            fut = fut.to(self.device)
            self.optim.zero_grad(set_to_none=True)

            if self._scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = self.model.training_loss(hist, fut)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                li = float(loss.detach().item())
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=clip_norm
                )
                self._scaler.step(self.optim)
                self._scaler.update()
            else:
                loss = self.model.training_loss(hist, fut)
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                li = float(loss.detach().item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=clip_norm
                )
                self.optim.step()

            if self.ema is not None:
                self.ema.update(self.model)
            losses.append(li)
        if skipped > 0:
            print(f"  [warn] 跳过非有限 loss 的 batch 数: {skipped}")
        if not losses:
            raise RuntimeError("本 epoch 无有效 loss，请检查数据与归一化。")
        return float(np.mean(losses))

    @torch.no_grad()
    def validate(self) -> float:
        losses: list[float] = []
        with self._param_snapshot():
            self.model.eval()
            for hist, fut in self.val_loader:
                hist = hist.to(self.device)
                fut = fut.to(self.device)
                loss = self.model.training_loss(hist, fut)
                if torch.isfinite(loss):
                    losses.append(float(loss.item()))
        if not losses:
            return float("nan")
        return float(np.mean(losses))

    @torch.no_grad()
    def validation_forecast_mae_sparse(self) -> float:
        """
        验证集上前若干个 batch：主变量通道平均 MAE（与评估一致的 forecast + MoM/ni_only）。
        仅在稀疏调度下调用；可通过 cfg.val_forecast_mae_num_samples 减小 K 以加速。
        """
        ch = self.primary_forecast_channel
        max_b = int(getattr(self.cfg, "val_forecast_mae_max_batches", 4))
        k_sparse = getattr(self.cfg, "val_forecast_mae_num_samples", None)
        sums = 0.0
        count = 0
        with self._param_snapshot():
            self.model.eval()
            with forecast_amp_context(self.device, bool(self.cfg.forecast_amp)):
                for bi, (hist, fut) in enumerate(self.val_loader):
                    if bi >= max_b:
                        break
                    hist = hist.to(self.device)
                    fut = fut.to(self.device)
                    ks = int(k_sparse) if k_sparse is not None else None
                    out = self.model.forecast(
                        hist,
                        future=fut,
                        num_samples=ks,
                    )
                    pred = point_prediction_from_forecast(out, self.cfg)
                    diff = torch.abs(pred[..., ch] - fut[..., ch])
                    sums += float(diff.sum().item())
                    count += int(diff.numel())
        if count <= 0:
            return float("nan")
        return sums / count

    def save(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.ema is not None:
            to_save = self.ema.state_dict()
            raw_model = self.model.state_dict()
        else:
            to_save = self.model.state_dict()
            raw_model = None
        torch.save(
            {
                "model": to_save,
                "raw_model": raw_model,
                "meta": _config_to_meta(self.cfg),
                "epoch_trained": epoch,
            },
            path,
        )

    def fit(self) -> None:
        ckpt_dir = self.cfg.resolved_checkpoint_dir()
        best_path = ckpt_dir / self.cfg.simdiff_checkpoint_filename()

        print("=== 训练模式：将更新模型权重；请勿使用 --eval_only 跳过训练 ===")
        print(f"=== 权重将保存为: {best_path.name} (ablation={self.cfg.simdiff_ablation}) ===")
        print(
            f"  train_amp={'on' if self._scaler is not None else 'off'} | "
            f"ema={'on' if self.ema is not None else 'off'} | lr 固定 {self.cfg.learning_rate:.2e}（无调度器）"
        )
        metric = str(getattr(self.cfg, "checkpoint_select_metric", "val_noise"))
        every_fc = int(getattr(self.cfg, "val_forecast_mae_every", 0))
        max_b_fc = int(getattr(self.cfg, "val_forecast_mae_max_batches", 4))
        k_fast = getattr(self.cfg, "val_forecast_mae_num_samples", None)
        _k_note = ""
        if every_fc > 0 and k_fast is not None:
            _k_note = f"，验证采样 K={int(k_fast)}（快于完整测试 K={cfg.forecast_num_samples}）"
        print(
            f"  checkpoint 依据: {metric}"
            + (
                f" | 稀疏预报 MAE: 每 {every_fc} epoch，≤{max_b_fc} val batches（主变量 ch={self.primary_forecast_channel}{_k_note}）"
                if every_fc > 0
                else ""
            )
        )

        for epoch in range(1, self.cfg.epochs + 1):
            tr = self.train_epoch()
            va = self.validate()
            lr_now = self.optim.param_groups[0]["lr"]

            run_fc = every_fc > 0 and (epoch % every_fc == 0)
            fmae: float | None = None
            if run_fc:
                fmae = float(self.validation_forecast_mae_sparse())

            msg = (
                f"Epoch {epoch:03d} | train_loss {tr:.6f} | val_noise {va:.6f} | lr {lr_now:.2e}"
            )
            _kf_sparse = getattr(self.cfg, "val_forecast_mae_num_samples", None)
            if fmae is not None and np.isfinite(fmae):
                _kpart = f", K={int(_kf_sparse)}" if _kf_sparse is not None else ""
                msg += (
                    f" | val_fc_MAE[ch{self.primary_forecast_channel}] {fmae:.6f} "
                    f"(sparse ≤{max_b_fc} batches{_kpart})"
                )
            print(msg)

            self.history_train.append(tr)
            self.history_val.append(float(va) if np.isfinite(va) else float("nan"))
            if not np.isfinite(va):
                print("  [warn] val 非有限，请检查数据。")
                continue

            improved = False
            if metric == "val_noise":
                if va < self.best_val - 1e-8:
                    self.best_val = va
                    improved = True
            else:
                # val_forecast_mae_sparse：有稀疏 MAE 的 epoch 优先按预报改进存盘；否则按噪声 loss（前几 epoch 热身）
                if fmae is not None and np.isfinite(fmae):
                    if fmae < self.best_forecast_mae - 1e-8:
                        self.best_forecast_mae = fmae
                        improved = True
                elif va < self.best_val - 1e-8:
                    self.best_val = va
                    improved = True

            if improved:
                self.bad_epochs = 0
                self.save(best_path, epoch)
                print(f"  -> saved best to {best_path}")
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.cfg.early_stop_patience:
                    print("Early stopping.")
                    break

        if best_path.exists():
            state = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"], strict=True)
            print("已加载验证集最优权重。")
