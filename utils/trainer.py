"""
训练 / 验证循环、早停、学习率衰减与最优权重保存。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config.config import Config
from models.simdiff import SimDiffWeather


def _config_to_meta(cfg: Config) -> dict:
    """可 JSON 序列化的元数据，便于检查 checkpoint 是否由当前配置训练。"""
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
        "use_global_standardization": cfg.use_global_standardization,
        "independent_future_normalization": cfg.independent_future_normalization,
        "seq_len": cfg.seq_len,
        "pred_len": cfg.pred_len,
        "input_dim": cfg.input_dim,
    }
    if cfg.global_mean is not None:
        meta["global_mean"] = np.asarray(cfg.global_mean, dtype=np.float64).tolist()
    if cfg.global_std is not None:
        meta["global_std"] = np.asarray(cfg.global_std, dtype=np.float64).tolist()
    return meta


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: SimDiffWeather,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optim = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.99),
        )
        self.scheduler = ReduceLROnPlateau(
            self.optim,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        )
        self.best_val = float("inf")
        self.bad_epochs = 0

    def train_epoch(self) -> float:
        self.model.train()
        losses = []
        skipped = 0
        clip_norm = self.cfg.grad_clip_max_norm
        for hist, fut in tqdm(self.train_loader, desc="train", leave=False):
            hist = hist.to(self.device)
            fut = fut.to(self.device)
            self.optim.zero_grad(set_to_none=True)
            loss = self.model.training_loss(hist, fut)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            self.optim.step()
            losses.append(loss.item())
        if skipped > 0:
            print(f"  [warn] 跳过非有限 loss 的 batch 数: {skipped}")
        if not losses:
            raise RuntimeError("本 epoch 无有效 loss，请检查数据与归一化。")
        return float(np.mean(losses))

    @torch.no_grad()
    def validate(self) -> float:
        """验证集噪声预测 MSE。"""
        self.model.eval()
        losses = []
        for hist, fut in self.val_loader:
            hist = hist.to(self.device)
            fut = fut.to(self.device)
            loss = self.model.training_loss(hist, fut)
            if torch.isfinite(loss):
                losses.append(loss.item())
        if not losses:
            return float("nan")
        return float(np.mean(losses))

    def save(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "meta": _config_to_meta(self.cfg),
                "epoch_trained": epoch,
            },
            path,
        )

    def fit(self) -> None:
        ckpt_dir = self.cfg.resolved_checkpoint_dir()
        best_path = ckpt_dir / "simdiff_weather_best.pt"

        print("=== 训练模式：将更新模型权重；请勿使用 --eval_only 跳过训练 ===")

        for epoch in range(1, self.cfg.epochs + 1):
            tr = self.train_epoch()
            va = self.validate()
            self.scheduler.step(va)
            lr_now = self.optim.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d} | train_loss {tr:.6f} | val_mse {va:.6f} | lr {lr_now:.2e}"
            )
            if not np.isfinite(va):
                print("  [warn] val 非有限，请检查数据。")
                continue
            if va < self.best_val - 1e-8:
                self.best_val = va
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
