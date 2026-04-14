"""
读取 weather.csv，滑动窗口构造 (历史, 未来) 样本，按时间顺序划分 train/val/test。
仅在训练集「未来窗口」上估计边际 μ/σ，供无真值推理时 inverse_transform_future（不涉及测试标签）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from config.config import Config


class WeatherWindowDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        """
        data: (T, C) 全序列
        """
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n = len(data) - seq_len - pred_len + 1
        if self.n <= 0:
            raise ValueError(
                f"序列太短：len={len(data)},需要至少 seq_len+pred_len={seq_len + pred_len}"
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = idx
        hist = self.data[i : i + self.seq_len]
        fut = self.data[i + self.seq_len : i + self.seq_len + self.pred_len]
        return torch.from_numpy(hist), torch.from_numpy(fut)


def load_weather_matrix(csv_path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    feature_names = list(df.columns)
    values = df.values.astype(np.float32)
    if np.isnan(values).any():
        col_mean = np.nanmean(values, axis=0)
        inds = np.where(np.isnan(values))
        values[inds] = np.take(col_mean, inds[1])
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values, feature_names


def fit_future_marginal_stats(
    train_ds: WeatherWindowDataset,
    abs_floor: float = 1e-3,
    rel_floor: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    仅用训练集中每个样本的 **未来窗口** (Lf, C) 聚合，得到每通道边际 mean/std。
    不拼接历史段，不把历史与未来放在同一向量里算统计量。
    """
    if len(train_ds) == 0:
        raise ValueError("训练集为空，无法估计未来边际统计量")
    futs = []
    for i in range(len(train_ds)):
        _, f = train_ds[i]
        futs.append(f.numpy())
    F = np.stack(futs, axis=0).astype(np.float32)  # (N, Lf, C)
    mu = F.mean(axis=(0, 1))
    raw_std = F.std(axis=(0, 1))
    rel = rel_floor * np.maximum(np.abs(mu), 1.0).astype(np.float32)
    sig = np.maximum(np.maximum(raw_std, abs_floor), rel).astype(np.float32)
    return mu.astype(np.float32), sig.astype(np.float32)


def make_loaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader, int, list[str]]:
    path = cfg.resolved_data_path()
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件: {path}")

    matrix, names = load_weather_matrix(path)
    T, C = matrix.shape
    cfg.input_dim = C

    train_end = int(T * cfg.train_ratio)
    val_end = int(T * (cfg.train_ratio + cfg.val_ratio))

    train_mat = matrix[:train_end]
    val_mat = matrix[train_end:val_end]
    test_mat = matrix[val_end:]

    train_ds = WeatherWindowDataset(train_mat, cfg.seq_len, cfg.pred_len)
    val_ds = WeatherWindowDataset(val_mat, cfg.seq_len, cfg.pred_len)
    test_ds = WeatherWindowDataset(test_mat, cfg.seq_len, cfg.pred_len)

    fut_mu, fut_sig = fit_future_marginal_stats(train_ds)
    cfg.train_future_marginal_mean = fut_mu
    cfg.train_future_marginal_std = fut_sig

    if cfg.use_global_standardization:
        # 保留字段供旧 checkpoint / 外部工具；SimDiff 核心路径已改用窗口独立归一化
        g_mean = train_mat.mean(axis=0).astype(np.float32)
        g_std = train_mat.std(axis=0).astype(np.float32)
        rel = 1e-4 * np.maximum(np.abs(g_mean), 1.0).astype(np.float32)
        g_std = np.maximum(np.maximum(g_std, 1e-3), rel).astype(np.float32)
        cfg.global_mean = g_mean
        cfg.global_std = g_std
    else:
        cfg.global_mean = None
        cfg.global_std = None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader, C, names
