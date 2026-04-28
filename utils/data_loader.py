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

# 多尺度：168=7×24（日窗），672=4×168（周窗）；需在窗口起点 i 之前保留足够上下文 → i>=576
_MULTISCALE_PREFIX = 672 - 96


def _concat_multiscale_history(
    data: np.ndarray,
    window_start: int,
    seq_len: int,
) -> np.ndarray:
    """拼接 [seq_len 原始步 | 7×日平均 | 4×周平均]，shape (seq_len+11, C)。window_start 为当前样本历史起点 i。"""
    anchor = window_start + seq_len
    fine = data[window_start : window_start + seq_len]
    blk168 = data[anchor - 168 : anchor]
    daily = blk168.reshape(7, 24, -1).mean(axis=1)
    blk672 = data[anchor - 672 : anchor]
    weekly = blk672.reshape(4, 168, -1).mean(axis=1)
    return np.concatenate([fine, daily, weekly], axis=0).astype(np.float32)


def resolve_temperature_column_name(feature_names: list[str]) -> str:
    """在 weather表头中定位气温列名（与 main 中逻辑一致）。"""
    for key in ("T (degC)", "T(degC)", "temp", "temperature"):
        for name in feature_names:
            if name.strip().lower() == key.lower():
                return name
    for name in feature_names:
        n = name.lower()
        if "degc" in n and "tlog" not in n and "tpot" not in n and n.strip().startswith("t"):
            return name
    raise ValueError(
        f"未找到气温列（期望如 T (degC)）。当前列: {feature_names[:15]}..."
    )


class WeatherWindowDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int,
        *,
        multiscale: bool = False,
        window_start_min: int = 0,
    ):
        """
        data: (T, C) 全序列
        window_start_min: 允许的最早窗口起点索引 i（相对本段 data 下标 0）。
        multiscale: True 时历史为 seq_len+11 步拼接向量（仍为一通道标量每步）。
        """
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.multiscale = bool(multiscale)
        self.window_start_min = int(window_start_min)
        self.n = len(data) - seq_len - pred_len + 1 - self.window_start_min
        if self.n <= 0:
            raise ValueError(
                f"序列太短：len={len(data)},需要至少 seq_len+pred_len+window_start_min="
                f"{seq_len + pred_len + self.window_start_min}"
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = idx + self.window_start_min
        fut = self.data[i + self.seq_len : i + self.seq_len + self.pred_len]
        if self.multiscale:
            hist = _concat_multiscale_history(self.data, i, self.seq_len)
        else:
            hist = self.data[i : i + self.seq_len]
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
    if getattr(cfg, "temperature_only", True):
        tcol = resolve_temperature_column_name(names)
        ti = names.index(tcol)
        matrix = matrix[:, ti : ti + 1]
        names = [names[ti]]
    T, C = matrix.shape
    cfg.input_dim = C

    ms = bool(getattr(cfg, "use_multiscale_hist", False))
    wmin = int(getattr(cfg, "hist_window_start_min", 0))
    if ms:
        wmin = max(wmin, _MULTISCALE_PREFIX)
        cfg.hist_window_start_min = wmin

    train_end = int(T * cfg.train_ratio)
    val_end = int(T * (cfg.train_ratio + cfg.val_ratio))

    train_mat = matrix[:train_end]
    val_mat = matrix[train_end:val_end]
    test_mat = matrix[val_end:]

    train_ds = WeatherWindowDataset(
        train_mat, cfg.seq_len, cfg.pred_len, multiscale=ms, window_start_min=wmin
    )
    val_ds = WeatherWindowDataset(
        val_mat, cfg.seq_len, cfg.pred_len, multiscale=ms, window_start_min=wmin
    )
    test_ds = WeatherWindowDataset(
        test_mat, cfg.seq_len, cfg.pred_len, multiscale=ms, window_start_min=wmin
    )

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

    pin = bool(torch.cuda.is_available())
    nw = int(cfg.num_workers)
    dl_common: dict = {
        "batch_size": cfg.batch_size,
        "num_workers": nw,
        "pin_memory": pin,
    }
    if nw > 0:
        dl_common["prefetch_factor"] = 2
        dl_common["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **dl_common,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **dl_common,
    )
    tbs = cfg.test_batch_size if cfg.test_batch_size is not None else cfg.batch_size
    dl_test = {**dl_common, "batch_size": int(tbs)}
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        drop_last=False,
        **dl_test,
    )
    return train_loader, val_loader, test_loader, C, names
