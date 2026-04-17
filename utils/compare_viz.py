"""
对比实验可视化：指标柱状图、按预报步 MAE、多模型曲线叠加、示例网格。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_metrics_bars(
    path: Path,
    names: list[str],
    maes: list[float],
    mses: list[float],
    title: str = "Test metrics (primary channel)",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 4.5))
    ax.bar(x - w / 2, maes, w, label="MAE", color="steelblue")
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, mses, w, label="MSE", color="coral", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("MAE")
    ax2.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_horizon_mae(
    path: Path,
    horizon_maes: dict[str, np.ndarray],
    pred_len: int,
    title: str = "MAE by forecast horizon",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    steps = np.arange(1, pred_len + 1)
    for name, arr in horizon_maes.items():
        ax.plot(steps, arr, marker="o", markersize=3, label=name, linewidth=1.5)
    ax.set_xlabel("Horizon (step)")
    ax.set_ylabel("MAE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forecast_compare(
    path: Path,
    t_hist: np.ndarray,
    t_fut: np.ndarray,
    hist: np.ndarray,
    true_fut: np.ndarray,
    preds: dict[str, np.ndarray],
    ylabel: str,
    title: str = "Forecast comparison",
) -> None:
    """单窗口：历史 + 真值未来 + 多模型预测（preds 值为 (Lf, C) 已对齐通道）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    c = 0
    if hist.ndim == 1:
        hist = hist.reshape(-1, 1)
    if true_fut.ndim == 1:
        true_fut = true_fut.reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_hist, hist[:, c], label="history", color="black", linewidth=1.2)
    ax.plot(t_fut, true_fut[:, c], label="ground truth", color="C1", linewidth=2)
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(preds), 1)))
    for i, (name, p) in enumerate(preds.items()):
        arr = p[:, c] if p.ndim > 1 else p
        ax.plot(t_fut, arr, linestyle="--", label=name, color=colors[i % 10], linewidth=1.5)
    ax.axvline(t_hist[-1] + 0.5, color="gray", linestyle=":")
    ax.set_xlabel("time step (index)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forecast_grid(
    path: Path,
    examples: list[dict],
    seq_len: int,
    pred_len: int,
    ylabel: str,
    title: str = "Forecast comparison (test samples)",
) -> None:
    """
    examples: 每项含 hist (Lh,C), true (Lf,C), preds dict name -> (Lf,C)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(examples)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows), squeeze=False)
    t_hist = np.arange(seq_len)
    t_fut = np.arange(seq_len, seq_len + pred_len)
    c = 0
    for i, ex in enumerate(examples):
        r, col = divmod(i, ncols)
        ax = axes[r][col]
        h = ex["hist"]
        tr = ex["true"]
        if h.ndim == 1:
            h = h.reshape(-1, 1)
        if tr.ndim == 1:
            tr = tr.reshape(-1, 1)
        ax.plot(t_hist, h[:, c], color="black", linewidth=1.0, label="hist")
        ax.plot(t_fut, tr[:, c], color="C1", linewidth=1.8, label="true")
        for j, (name, p) in enumerate(ex["preds"].items()):
            arr = p[:, c] if p.ndim > 1 else p
            ax.plot(
                t_fut,
                arr,
                linestyle="--",
                linewidth=1.2,
                label=name,
                color=plt.cm.tab10(j % 10),
            )
        ax.axvline(seq_len - 0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"sample {i + 1}")
        ax.legend(fontsize=7, loc="best")
    for j in range(n, nrows * ncols):
        r, col = divmod(j, ncols)
        axes[r][col].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pred_vs_true_scatter(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs true (pooled test)",
    max_points: int = 4000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = y_true.shape[0]
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, c="steelblue")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
