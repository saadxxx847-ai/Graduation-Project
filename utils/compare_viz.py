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


def plot_forecast_predictive_intervals(
    path: Path,
    t_hist: np.ndarray,
    t_fut: np.ndarray,
    hist: np.ndarray,
    true_fut: np.ndarray,
    samples_k_lc: np.ndarray,
    channel: int,
    ylabel: str,
    title: str = "Predictive intervals (K samples)",
    q_low: float = 0.05,
    q_high: float = 0.95,
    point_pred: np.ndarray | None = None,
    point_label: str = "MoM / point",
) -> None:
    """
    samples_k_lc: (K, Lf, C) 原始尺度；画分位阴影 + 真值 + 可选点预测。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if hist.ndim == 1:
        hist = hist.reshape(-1, 1)
    if true_fut.ndim == 1:
        true_fut = true_fut.reshape(-1, 1)
    s = samples_k_lc[..., channel]
    lo = np.quantile(s, q_low, axis=0)
    hi = np.quantile(s, q_high, axis=0)
    med = np.median(s, axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_hist, hist[:, channel], color="black", linewidth=1.1, label="history")
    ax.plot(t_fut, true_fut[:, channel], color="C1", linewidth=2.0, label="ground truth")
    ax.fill_between(
        t_fut,
        lo,
        hi,
        alpha=0.35,
        color="steelblue",
        label=f"{int((q_high - q_low) * 100)}% pred. interval",
    )
    ax.plot(
        t_fut,
        med,
        color="navy",
        linestyle=":",
        linewidth=1.5,
        label="sample median",
    )
    if point_pred is not None:
        pp = point_pred[:, channel] if point_pred.ndim > 1 else point_pred
        ax.plot(
            t_fut,
            pp,
            color="darkorange",
            linestyle="--",
            linewidth=1.5,
            label=point_label,
        )
    ax.axvline(t_hist[-1] + 0.5, color="gray", linestyle=":")
    ax.set_xlabel("time step (index)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_denoise_trajectory_heatmap(
    path: Path,
    frames: list[np.ndarray],
    channel: int,
    ylabel: str,
    title: str = "Diffusion denoising trajectory",
) -> None:
    """
    frames: 每帧 (Lf, C) 物理尺度；纵轴为去噪时间（上近噪声、下近收敛）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    M = np.stack([f[..., channel] for f in frames], axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xlabel("Forecast horizon (step)")
    ax.set_ylabel("Denoising snapshot (top: noisy -> bottom: clean)")
    n = M.shape[0]
    if n <= 10:
        ax.set_yticks(np.arange(n))
    else:
        ax.set_yticks([0, n // 2, n - 1])
        ax.set_yticklabels(["start", "mid", "end"])
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    path: Path,
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training convergence (noise prediction objective)",
) -> None:
    """训练/验证损失随 epoch（与终端一致：均为 diffusion training_loss标量平均）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not train_losses:
        return
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(epochs, train_losses, label="train", color="C0", linewidth=1.8)
    ax.plot(epochs, val_losses, label="validation", color="C1", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_kde(
    path: Path,
    residual: np.ndarray,
    abs_err: np.ndarray,
    unit_label: str,
    title: str = "Test forecast error distribution",
    residual_multi: dict[str, np.ndarray] | None = None,
) -> None:
    """
    左：残差 KDE。若提供 residual_multi（多模型），则叠多条曲线；否则仅画 residual（通常为 SimDiff）。
    右：绝对误差 |error| 的 KDE（始终对应与 residual / abs_err 同一套点预测，即 SimDiff）。
    """
    from scipy.stats import gaussian_kde

    path.parent.mkdir(parents=True, exist_ok=True)
    res = np.asarray(residual, dtype=np.float64).ravel()
    ae = np.asarray(abs_err, dtype=np.float64).ravel()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    def _kde_axis_single_fill(
        ax, data: np.ndarray, xlab: str, subt: str, vline: float | None
    ) -> None:
        data = data[np.isfinite(data)]
        if data.size < 3:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(subt)
            ax.set_xlabel(xlab)
            return
        lo, hi = np.percentile(data, [0.5, 99.5])
        span = max(hi - lo, 1e-9)
        xs = np.linspace(lo - 0.2 * span, hi + 0.2 * span, 400)
        kde = gaussian_kde(data)
        dens = kde(xs)
        ax.fill_between(xs, dens, alpha=0.35, color="steelblue")
        ax.plot(xs, dens, color="navy", linewidth=1.5)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Density (KDE)")
        ax.set_title(subt)
        if vline is not None:
            ax.axvline(vline, color="gray", linestyle="--", linewidth=1)

    xlab_res = f"residual (pred - true) [{unit_label}]"
    if residual_multi:
        cleaned: dict[str, np.ndarray] = {}
        for name, arr in residual_multi.items():
            x = np.asarray(arr, dtype=np.float64).ravel()
            x = x[np.isfinite(x)]
            if x.size >= 3:
                cleaned[name] = x
        if cleaned:
            lo = min(np.percentile(v, 0.5) for v in cleaned.values())
            hi = max(np.percentile(v, 99.5) for v in cleaned.values())
            span = max(hi - lo, 1e-9)
            xs = np.linspace(lo - 0.15 * span, hi + 0.15 * span, 400)
            colors = plt.cm.tab10(np.linspace(0, 0.88, len(cleaned)))
            for i, (name, x) in enumerate(cleaned.items()):
                kde = gaussian_kde(x)
                ax1.plot(xs, kde(xs), label=name, color=colors[i % 10], linewidth=2)
            ax1.axvline(0.0, color="gray", linestyle="--", linewidth=1)
            ax1.set_xlabel(xlab_res)
            ax1.set_ylabel("Density (KDE)")
            ax1.set_title("Residual (multi-model)")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
        else:
            _kde_axis_single_fill(ax1, res, xlab_res, "Residual (SimDiff)", 0.0)
    else:
        _kde_axis_single_fill(ax1, res, xlab_res, "Residual (SimDiff)", 0.0)

    _kde_axis_single_fill(
        ax2,
        ae,
        f"absolute error [{unit_label}]",
        "|error| (SimDiff)",
        None,
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residual_kde_multi(
    path: Path,
    residuals: dict[str, np.ndarray],
    unit_label: str,
    title: str = "Residual KDE comparison (test set)",
) -> None:
    """多模型残差 (pred−true) KDE 叠在同一张图。"""
    from scipy.stats import gaussian_kde

    path.parent.mkdir(parents=True, exist_ok=True)
    if not residuals:
        return
    cleaned: dict[str, np.ndarray] = {}
    for name, arr in residuals.items():
        x = np.asarray(arr, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size >= 3:
            cleaned[name] = x
    if not cleaned:
        return
    lo = min(np.percentile(v, 0.5) for v in cleaned.values())
    hi = max(np.percentile(v, 99.5) for v in cleaned.values())
    span = max(hi - lo, 1e-9)
    xs = np.linspace(lo - 0.15 * span, hi + 0.15 * span, 400)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = plt.cm.tab10(np.linspace(0, 0.88, len(cleaned)))
    for i, (name, x) in enumerate(cleaned.items()):
        kde = gaussian_kde(x)
        ax.plot(xs, kde(xs), label=name, color=colors[i % 10], linewidth=2)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"residual (pred - true) [{unit_label}]")
    ax.set_ylabel("Density (KDE)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_crps_by_horizon(
    path: Path,
    crps_per_step: np.ndarray,
    title: str = "CRPS by forecast horizon",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, len(crps_per_step) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, crps_per_step, marker="o", markersize=4, color="darkgreen")
    ax.set_xlabel("Horizon (step)")
    ax.set_ylabel("CRPS")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
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
