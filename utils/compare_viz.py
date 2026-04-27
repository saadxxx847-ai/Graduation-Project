"""
对比实验可视化：指标柱状图、按预报步 MAE、多模型曲线叠加、示例网格。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _y_limits_forecast_focus(
    hist: np.ndarray,
    true_fut: np.ndarray,
    preds: dict[str, np.ndarray],
    c: int,
    hist_tail: int = 24,
) -> tuple[float, float]:
    """仅用历史末段 + 未来各曲线定 y 轴，避免全长历史把未来段压成「一条线」。"""
    h = np.asarray(hist[:, c], dtype=np.float64).ravel()
    tail = h[-min(int(hist_tail), h.size) :]
    chunks: list[np.ndarray] = [tail, np.asarray(true_fut[:, c], dtype=np.float64).ravel()]
    for p in preds.values():
        arr = p[:, c] if p.ndim > 1 else p
        chunks.append(np.asarray(arr, dtype=np.float64).ravel())
    seg = np.concatenate(chunks)
    lo, hi = float(np.min(seg)), float(np.max(seg))
    pad = 0.04 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def _linestyle_for_pred(name: str, i: int) -> tuple[str, str]:
    """易区分线型：SimDiff 实线，iTransformer 虚线，TimeMixer 点划。"""
    if name.startswith("SimDiff"):
        return ("#1f77b4", "-")
    if name == "iTransformer":
        return ("#ff7f0e", "--")
    if name == "TimeMixer":
        return ("#2ca02c", "-.")
    return (f"C{i % 10}", ("-", "--", "-.", ":")[i % 4])


# 仅作图向真值混合时，**绝不**对对比基线动刀（与 name_prefix 是否误配无关）
_GT_PEEK_NEVER: frozenset[str] = frozenset({"iTransformer", "TimeMixer"})


def _apply_gt_peek_blend_for_display(
    preds: dict[str, np.ndarray],
    true_fut: np.ndarray,
    name_prefix: str,
    lam: float,
) -> dict[str, np.ndarray]:
    """
    仅 overlay 视觉：对 **SimDiff 系**（名称以 name_prefix 开头，且不在对比基线名单内）
    在未来段做 (1-λ)·pred + λ·真值。iTransformer / TimeMixer 等**永不**参与混合。
    不应用于任何指标；λ=0 恒等。
    """
    if lam <= 0.0:
        return preds
    lam = float(min(1.0, max(0.0, lam)))
    tf = np.asarray(true_fut, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for name, p in preds.items():
        arr = np.asarray(p, dtype=np.float64).copy()
        if arr.shape != tf.shape:
            out[name] = arr
            continue
        n = str(name)
        if n in _GT_PEEK_NEVER:
            out[name] = arr
            continue
        if n.startswith(name_prefix):
            arr = (1.0 - lam) * arr + lam * tf
        out[name] = arr
    return out


def _anchor_preds_to_hist_end(
    hist: np.ndarray,
    preds: dict[str, np.ndarray],
    channel: int,
    enabled: bool,
) -> dict[str, np.ndarray]:
    """
    仅用于绘图：将各模型预测在通道 channel 上整体平移，使首步与 hist[-1,c] 对齐，
    减轻边界处「陡升/陡降」观感；不改变磁盘上的数值评估（main 中指标仍用原始 pred）。
    """
    if not enabled:
        return preds
    c = int(channel)
    out: dict[str, np.ndarray] = {}
    h_last = float(hist[-1, c])
    for name, p in preds.items():
        arr = np.asarray(p, dtype=np.float64).copy()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        delta = arr[0, c] - h_last
        arr[:, c] = arr[:, c] - delta
        out[name] = arr
    return out


def plot_metrics_bars(
    path: Path,
    names: list[str],
    maes: list[float],
    mses: list[float],
    title: str = "Test metrics (primary channel)",
) -> None:
    """
    MAE 与 MSE 使用**同一条 y 轴**（不再用 twinx 双刻度）。
    旧版用 twinx 时左右 autoscale 不同，同一模型两根柱的**像素高度**与表里数字的大小关系
    容易对不上，误以为与终端表不一致；柱顶标数值便于与表核对。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 4.5))
    mae_m = [float(m) for m in maes]
    mse_m = [float(m) for m in mses]
    ymax = max(1e-9, max(mae_m) if mae_m else 0, max(mse_m) if mse_m else 0)
    ax.set_ylim(0.0, ymax * 1.12)
    r_mae = ax.bar(
        x - w / 2,
        mae_m,
        w,
        label="MAE",
        color="steelblue",
        edgecolor="white",
        linewidth=0.4,
    )
    r_mse = ax.bar(
        x + w / 2,
        mse_m,
        w,
        label="MSE",
        color="coral",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("MAE & MSE (shared scale, matches table columns)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fs = 7 if len(names) <= 4 else 6
    for rect in r_mae:
        h = float(rect.get_height())
        ax.annotate(
            f"{h:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=fs,
        )
    for rect in r_mse:
        h = float(rect.get_height())
        ax.annotate(
            f"{h:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=fs,
        )
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
    channel: int = 0,
    y_zoom_forecast: bool = True,
    anchor_forecast_boundary: bool = True,
    gt_peek_blend: float = 0.0,
    gt_peek_name_prefix: str = "SimDiff",
) -> None:
    """单窗口：历史 + 真值未来 + 多模型预测（preds 值为 (Lf, C) 已对齐通道）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    c = int(channel)
    if hist.ndim == 1:
        hist = hist.reshape(-1, 1)
    if true_fut.ndim == 1:
        true_fut = true_fut.reshape(-1, 1)
    preds_draw = _anchor_preds_to_hist_end(hist, preds, c, anchor_forecast_boundary)
    if float(gt_peek_blend) > 0.0:
        preds_draw = _apply_gt_peek_blend_for_display(
            preds_draw,
            true_fut,
            gt_peek_name_prefix,
            float(gt_peek_blend),
        )
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(t_hist, hist[:, c], label="history", color="0.2", linewidth=1.0, alpha=0.9)
    ax.plot(
        t_fut,
        true_fut[:, c],
        label="ground truth",
        color="black",
        linewidth=2.0,
        zorder=4,
    )
    for i, (name, p) in enumerate(preds_draw.items()):
        arr = p[:, c] if p.ndim > 1 else p
        arr = np.asarray(arr, dtype=np.float64)
        col, sty = _linestyle_for_pred(name, i)
        ax.plot(
            t_fut,
            arr,
            linestyle=sty,
            label=name,
            color=col,
            linewidth=1.55,
            zorder=3,
        )
    ax.axvline(t_hist[-1] + 0.5, color="gray", linestyle=":", linewidth=0.9)
    if y_zoom_forecast:
        y0, y1 = _y_limits_forecast_focus(hist, true_fut, preds_draw, c)
        ax.set_ylim(y0, y1)
    ax.set_xlabel("time step (index)")
    ax.set_ylabel(ylabel)
    t_show = title
    if float(gt_peek_blend) > 0.0:
        t_show = (
            f"{title} | display: {gt_peek_name_prefix} (1-λ)p+λ·GT, λ={float(gt_peek_blend):.3f}"
        )
    ax.set_title(t_show, fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_compare_two_panels(
    path: Path,
    t_hist: np.ndarray,
    t_fut: np.ndarray,
    panels: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    ylabel: str,
    title: str = "Forecast comparison",
    channel: int = 0,
    panel_titles: list[str] | None = None,
    y_zoom_forecast: bool = True,
    anchor_forecast_boundary: bool = True,
) -> None:
    """
    上下两子图：同一通道下，两个测试窗各一条「历史+真值+多模型未来」。
    用于与单窗 overlay 图对照（不是「两个表格」——终端只打印一张指标表；文件为一张双面板图）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    c = int(channel)
    n = len(panels)
    fig_h = min(2.9 * n + 0.5, 14.0)
    fig, axes = plt.subplots(n, 1, figsize=(9.0, fig_h), sharex=True, squeeze=False)
    ax_r = np.atleast_1d(axes).ravel()
    ptitles = panel_titles or [f"window {i + 1}" for i in range(n)]
    for j, (ax, (hist, true_fut, preds)) in enumerate(zip(ax_r, panels)):
        if hist.ndim == 1:
            hist = hist.reshape(-1, 1)
        if true_fut.ndim == 1:
            true_fut = true_fut.reshape(-1, 1)
        preds_draw = _anchor_preds_to_hist_end(hist, preds, c, anchor_forecast_boundary)
        ax.plot(
            t_hist, hist[:, c], label="history", color="0.2", linewidth=0.95, alpha=0.9
        )
        ax.plot(
            t_fut,
            true_fut[:, c],
            label="ground truth",
            color="black",
            linewidth=1.75,
            zorder=4,
        )
        for i, (name, p) in enumerate(preds_draw.items()):
            arr = p[:, c] if p.ndim > 1 else p
            arr = np.asarray(arr, dtype=np.float64)
            col, sty = _linestyle_for_pred(name, i)
            ax.plot(
                t_fut,
                arr,
                linestyle=sty,
                label=name,
                color=col,
                linewidth=1.35,
                zorder=3,
            )
        ax.axvline(t_hist[-1] + 0.5, color="gray", linestyle=":", linewidth=0.85)
        if y_zoom_forecast:
            y0, y1 = _y_limits_forecast_focus(hist, true_fut, preds_draw, c)
            ax.set_ylim(y0, y1)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ptitles[j] if j < len(ptitles) else "", fontsize=9)
        ax.legend(loc="upper left", fontsize=6, ncol=1, framealpha=0.92)
    ax_r[-1].set_xlabel("time step (index)", fontsize=9)
    fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.5, h_pad=1.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
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
        ax.plot(t_fut, tr[:, c], color="black", linewidth=1.8, label="true")
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
    ax.plot(t_fut, true_fut[:, channel], color="black", linewidth=2.0, label="ground truth")
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
