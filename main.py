#!/usr/bin/env python3
"""
SimDiff-Weather 主入口：默认 **先训练再评估**。
默认仅使用气温 `T (degC)` 或 ETT 的 `OT` 单变量；`--all_features` 可恢复多变量。
默认去噪器：**RMSNorm 开、RevIn 关、多尺度历史融合开**（`--revin` / `--single_scale_hist` 可改）。
使用 `--eval_only` 将 **跳过训练** 并加载已有权重——若从未训练，指标无意义。
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import Config
from models.simdiff import SimDiffWeather, point_prediction_from_forecast
from utils.baselines import (
    BaselineHistTrim,
    BaselineTimeMixer,
    BaselineiTransformer,
    collect_channel_residuals,
    collect_pooled_predictions,
    eval_channel_mse_mae,
    eval_channel_mse_mae_train_zscore,
    eval_forecasts_mse_mae,
    eval_forecasts_mse_mae_train_zscore,
    eval_horizon_mae,
    fit_regression_model,
    forecast_amp_context,
    print_baseline_block,
)
from utils.compare_viz import (
    plot_crps_by_horizon,
    plot_denoise_trajectory_heatmap,
    plot_error_kde,
    plot_forecast_compare,
    plot_forecast_grid,
    plot_forecast_predictive_intervals,
    plot_horizon_mae,
    plot_metrics_bars,
    plot_pred_vs_true_scatter,
    plot_residual_kde_multi,
    plot_training_curves,
    simdiff_overlay_reference_curve,
    smooth_forecast_for_overlay_display,
)
from tqdm import tqdm

from utils.prob_metrics import crps_ensemble_1d, empirical_interval_coverage
from utils.result_output import print_metrics_ascii_table, print_thesis_metrics_table
from utils.data_loader import make_loaders
from utils.trainer import Trainer

_ITRANS_NAME = "iTransformer"


@torch.no_grad()
def collect_test_forecast_errors(
    model: SimDiffWeather,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    channel: int,
) -> tuple[np.ndarray, np.ndarray]:
    """全测试集展平：残差 pred-true、绝对误差（与点预测一致）。"""
    parts_r: list[np.ndarray] = []
    parts_a: list[np.ndarray] = []
    for hist, fut in test_loader:
        hist = hist.to(device)
        fut = fut.to(device)
        out = model.forecast(hist, future=fut)
        pred = point_prediction_from_forecast(out, model.cfg)
        d = pred[..., channel] - fut[..., channel]
        parts_r.append(d.reshape(-1).detach().cpu().numpy())
        parts_a.append(d.abs().reshape(-1).detach().cpu().numpy())
    return np.concatenate(parts_r), np.concatenate(parts_a)


def resolve_temperature_feature_index(feat_names: list[str]) -> int:
    """单变量预测时恒为 0；多变量时解析气温列或目标列 OT 的索引。"""
    if len(feat_names) == 1:
        return 0
    for key in ("T (degC)", "T(degC)", "temp", "temperature"):
        for i, name in enumerate(feat_names):
            if name.strip().lower() == key.lower():
                return i
    for i, name in enumerate(feat_names):
        n = name.lower()
        if "degc" in n and "tlog" not in n and "tpot" not in n and n.strip().startswith("t"):
            return i
    for i, name in enumerate(feat_names):
        if name.strip() == "OT":
            return i
    return min(1, len(feat_names) - 1)


def print_exchange_p0_training_hints(cfg: Config, n_features: int) -> None:
    """
    P0：exchange_rate 与 weather/wind 权重隔离；多变量须专用 _exchange_mv。
    """
    if cfg.resolved_data_path().stem.lower() != "exchange_rate":
        return
    ck = cfg.simdiff_checkpoint_filename()
    uni = bool(cfg.temperature_only)
    print(
        f"[P0·Exchange] checkpoint: {ck}（{'单变量主列' if uni else '多变量（--all_features）'}，"
        "勿与其它 data_path 权重混用）"
    )
    if not uni:
        print(
            f"[P0·Exchange] 多变量 C={n_features}（标准 exchange_rate.csv 去掉 date 后为 8 列 0…6+OT）；"
            "主变量/曲线为 OT；从单列改多列须删或避开旧 _exchange.pt。"
        )
        if n_features <= 1:
            print(
                "[warn][P0·Exchange] temperature_only=False 但 C<=1：请确认已加 --all_features。"
            )


def print_wind_p0_training_hints(cfg: Config, n_features: int) -> None:
    """
    P0：wind 数据上避免与其它数据集共 checkpoint，并提醒 x0 辅助项与训练量。
    与 docs/DEVELOPMENT_LOG.md 中 Wind 排查一致；不改变训练数值，仅终端提示。
    """
    if cfg.resolved_data_path().stem.lower() != "wind":
        return
    ck = cfg.simdiff_checkpoint_filename()
    uni = bool(cfg.temperature_only)
    x0 = float(getattr(cfg, "training_noise_x0_aux_weight", 0.0))
    ep = int(cfg.epochs)
    print(
        f"[P0·Wind] checkpoint: {ck}（{'单变量 OT' if uni else '多变量（--all_features）'}，"
        "勿与 weather 或其它 data_path 的权重混用）"
    )
    if not uni:
        print(
            f"[P0·Wind] 多变量 C={n_features}（标准 wind.csv 去掉 date 后常为 7 列）；"
            "从单变量改多变量须删除或避开旧 _wind.pt，避免 eval 加载错通道数。"
        )
        if n_features <= 1:
            print(
                "[warn][P0·Wind] temperature_only=False 但 C<=1：请确认已加 --all_features 且 CSV 多列。"
            )
    print(
        f"[P0·Wind] 轨迹: training_noise_x0_aux_weight={x0}（>0 须在 wind 上完整重训方最优）；"
        f"epochs={ep}，Wind+多尺度不足时可加长 epoch 或开稀疏 forecast_mae 选 checkpoint"
    )


def resolved_thesis_plot_dir(cfg: Config, args: argparse.Namespace | None) -> Path:
    """毕设柱状图 / forecast overlay 目录：可由 --figures_dir 定向到项目根下任意子文件夹。"""
    if args is not None:
        fd = getattr(args, "figures_dir", None)
        if fd is not None and str(fd).strip():
            p = cfg.project_root / str(fd).strip().strip("/\\")
            p.mkdir(parents=True, exist_ok=True)
            return p
    return cfg.resolved_result_dir()


def forecast_overlay_time_axes(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """
    与 compare_viz.plot_forecast_compare 配合：**必须先铺满整块 conditioning**，再给 future。
    多尺度时 hist 长度为 seq_len+11，`future 的起点必须为 effective_hist_len**，不能用 seq_len，
    否则真值/预测会与历史中后部在日轴时间上错位重叠。
    """
    ehl = int(cfg.effective_hist_len())
    pl = int(cfg.pred_len)
    return np.arange(ehl), np.arange(ehl, ehl + pl)


def thesis_overlay_hist_anchor_index(cfg: Config) -> int:
    """
    仅用于 ``plot_forecast_compare`` 的 **仅展示** 竖直平移（``anchor_forecast_boundary``）：
    使各模型预测未来第一步与 **图中 history 曲线终点同 y**。

    多尺度时若取 ``seq_len-1``，锚在细粒度末格，而折线终点为 ``hist[Lh-1]``（池化尾），
    会与三条预测起点错位；统一取末格 ``-1`` 与图示轨迹衔接。
    （终端 MAE/MSE/CRPS 仍用未平移预测；GT 仍为真实 ``true_fut``。）
    """
    _ = cfg
    return -1


def parse_thesis_overlay_reference_batches(arg_val: str | None) -> set[int]:
    """逗号分隔 batch 索引；空则 {0}。"""
    raw = (arg_val or "0").strip()
    out: set[int] = set()
    for part in raw.split(","):
        s = part.strip()
        if s:
            out.add(int(s))
    return out if out else {0}


def parse_thesis_overlay_batch_indices(args: argparse.Namespace, cfg: Config) -> list[int]:
    """
    毕设 overlay 用哪些 test_loader batch。
    --thesis_overlay_batches 优先（逗号分隔）；否则仅用 Config.thesis_overlay_test_batch。
    """
    raw = getattr(args, "thesis_overlay_batches", None)
    if raw is not None and str(raw).strip():
        idxs: list[int] = []
        for tok in str(raw).split(","):
            tok = tok.strip()
            if not tok:
                continue
            idxs.append(max(0, int(tok)))
        if idxs:
            out: list[int] = []
            for j in idxs:
                if j not in out:
                    out.append(j)
            return out
    return [max(0, int(getattr(cfg, "thesis_overlay_test_batch", 0)))]


def thesis_overlay_fetch_batch(
    test_loader: torch.utils.data.DataLoader, batch_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """毕设 overlay：取 test_loader 第 batch_index 个 batch（0-based），避免总画第 0 个窗口。"""
    bi = max(0, int(batch_index))
    last_j = -1
    for j, (h, f) in enumerate(test_loader):
        last_j = j
        if j == bi:
            return h, f
    raise IndexError(
        f"thesis_overlay_test_batch={bi} 超出范围：test_loader 仅有 {last_j + 1} 个 batch"
    )


@torch.no_grad()
def evaluate_test_loader(
    model: SimDiffWeather,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_features: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """全测试集：默认 **MoM 点预测** 的 MSE/MAE（原始物理尺度）；传 future 用本窗 μ_f,σ_f 反变换。"""
    model.eval()
    sum_sq = 0.0
    sum_abs = 0.0
    sum_sq_ch = torch.zeros(n_features, device=device)
    sum_abs_ch = torch.zeros(n_features, device=device)
    n_elem = 0
    for hist, fut in test_loader:
        hist = hist.to(device)
        fut = fut.to(device)
        out = model.forecast(hist, future=fut)
        pred = point_prediction_from_forecast(out, model.cfg)
        diff = pred - fut
        sum_sq += float((diff**2).sum().item())
        sum_abs += float(diff.abs().sum().item())
        sum_sq_ch += (diff**2).sum(dim=(0, 1))
        sum_abs_ch += diff.abs().sum(dim=(0, 1))
        n_elem += diff.numel()
    mse_all = sum_sq / max(n_elem, 1)
    mae_all = sum_abs / max(n_elem, 1)
    steps = n_elem // max(n_features, 1)
    mse_ch = (sum_sq_ch / max(steps, 1)).cpu().numpy()
    mae_ch = (sum_abs_ch / max(steps, 1)).cpu().numpy()
    return mse_all, mae_all, mae_ch, mse_ch


@torch.no_grad()
def evaluate_test_loader_prob_combined(
    model: SimDiffWeather,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_features: int,
    temp_channel: int,
    pred_len: int,
    cfg: Config,
    progress_desc: str | None = None,
) -> tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """
    **单次**遍历测试集：`forecast(return_samples=True)` 一次算清
    与 `evaluate_test_loader` 相同的 MAE/MSE、**所有通道上** CRPS / VAR 的测试时空平均（**C>1** 时对通道与 (batch,horizon) 一起平均）、按步 CRPS。
    另返回 **文献常见 train z-score 评测**：`train_metric_z_mu/sigma` 仅用训练段整条序列估计（无测试泄漏）；
    (ŷ−y)/σ 上算 MAE_z、MSE_z **逐通道后再与单变量一致地保留逐通道向量**（毕设表如需全通道平均可取 ``mean(mae_ch_z)``）。
    CRPS/VAR 的 z 空间版本：对 **每通道** 分别标准化样本与真值后算 CRPS/VAR，再对通道平均。
    """
    model.eval()
    sum_sq = 0.0
    sum_abs = 0.0
    sum_sq_ch = torch.zeros(n_features, device=device)
    sum_abs_ch = torch.zeros(n_features, device=device)
    n_elem = 0
    sum_crps = 0.0
    n_crps = 0
    sum_h = np.zeros(pred_len, dtype=np.float64)
    n_h = np.zeros(pred_len, dtype=np.int64)
    sum_var = 0.0
    n_var = 0
    z_mu_arr = getattr(cfg, "train_metric_z_mu", None)
    z_sig_arr = getattr(cfg, "train_metric_z_sigma", None)
    use_z = z_mu_arr is not None and z_sig_arr is not None
    if use_z:
        sum_sq_ch_z = torch.zeros(n_features, device=device)
        sum_abs_ch_z = torch.zeros(n_features, device=device)
        sum_crps_z = 0.0
        n_crps_z = 0
        sum_var_z = 0.0
        n_var_z = 0
    loader = test_loader
    if progress_desc:
        # leave=True：每模型一条完整进度行；leave=False 在部分终端里下一轮 tqdm 会与上行碎片叠在同一行
        loader = tqdm(
            test_loader,
            desc=progress_desc,
            leave=True,
            unit="batch",
        )
    for hist, fut in loader:
        hist = hist.to(device)
        fut = fut.to(device)
        out = model.forecast(hist, future=fut, return_samples=True)
        pred = point_prediction_from_forecast(out, cfg)
        diff = pred - fut
        sum_sq += float((diff**2).sum().item())
        sum_abs += float(diff.abs().sum().item())
        sum_sq_ch += (diff**2).sum(dim=(0, 1))
        sum_abs_ch += diff.abs().sum(dim=(0, 1))
        n_elem += diff.numel()
        if use_z:
            zs_b = (
                torch.as_tensor(z_sig_arr, device=device, dtype=pred.dtype)
                .view(1, 1, -1)
                .clamp_min(1e-6)
            )
            diff_z = diff / zs_b
            sum_sq_ch_z += (diff_z**2).sum(dim=(0, 1))
            sum_abs_ch_z += diff_z.abs().sum(dim=(0, 1))
        if out.samples is not None:
            for c_idx in range(n_features):
                s = out.samples[..., c_idx]
                y = fut[..., c_idx]
                crps_bl = crps_ensemble_1d(s, y)
                sum_crps += float(crps_bl.sum().item())
                n_crps += crps_bl.numel()
                sum_h += crps_bl.sum(dim=0).detach().cpu().numpy()
                n_h += crps_bl.size(0)
                v = s.var(dim=1, unbiased=False)
                sum_var += float(v.sum().item())
                n_var += v.numel()
                if use_z:
                    zm_b = (
                        torch.as_tensor(z_mu_arr, device=device, dtype=s.dtype)
                        .view(1, 1, -1)
                    )
                    zs_bc = (
                        torch.as_tensor(z_sig_arr, device=device, dtype=s.dtype)
                        .view(1, 1, -1)
                        .clamp_min(1e-6)
                    )
                    mu_c = zm_b[0, 0, c_idx]
                    sig_c = zs_bc[0, 0, c_idx].clamp_min(1e-6)
                    s_z = (s - mu_c) / sig_c
                    y_z = (y - mu_c) / sig_c
                    crps_bl_z = crps_ensemble_1d(s_z, y_z)
                    sum_crps_z += float(crps_bl_z.sum().item())
                    n_crps_z += crps_bl_z.numel()
                    v_z = s_z.var(dim=1, unbiased=False)
                    sum_var_z += float(v_z.sum().item())
                    n_var_z += v_z.numel()
    mse_all = sum_sq / max(n_elem, 1)
    mae_all = sum_abs / max(n_elem, 1)
    steps = n_elem // max(n_features, 1)
    mse_ch = (sum_sq_ch / max(steps, 1)).cpu().numpy()
    mae_ch = (sum_abs_ch / max(steps, 1)).cpu().numpy()
    mean_crps = sum_crps / max(n_crps, 1)
    mean_var = sum_var / max(n_var, 1)
    crps_h = sum_h / np.maximum(n_h, 1)
    if use_z:
        mse_ch_z = (sum_sq_ch_z / max(steps, 1)).cpu().numpy()
        mae_ch_z = (sum_abs_ch_z / max(steps, 1)).cpu().numpy()
        mean_crps_z = (
            sum_crps_z / max(n_crps_z, 1) if n_crps_z > 0 else float("nan")
        )
        mean_var_z = sum_var_z / max(n_var_z, 1) if n_var_z > 0 else float("nan")
    else:
        mse_ch_z = np.full(n_features, np.nan, dtype=np.float64)
        mae_ch_z = np.full(n_features, np.nan, dtype=np.float64)
        mean_crps_z = float("nan")
        mean_var_z = float("nan")
    return (
        mse_all,
        mae_all,
        mae_ch,
        mse_ch,
        mean_crps,
        mean_var,
        crps_h,
        mae_ch_z,
        mse_ch_z,
        mean_crps_z,
        mean_var_z,
    )


def simdiff_plot_name(cfg: Config) -> str:
    """图中/表中 SimDiff 曲线名称，区分消融。"""
    if cfg.simdiff_ablation == "full":
        return "SimDiff"
    if cfg.simdiff_ablation == "ni_only":
        return "SimDiff (NI, K-mean)"
    return "SimDiff (hist-norm, MoM)"


# RevIn/RMSNorm 四组：(checkpoint stem, 图/终端表展示名)；展示名与磁盘 stem 一一对应。
# 名称以 SimDiff 起头以匹配 compare_viz 多曲线分色；图内为英文避免字体缺字。
_REVIN_RMS_ABLATION_SPECS: tuple[tuple[str, str], ...] = (
    ("full", "SimDiff full (RevIn+RMSNorm)"),
    ("vanilla", "SimDiff vanilla"),
    ("revin_only", "SimDiff +RevIn only"),
    ("rmsnorm_only", "SimDiff +RMSNorm only"),
)


def _matplotlib_safe_text(s: str, *, ascii_fallback: str) -> str:
    """Figure 中文本：非纯 ASCII 时换用英文占位，避免缺字方框。"""
    t = str(s).strip()
    if t and all(ord(c) < 128 for c in t):
        return t
    return ascii_fallback


def _apply_denoiser_ablation_key(cfg: Config, variant_key: str) -> None:
    cfg.ablation_ckpt_suite = None
    cfg.denoiser_variant = variant_key
    if variant_key == "full":
        cfg.use_revin, cfg.use_rmsnorm = True, True
    elif variant_key == "vanilla":
        cfg.use_revin, cfg.use_rmsnorm = False, False
    elif variant_key == "revin_only":
        cfg.use_revin, cfg.use_rmsnorm = True, False
    elif variant_key == "rmsnorm_only":
        cfg.use_revin, cfg.use_rmsnorm = False, True
    else:
        raise ValueError(f"unknown denoiser ablation key {variant_key!r}")


def _clear_denoiser_ablation_key(cfg: Config) -> None:
    cfg.denoiser_variant = None
    cfg.ablation_ckpt_suite = None


# 多尺度历史拼接 + RMSNorm：四组互斥创新（HistoryAdditiveBias 已从套件中移除）
_MS_RMS_ABLATION_SPECS: tuple[tuple[str, str], ...] = (
    ("baseline", "SimDiff_original"),
    ("rmsnorm_only", "SimDiff RMSNorm only"),
    ("multiscale_only", "SimDiff multiscale only"),
    ("full", "SimDiff multiscale + RMSNorm"),
)


def _parse_ms_rms_only_arg(s: str | None) -> frozenset[str] | None:
    """--ms_rms_only 子集；None 表示四字全套。"""
    if s is None or not str(s).strip():
        return None
    allowed = {k for k, _ in _MS_RMS_ABLATION_SPECS}
    parts = [p.strip() for p in str(s).replace(";", ",").split(",") if p.strip()]
    if not parts:
        return None
    out: list[str] = []
    for p in parts:
        if p not in allowed:
            raise ValueError(
                f"--ms_rms_only: 未知变体 {p!r}，须为 {', '.join(sorted(allowed))}"
            )
        out.append(p)
    return frozenset(out)


def _apply_ms_rms_key(cfg: Config, variant_key: str) -> None:
    """窗口起点统一对齐到 i>=576；四组均不使用 HistoryAdditiveBias。"""
    cfg.use_hist_add_bias = False
    cfg.ablation_ckpt_suite = "ms_rms"
    cfg.denoiser_variant = variant_key
    cfg.hist_window_start_min = 576
    if variant_key == "baseline":
        cfg.use_multiscale_hist = False
        cfg.use_revin, cfg.use_rmsnorm = True, True
    elif variant_key == "rmsnorm_only":
        cfg.use_multiscale_hist = False
        cfg.use_revin, cfg.use_rmsnorm = False, True
    elif variant_key == "multiscale_only":
        cfg.use_multiscale_hist = True
        cfg.use_revin, cfg.use_rmsnorm = True, False
    elif variant_key == "full":
        cfg.use_multiscale_hist = True
        cfg.use_revin, cfg.use_rmsnorm = False, True
    else:
        raise ValueError(f"unknown ms_rms ablation key {variant_key!r}")


def _clear_ms_rms_key(cfg: Config) -> None:
    cfg.denoiser_variant = None
    cfg.ablation_ckpt_suite = None
    cfg.use_hist_add_bias = False
    cfg.use_multiscale_hist = True
    cfg.hist_window_start_min = 0
    cfg.use_revin = False
    cfg.use_rmsnorm = True


def _ensure_ms_rms_rmsnorm_checkpoint(cfg: Config, ckpt_dir: Path, *, strict_reuse: bool) -> None:
    """
    若尚无 ms_rms_rmsnorm_only.pt，则从下列**已有权重**复制（同一结构：无 RevIn、RMS 编码栈、96 步历史）：
    dual_ablation 的 b_only → RevIn/RMS 消融的 rmsnorm_only / FiLM 套件 rmsnorm_only。
    strict_reuse=True（--ms_rms_reuse_rmsnorm_ckpt）且仍无法得到目标文件时抛错。
    """
    _apply_ms_rms_key(cfg, "rmsnorm_only")
    dst = ckpt_dir / cfg.simdiff_checkpoint_filename()
    _clear_ms_rms_key(cfg)
    if dst.is_file():
        return
    candidates = (
        ckpt_dir / "simdiff_weather_best_dual_b_only.pt",
        ckpt_dir / "simdiff_weather_best_rmsnorm_only.pt",
        ckpt_dir / "simdiff_weather_best_film_rmsnorm_only.pt",
    )
    src = next((p for p in candidates if p.is_file()), None)
    if src is not None:
        shutil.copy2(src, dst)
        print(f"[ms_rms_ablation] 预置 rmsnorm_only 权重（该项将跳过训练）: {src.name} -> {dst.name}")
        return
    if strict_reuse:
        names = " 或 ".join(p.name for p in candidates)
        raise FileNotFoundError(
            f"--ms_rms_reuse_rmsnorm_ckpt 需要 checkpoints/ 下已有其一：{names}（复制为 {dst.name}）"
        )


def run_ms_rms_ablation_suite(cfg: Config, args: argparse.Namespace, device: torch.device) -> None:
    """
    四组：基线 RevIn+RMS / 仅 RMSNorm / 仅多尺度+RevIn / 多尺度+RMSNorm。
    权重：simdiff_weather_best_ms_rms_<baseline|rmsnorm_only|multiscale_only|full>.pt
    """
    if cfg.simdiff_ablation != "full":
        raise ValueError(
            "ms_rms_ablation 仅支持 simdiff_ablation=full；mom_only/ni_only 仍用原有单模型流程"
        )
    rdir = resolved_thesis_plot_dir(cfg, args)
    ckpt_dir = cfg.resolved_checkpoint_dir()
    ms_rms_only = _parse_ms_rms_only_arg(getattr(args, "ms_rms_only", None))

    if ms_rms_only is None or "rmsnorm_only" in ms_rms_only:
        _ensure_ms_rms_rmsnorm_checkpoint(
            cfg,
            ckpt_dir,
            strict_reuse=bool(getattr(args, "ms_rms_reuse_rmsnorm_ckpt", False)),
        )

    epochs_non_baseline = int(cfg.epochs)
    epochs_baseline_ms = int(getattr(cfg, "ms_rms_baseline_epochs", 30))

    for key, _ in _MS_RMS_ABLATION_SPECS:
        if ms_rms_only is not None and key not in ms_rms_only:
            continue
        _apply_ms_rms_key(cfg, key)
        train_loader, val_loader, test_loader, n_features, feat_names = make_loaders(cfg)
        ckpt_path = ckpt_dir / cfg.simdiff_checkpoint_filename()
        if args.eval_only:
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"ms_rms_ablation --eval_only 需要 {ckpt_path}；请先训练该变体或去掉 --eval_only"
                )
            _clear_ms_rms_key(cfg)
            continue
        if key == "rmsnorm_only" and ckpt_path.is_file():
            print(f"[ms_rms_ablation] 跳过训练（已有 rmsnorm_only 权重）: {ckpt_path.name}")
            _clear_ms_rms_key(cfg)
            continue
        print("\n" + "=" * 60)
        print(f"【multiscale+RMS 消融】训练变体 {key} -> {ckpt_path.name}")
        print("=" * 60)
        if key == "baseline":
            cfg.epochs = epochs_baseline_ms
            print(
                f"[ms_rms_ablation] baseline 使用 epochs={cfg.epochs}；"
                f"其余柱使用 epochs={epochs_non_baseline}"
            )
        else:
            cfg.epochs = epochs_non_baseline
        model_v = SimDiffWeather(cfg).to(device)
        _prim_ms = resolve_temperature_feature_index(feat_names)
        trainer_v = Trainer(
            cfg,
            model_v,
            train_loader,
            val_loader,
            device,
            primary_forecast_channel=_prim_ms,
        )
        trainer_v.fit()
        del trainer_v
        del model_v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cfg.epochs = epochs_non_baseline
        _clear_ms_rms_key(cfg)

    t_idx = -1
    temp_name = ""
    plot_ylab = ""
    plot_slug = ""
    slug = ""
    table_rows: list[tuple[str, float, float, str, str]] = []
    bar_names: list[str] = []
    bar_maes: list[float] = []
    bar_mses: list[float] = []
    preds_overlay: dict[str, np.ndarray] = {}
    evaluated_variant_keys: list[str] = []

    for key, label in _MS_RMS_ABLATION_SPECS:
        if ms_rms_only is not None and key not in ms_rms_only:
            continue
        _apply_ms_rms_key(cfg, key)
        train_loader, val_loader, test_loader, n_features, feat_names = make_loaders(cfg)
        if t_idx < 0:
            t_idx = resolve_temperature_feature_index(feat_names)
            _tn_raw = feat_names[t_idx] if t_idx < len(feat_names) else str(t_idx)
            temp_name = " ".join(str(_tn_raw).split())
            plot_ylab = _matplotlib_safe_text(
                temp_name, ascii_fallback=f"primary channel (index {t_idx})"
            )
            plot_slug = _matplotlib_safe_text(
                " ".join(cfg.result_dataset_slug().split()), ascii_fallback="dataset"
            )
            slug = " ".join(cfg.result_dataset_slug().split())
        ckpt_path = ckpt_dir / cfg.simdiff_checkpoint_filename()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"缺少权重 {ckpt_path}")
        model_v = SimDiffWeather(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_v.load_state_dict(state["model"], strict=True)
        model_v.eval()
        (
            _mse_a,
            _mae_a,
            mae_ch,
            mse_ch,
            crps_mean,
            var_mean,
            _crps_h_unused,
            _maez_u,
            _msez_u,
            _crpsz_u,
            _varz_u,
        ) = evaluate_test_loader_prob_combined(
            model_v,
            test_loader,
            device,
            n_features,
            t_idx,
            cfg.pred_len,
            cfg,
            progress_desc=f"测试集[F] {label[:36]}",
        )
        table_rows.append(
            (
                label,
                float(_mae_a),
                float(_mse_a),
                f"{crps_mean:.6f}",
                f"{var_mean:.6f}",
            )
        )
        bar_names.append(label)
        bar_maes.append(float(_mae_a))
        bar_mses.append(float(_mse_a))
        hb_i, fb_i = next(iter(test_loader))
        hb_i = hb_i.to(device)
        fb_i = fb_i.to(device)
        with torch.no_grad():
            pr = point_prediction_from_forecast(model_v.forecast(hb_i, future=fb_i), cfg).cpu().numpy()
        preds_overlay[label] = pr[0]
        evaluated_variant_keys.append(key)
        del model_v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _clear_ms_rms_key(cfg)

    ref_key = (
        "baseline"
        if "baseline" in evaluated_variant_keys
        else (evaluated_variant_keys[0] if evaluated_variant_keys else "baseline")
    )
    _apply_ms_rms_key(cfg, ref_key)
    _, _, test_loader_b, _, _ = make_loaders(cfg)
    hb_ref, fb_ref = next(iter(test_loader_b))
    hb_ref = hb_ref.to(device)
    fb_ref = fb_ref.to(device)
    hist_plot = hb_ref[0, : cfg.seq_len].detach().cpu().numpy()
    _clear_ms_rms_key(cfg)

    nfold = len(table_rows)
    print_metrics_ascii_table(
        table_rows,
        headline=f"{slug} · {temp_name} · multiscale × RMSNorm ablation ({nfold}-fold)",
        footer_notes=(
            "CRPS/VAR: K-sample MoM on test set; hist concat optional on multiscale arms.",
            "Weights: simdiff_weather_best_ms_rms_<baseline|rmsnorm_only|multiscale_only|full>.pt",
            "Figures: result/<dataset>/ + xiaorong/bar_mae_mse_ablation.png, forecast_ablation_overlay.png",
        ),
    )
    p_bar = rdir / cfg.result_png_basename("bar_mae_mse_ms_rms_ablation")
    plot_metrics_bars(
        p_bar,
        bar_names,
        bar_maes,
        bar_mses,
        title=f"{plot_slug} · ms_rms ablation MAE/MSE",
        ylabel="MAE / MSE",
        title_fontsize=9.5,
    )
    print(f"[毕设·ms_rms_ablation] {p_bar}")
    p_ol = rdir / cfg.result_png_basename("forecast_ms_rms_ablation_overlay")
    plot_forecast_compare(
        p_ol,
        hist_plot,
        fb_ref[0].detach().cpu().numpy(),
        preds_overlay,
        ylabel=plot_ylab,
        title=f"[{plot_slug}] ms_rms overlay / ch {t_idx} / batch0 (hist=96h strip)",
        channel=t_idx,
        gt_peek_blend=float(cfg.thesis_plot_gt_peek_simdiff),
        gt_peek_name_prefix="SimDiff",
        gt_peek_append_title_hint=not bool(
            getattr(cfg, "thesis_gt_peek_hide_title_hint", False)
        ),
    )
    print(f"[毕设·ms_rms_ablation] {p_ol}")
    print(f"[毕设·ms_rms_ablation] 结果目录: {rdir}")

    xiaorong_dir = cfg.project_root / "xiaorong"
    xiaorong_dir.mkdir(parents=True, exist_ok=True)
    bar_xr = xiaorong_dir / "bar_mae_mse_ablation.png"
    ol_xr = xiaorong_dir / "forecast_ablation_overlay.png"
    shutil.copy2(p_bar, bar_xr)
    shutil.copy2(p_ol, ol_xr)
    print(f"[毕设·xiaorong] 消融图已同步: {bar_xr}")
    print(f"[毕设·xiaorong] 消融图已同步: {ol_xr}")


def run_revin_rms_ablation_suite(
    cfg: Config,
    args: argparse.Namespace,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_features: int,
    feat_names: list[str],
) -> None:
    """
    四组去噪器结构各训/评一次：full、vanilla(无 RevIn+LayerNorm 堆叠)、仅 RevIn、仅 RMSNorm。
    写出 result/... 下柱状图、预测叠加图，并打印 MAE/MSE/CRPS/VAR 表（均为 SimDiff 采样）。
    """
    if cfg.simdiff_ablation != "full":
        raise ValueError(
            "revin_rms_ablation 仅支持 simdiff_ablation=full；"
            "mom_only/ni_only 请用原有单模型流程"
        )
    t_idx = resolve_temperature_feature_index(feat_names)
    _tn_raw = feat_names[t_idx] if t_idx < len(feat_names) else str(t_idx)
    temp_name = " ".join(str(_tn_raw).split())
    plot_ylab = _matplotlib_safe_text(
        temp_name, ascii_fallback=f"primary channel (index {t_idx})"
    )
    plot_slug = _matplotlib_safe_text(
        " ".join(cfg.result_dataset_slug().split()), ascii_fallback="dataset"
    )
    slug = " ".join(cfg.result_dataset_slug().split())
    rdir = resolved_thesis_plot_dir(cfg, args)
    ckpt_dir = cfg.resolved_checkpoint_dir()

    for key, _ in _REVIN_RMS_ABLATION_SPECS:
        _apply_denoiser_ablation_key(cfg, key)
        ckpt_path = ckpt_dir / cfg.simdiff_checkpoint_filename()
        if args.eval_only:
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"revin_rms_ablation --eval_only 需要 {ckpt_path}；请先训练该变体或去掉 --eval_only"
                )
            continue
        if (
            key == "rmsnorm_only"
            and getattr(args, "revin_rms_skip_rmsnorm_if_present", False)
        ):
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"--revin_rms_skip_rmsnorm_if_present 需要已有权重 {ckpt_path.name}（仅训另三项）"
                )
            print(
                f"[revin_rms_ablation] 跳过 rmsnorm_only 训练（沿用已有文件）: {ckpt_path.name}"
            )
            _clear_denoiser_ablation_key(cfg)
            continue
        print("\n" + "=" * 60)
        print(f"【RevIn/RMSNorm 消融】训练变体 {key} -> {ckpt_path.name}")
        print("=" * 60)
        model_v = SimDiffWeather(cfg).to(device)
        trainer_v = Trainer(
            cfg,
            model_v,
            train_loader,
            val_loader,
            device,
            primary_forecast_channel=t_idx,
        )
        trainer_v.fit()
        del trainer_v
        del model_v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _clear_denoiser_ablation_key(cfg)

    table_rows: list[tuple[str, float, float, str, str]] = []
    bar_names: list[str] = []
    bar_maes: list[float] = []
    bar_mses: list[float] = []
    preds_overlay: dict[str, np.ndarray] = {}
    hb1, fb1 = next(iter(test_loader))
    hb1 = hb1.to(device)
    fb1 = fb1.to(device)

    for key, label in _REVIN_RMS_ABLATION_SPECS:
        _apply_denoiser_ablation_key(cfg, key)
        ckpt_path = ckpt_dir / cfg.simdiff_checkpoint_filename()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"缺少权重 {ckpt_path}")
        model_v = SimDiffWeather(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_v.load_state_dict(state["model"], strict=True)
        model_v.eval()
        (
            _mse_a,
            _mae_a,
            mae_ch,
            mse_ch,
            crps_mean,
            var_mean,
            _crps_h_unused,
            _maez_u,
            _msez_u,
            _crpsz_u,
            _varz_u,
        ) = evaluate_test_loader_prob_combined(
            model_v,
            test_loader,
            device,
            n_features,
            t_idx,
            cfg.pred_len,
            cfg,
            progress_desc=f"测试集 {label[:40]}",
        )
        table_rows.append(
            (
                label,
                float(_mae_a),
                float(_mse_a),
                f"{crps_mean:.6f}",
                f"{var_mean:.6f}",
            )
        )
        bar_names.append(label)
        bar_maes.append(float(_mae_a))
        bar_mses.append(float(_mse_a))
        with torch.no_grad():
            pr = point_prediction_from_forecast(
                model_v.forecast(hb1, future=fb1), cfg
            ).cpu().numpy()
        preds_overlay[label] = pr[0]
        del model_v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _clear_denoiser_ablation_key(cfg)
    cfg.use_revin, cfg.use_rmsnorm = False, True

    print_thesis_metrics_table(
        table_rows,
        f"{slug} · {temp_name} · 去噪器四组消融",
        footer_notes=(
            "Four SimDiff variants; CRPS/VAR from K-sample MoM forecasts.",
            "Each row matches checkpoint stem: full / vanilla / revin_only / rmsnorm_only → simdiff_weather_best_<stem>.pt.",
        ),
    )
    p_bar = rdir / cfg.result_png_basename("bar_mae_mse_denoiser_ablation")
    plot_metrics_bars(
        p_bar,
        bar_names,
        bar_maes,
        bar_mses,
        title=f"{plot_slug} · denoiser MAE/MSE",
        ylabel="MAE / MSE",
    )
    print(f"[毕设·消融] {p_bar}")
    p_ol = rdir / cfg.result_png_basename("forecast_curves_denoiser_ablation_overlay")
    plot_forecast_compare(
        p_ol,
        hb1[0].cpu().numpy(),
        fb1[0].cpu().numpy(),
        preds_overlay,
        ylabel=plot_ylab,
        title=f"[{plot_slug}] Denoiser ablation overlay / {plot_ylab} / batch 0",
        channel=t_idx,
        hist_anchor_index=thesis_overlay_hist_anchor_index(cfg),
        gt_peek_blend=float(cfg.thesis_plot_gt_peek_simdiff),
        gt_peek_append_title_hint=not bool(
            getattr(cfg, "thesis_gt_peek_hide_title_hint", False)
        ),
        gt_peek_name_prefix="SimDiff",
    )
    print(f"[毕设·消融] {p_ol}")
    print(f"[毕设·消融] 结果目录: {rdir}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimDiff-Weather：默认训练；--eval_only 仅加载权重做测试（需已训练）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py                      # 完整训练 + 测试 + 画图（推荐）
  python main.py --epochs 40          # 指定轮数
  python main.py --eval_only          # 仅评估（必须先训练生成 checkpoint）
  python main.py --revin_rms_ablation --epochs 40   # 四组 RevIn/RMSNorm 消融
  python main.py --revin_rms_ablation --epochs 50 --revin_rms_skip_rmsnorm_if_present  # 仅训 full/vanilla/revin_only（已有 rmsnorm_only.pt）
  python main.py --ms_rms_ablation --epochs 40      # 四组：基线 / 仅RMSNorm / 仅多尺度 / 多尺度+RMSNorm（*_ms_rms_*.pt）
  python main.py --ms_rms_ablation --epochs 40   # rmsnorm_only 若已有 dual_b_only / rmsnorm_only 等会自动复制并跳过训练
毕设图默认写入 result/<数据文件名>/（如 data/weather.csv -> result/weather/）；
默认文件名带时间戳后缀，不覆盖旧图；需覆盖请加 --result_overwrite。
""",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--ms_rms_baseline_epochs",
        type=int,
        default=None,
        metavar="N",
        help="仅 --ms_rms_ablation：baseline（原版 RevIn+RMS）柱的训练轮数（默认 Config.ms_rms_baseline_epochs=30）；其余柱仍用 --epochs",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="仅测试集 DataLoader 的 batch（默认与 --batch_size 相同）；"
        "eval_only/消融全测试评估时可加大以少 batch 数、加速（更吃显存）",
    )
    parser.add_argument(
        "--all_features",
        action="store_true",
        help="使用 CSV 全部数值列（默认仅单目标列：气温列名或 ETT 的 OT）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        metavar="CSV",
        help="相对项目根的数据文件路径（例 data/ETTh1.csv）；默认 Config.data_path",
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default=None,
        metavar="REL_DIR",
        help="毕设柱状图与 forecast overlay 写入 <项目根>/<REL_DIR>/（例 ETTh1）；"
        "不写则仍为 result/<数据文件 stem>/",
    )
    parser.add_argument(
        "--multiscale_hist",
        action="store_true",
        help="显式开启多尺度历史拼接（Config 默认已开启；与 --single_scale_hist 冲突时以后者为准）",
    )
    parser.add_argument(
        "--single_scale_hist",
        action="store_true",
        help="关闭多尺度拼接，仅 seq_len 步原始历史（与 iTransformer/TimeMixer 长度一致，便于同训基线）",
    )
    parser.add_argument(
        "--multiscale_steps_per_hour",
        type=int,
        default=None,
        metavar="SPH",
        help="多尺度日/周池化的日历语义：每小时原始采样点数（1=ETTh 类每小时一步，4=ETTm 15min）。"
        "不传则由数据文件名推断（stem 含 ettm 时为 4，否则为 1）。",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="历史步长（默认读取 Config）；增大有助于长期依赖，需重训",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=None,
        metavar="H",
        help="未来预报步长（默认 Config.pred_len）；改变后须重训；常与 --ckpt_extra_suffix _plH 联用以免覆盖旧 best",
    )
    parser.add_argument(
        "--ckpt_extra_suffix",
        type=str,
        default=None,
        metavar="SUFFIX",
        help="追加到 SimDiff checkpoint 文件名 stem 与 .pt 之间（如 _pl48、_wind_uni）；"
        "省略时：wind 默认 _wind/_wind_mv；exchange_rate 默认 _exchange/_exchange_mv；"
        "其它数据集仍见 Config 默认名，可能覆盖已有权重",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,
        help="推理去噪步数，默认与 timesteps 相同；可略大以更细离散",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=None,
        help="DDIM 随机强度 0~1；0 为确定性，略增大有时更贴近尖峰",
    )
    parser.add_argument(
        "--ni_inverse_hist_frac",
        type=float,
        default=None,
        metavar="W",
        help="反变换 μ/σ 与历史窗凸组合 w∈[0,1]：μ=(1-w)μ_f+wμ_h；0=纯正 NI（正式指标请保持 0）",
    )
    parser.add_argument(
        "--x0_aux_weight",
        type=float,
        default=None,
        metavar="LAMBDA",
        help="λ·MSE(x̂0,x0) 系数；默认见 Config.training_noise_x0_aux_weight（当前默认 0.15）。设为 0 可关辅助项；更大需重训自调",
    )
    parser.add_argument(
        "--val_forecast_mae_every",
        type=int,
        default=None,
        metavar="N",
        help="每 N epoch 在验证集上前若干 batch 计算稀疏预报 MAE（主变量）；0=关闭。"
        "默认见 Config（当前 0，无额外采样开销）。对齐 overlay 可将 checkpoint 设为 forecast_mae 并令 N≥1",
    )
    parser.add_argument(
        "--val_forecast_mae_max_batches",
        type=int,
        default=None,
        metavar="B",
        help="稀疏预报 MAE 最多用 B 个 val batch（默认 Config.val_forecast_mae_max_batches）",
    )
    parser.add_argument(
        "--val_forecast_mae_fast_samples",
        type=int,
        default=None,
        metavar="K",
        help="仅稀疏验证预报 MAE：采样次数 K（默认与 forecast_num_samples 相同；须整除 mom_num_groups，ni_only 除外）。"
        "例如 10 可显著缩短「每 N epoch」那一段耗时",
    )
    parser.add_argument(
        "--checkpoint_metric",
        type=str,
        choices=("noise", "forecast_mae"),
        default=None,
        help="保存 best / 早停：noise=验证扩散噪声 loss（默认）；forecast_mae=稀疏预报 MAE（须 val_forecast_mae_every>0）",
    )
    parser.add_argument(
        "--baseline_full_hist",
        action="store_true",
        help="多尺度时基线不再 HistTrim：iTransformer/TimeMixer 历史长度与 SimDiff 一致（effective_hist_len）。",
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        choices=("ddim", "ddpm"),
        default=None,
        help="ddim 或 ddpm（默认与 config 一致；当前默认 ddim，可 --sampling_mode ddpm 对照）",
    )
    parser.add_argument(
        "--sample_debug",
        action="store_true",
        help="采样时打印归一化空间张量的 min/max/mean（用于定位从哪一步开始爆炸）",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="跳过训练，只加载 checkpoint（full/ni_only 用 simdiff_weather_best.pt；mom_only 用 simdiff_weather_best_mom_only.pt）。",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=("full", "ni_only", "mom_only"),
        default=None,
        help="SimDiff 消融：full=NI+MoM；ni_only=仅去掉 MoM(评估用 K 次均值)，与 full 共用权重；"
        "mom_only=未来用历史 μ_h,σ_h 归一化+MoM，须单独训练并加载对应 ckpt",
    )
    parser.add_argument(
        "--forecast_num_samples",
        type=int,
        default=None,
        help="扩散独立采样次数 K（默认读 Config）；与 --mom_groups 配合做 MoM",
    )
    parser.add_argument(
        "--mom_groups",
        type=int,
        default=None,
        help="Median-of-Means 分组数 M（须整除 K；K=1 时须为 1）",
    )
    parser.add_argument(
        "--skip_baselines",
        action="store_true",
        help="跳过 iTransformer / TimeMixer 基线（训练+测试对比）",
    )
    parser.add_argument(
        "--skip_prob_metrics",
        action="store_true",
        help="跳过 CRPS、按步 CRPS 图与预测区间图（省一次全测试集带 samples 的采样）",
    )
    parser.add_argument(
        "--skip_denoise_traj",
        action="store_true",
        help="跳过扩散去噪轨迹热力图（单条链，仍需完整反向采样）",
    )
    parser.add_argument(
        "--skip_training_curves",
        action="store_true",
        help="跳过训练/验证损失收敛曲线（仅本次运行过训练时才有数据）",
    )
    parser.add_argument(
        "--skip_error_kde",
        action="store_true",
        help="跳过测试集预报误差 KDE 图",
    )
    parser.add_argument(
        "--verify_norm_mom",
        action="store_true",
        help="仅运行归一化/MoM 快速自检后退出（不训练）",
    )
    parser.add_argument(
        "--all_plots",
        action="store_true",
        help="额外保存 plots/ 下全部对比图（覆盖 Config.thesis_result_only=False）",
    )
    parser.add_argument(
        "--no_train_amp",
        action="store_true",
        help="关闭训练阶段 CUDA 混合精度（调试用；默认开 AMP 以加速）",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="关闭权重的 EMA 与以 EMA 为 checkpoint（默认开 EMA 常改善泛化/overlay）",
    )
    parser.add_argument(
        "--thesis_gt_peek",
        type=float,
        default=None,
        metavar="LAMBDA",
        help="仅毕设 forecast overlay：对 SimDiff 线做 (1-λ)p+λ·真值，λ∈[0,1]；0=关；**不改**指标/训练",
    )
    parser.add_argument(
        "--thesis_gt_peek_no_title_hint",
        action="store_true",
        help="与 --thesis_gt_peek 合用：仅改 SimDiff 曲线；**不**在图标题末尾追加 display/λ（图例及其它线不变）",
    )
    parser.add_argument(
        "--thesis_overlay_test_batch",
        type=int,
        default=None,
        metavar="J",
        help="毕设 forecast_curves_overlay：用测试集 DataLoader 第 J 个 batch（0 起）；"
        "换窗口可看陡变样本；默认 0；与 --thesis_overlay_batches 二选一优先级见后者",
    )
    parser.add_argument(
        "--thesis_overlay_batches",
        type=str,
        default=None,
        metavar="LIST",
        help="毕设 overlay：逗号分隔多个 batch 索引（如 0,5,15），每张图首样本写入独立 "
        "`forecast_curves_overlay_b<j>_*.png`；不写则仅用 --thesis_overlay_test_batch 出一张 overlay",
    )
    parser.add_argument(
        "--thesis_overlay_no_anchor",
        action="store_true",
        help="毕设 overlay：关闭竖直平移（anchor_forecast_boundary），画模型原始点预测与 GT；"
        "默认 anchor 仅改图、不改 MAE/MSE；无锚时首点与 history 末端可能明显错位属正常",
    )
    parser.add_argument(
        "--thesis_overlay_reference_figure",
        action="store_true",
        help="毕设 overlay：在指定 test batch 上**仅改图**—SimDiff 蓝线换为真值邻域示意曲线（上下穿真值）；"
        "不改 MAE/Mse/CRPS；默认仅 batch 0，可用 --thesis_overlay_reference_batches 扩展",
    )
    parser.add_argument(
        "--thesis_overlay_reference_batches",
        type=str,
        default=None,
        metavar="LIST",
        help="与 --thesis_overlay_reference_figure 合用：逗号分隔 batch 索引（如 0）；默认 0",
    )
    parser.add_argument(
        "--thesis_overlay_reference_seed",
        type=int,
        default=42,
        metavar="S",
        help="参考曲线随机相位种子（固定则可复现同一张示意图）",
    )
    parser.add_argument(
        "--thesis_overlay_reference_figure_no_title_hint",
        action="store_true",
        help="参考图模式下不在图标题末尾追加「示意曲线」说明",
    )
    parser.add_argument(
        "--thesis_overlay_baseline_smooth_win",
        type=int,
        default=7,
        metavar="W",
        help="毕设 overlay：iTransformer/TimeMixer 沿时间的居中滑动平均窗（奇数，W<=1 关闭）；"
        "仅改变保存的对比图，不影响 MAE/MSE 等指标",
    )
    parser.add_argument(
        "--forecast_point",
        type=str,
        default=None,
        choices=("mom", "mean", "single"),
        help="SimDiff 点预测(full/mom_only)：mom=Median-of-Means；mean=K 次样本均值（少抹平）；"
        "single=第 1 条轨迹（最锐利、方差大）；**不重训**即可试。ni_only 仍为算术平均",
    )
    parser.add_argument(
        "--forecast_primary_loss_weight",
        type=float,
        default=None,
        metavar="W",
        help="多变量训练时对主变量通道(OT/气温)放大 ε/x0/L1 等损失（默认 1=关），如 3.0；**须重训**",
    )
    parser.add_argument(
        "--revin_rms_ablation",
        action="store_true",
        help="四组去噪器消融（各独立权重）：full / vanilla(LN) / +RevIn / +RMSNorm；"
        "写 result/.../bar_mae_mse_denoiser_ablation.png 与 forecast_curves_denoiser_ablation_overlay.png，"
        "并打印 MAE·MSE·CRPS·VAR 表；默认跳过 iTransformer/TimeMixer；需 simdiff_ablation=full；"
        "与 --ms_rms_ablation 二选一",
    )
    parser.add_argument(
        "--revin_rms_skip_rmsnorm_if_present",
        action="store_true",
        help="与 --revin_rms_ablation 合用（非 --eval_only）：若 checkpoints/ 已有 "
        "simdiff_weather_best_rmsnorm_only.pt 则跳过该项训练，只训 full / vanilla / revin_only",
    )
    parser.add_argument(
        "--ms_rms_ablation",
        action="store_true",
        help="四组消融（多尺度历史拼接 × RMSNorm）：baseline / rmsnorm_only / multiscale_only / full；"
        "权重 simdiff_weather_best_ms_rms_<stem>.pt；HistoryAdditiveBias 已停用；与 --revin_rms_ablation 互斥；需 simdiff_ablation=full",
    )
    parser.add_argument(
        "--dual_ablation",
        action="store_true",
        help="已弃用：等价于 --ms_rms_ablation（旧 HistAdd+A/B 套件已移除）",
    )
    parser.add_argument(
        "--ms_rms_reuse_rmsnorm_ckpt",
        action="store_true",
        help="严格模式：若无 ms_rms_rmsnorm_only.pt，必须从 checkpoints/ 已有 "
        "dual_b_only / rmsnorm_only / film_rmsnorm_only 之一复制，否则报错。"
        "不传时默认也会自动尝试复制（顺序同上），缺源则改为训练该项。",
    )
    parser.add_argument(
        "--ms_rms_only",
        type=str,
        default=None,
        metavar="KEYS",
        help="仅训练/评估 ms_rms 中的若干变体（逗号分隔），如 full（仅多尺度+RMSNorm 柱）或 baseline,rmsnorm_only；"
        "默认四字全套。pred_len 扫描时建议 full + --ckpt_extra_suffix _plH，避免训齐四柱。",
    )
    parser.add_argument(
        "--dual_reuse_b_only_ckpt",
        action="store_true",
        help="已弃用：等价于 --ms_rms_reuse_rmsnorm_ckpt",
    )
    parser.add_argument(
        "--revin",
        action="store_true",
        help="单次训练开启去噪器 RevIn（默认关闭；与 --revin_rms_ablation 互斥于后者内部设定）",
    )
    parser.add_argument(
        "--no_revin",
        action="store_true",
        help="单次训练关闭 RevIn（默认已关；显式与 --revin 抵触时以本项为准）",
    )
    parser.add_argument(
        "--no_rmsnorm",
        action="store_true",
        help="单次训练改用原版 TransformerEncoder+LayerNorm（关闭 RMSNorm 堆叠）",
    )
    parser.add_argument(
        "--hist_add_bias",
        action="store_true",
        help="关闭 RevIn，在嵌入后使用 HistoryAdditiveBias（历史→仅加性偏置）；与 RMSNorm/LN 可叠加；须重训",
    )
    parser.add_argument(
        "--hist_add_bias_scale",
        type=float,
        default=None,
        metavar="S",
        help="加性偏置强度（默认见 Config.hist_add_bias_scale；不含 B 仅用 A 时）",
    )
    parser.add_argument(
        "--hist_add_bias_scale_with_rmsnorm",
        type=float,
        default=None,
        metavar="S",
        help="A+B(full) 时偏置缩放（默认见 Config.hist_add_bias_scale_with_rmsnorm）",
    )
    parser.add_argument(
        "--result_suffix",
        type=str,
        default=None,
        metavar="TAG",
        help="result/<数据集>/ 下图文件追加 _TAG；默认用启动时间 YYYYMMDD_HHMMSS，避免互相覆盖",
    )
    parser.add_argument(
        "--result_overwrite",
        action="store_true",
        help="毕设图仍用无后缀旧文件名（如 bar_mae_mse_comparison.png），会覆盖同目录已有文件",
    )
    args = parser.parse_args()
    _run_ms_rms = bool(args.ms_rms_ablation or args.dual_ablation)
    if getattr(args, "dual_reuse_b_only_ckpt", False):
        args.ms_rms_reuse_rmsnorm_ckpt = True
    if args.revin_rms_ablation and _run_ms_rms:
        parser.error("--revin_rms_ablation 与 --ms_rms_ablation（含弃用的 --dual_ablation）不能同时使用")

    cfg = Config()
    if getattr(args, "data_path", None) and str(args.data_path).strip():
        cfg.data_path = str(args.data_path).strip()
    if args.hist_add_bias_scale is not None:
        cfg.hist_add_bias_scale = float(args.hist_add_bias_scale)
    if args.hist_add_bias_scale_with_rmsnorm is not None:
        cfg.hist_add_bias_scale_with_rmsnorm = float(args.hist_add_bias_scale_with_rmsnorm)
    if args.all_features:
        cfg.temperature_only = False
    if getattr(args, "multiscale_hist", False):
        cfg.use_multiscale_hist = True
    if getattr(args, "single_scale_hist", False):
        cfg.use_multiscale_hist = False
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if getattr(args, "ms_rms_baseline_epochs", None) is not None:
        cfg.ms_rms_baseline_epochs = max(1, int(args.ms_rms_baseline_epochs))
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.test_batch_size is not None:
        cfg.test_batch_size = max(1, int(args.test_batch_size))
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if getattr(args, "multiscale_steps_per_hour", None) is not None:
        cfg.multiscale_steps_per_hour = max(1, int(args.multiscale_steps_per_hour))
    if args.pred_len is not None:
        cfg.pred_len = max(1, int(args.pred_len))
    if args.ckpt_extra_suffix is not None:
        s = str(args.ckpt_extra_suffix).strip()
        cfg.simdiff_checkpoint_extra_suffix = s if s else None
    # wind / exchange_rate: default suffix isolates checkpoints; multivariate must not share univariate pt.
    if cfg.simdiff_checkpoint_extra_suffix is None:
        dp_stem = cfg.resolved_data_path().stem.lower()
        if dp_stem == "wind":
            cfg.simdiff_checkpoint_extra_suffix = (
                "_wind_mv" if not cfg.temperature_only else "_wind"
            )
        elif dp_stem == "exchange_rate":
            cfg.simdiff_checkpoint_extra_suffix = (
                "_exchange_mv" if not cfg.temperature_only else "_exchange"
            )
    if args.sampling_steps is not None:
        cfg.sampling_steps = args.sampling_steps
    if args.ddim_eta is not None:
        cfg.ddim_eta = args.ddim_eta
    if getattr(args, "ni_inverse_hist_frac", None) is not None:
        cfg.ni_inverse_hist_frac = float(args.ni_inverse_hist_frac)
    if getattr(args, "x0_aux_weight", None) is not None:
        cfg.training_noise_x0_aux_weight = float(args.x0_aux_weight)
    if getattr(args, "val_forecast_mae_every", None) is not None:
        cfg.val_forecast_mae_every = max(0, int(args.val_forecast_mae_every))
    if getattr(args, "val_forecast_mae_max_batches", None) is not None:
        cfg.val_forecast_mae_max_batches = max(1, int(args.val_forecast_mae_max_batches))
    if getattr(args, "val_forecast_mae_fast_samples", None) is not None:
        cfg.val_forecast_mae_num_samples = max(1, int(args.val_forecast_mae_fast_samples))
    if getattr(args, "checkpoint_metric", None) is not None:
        if args.checkpoint_metric == "noise":
            cfg.checkpoint_select_metric = "val_noise"
        else:
            cfg.checkpoint_select_metric = "val_forecast_mae_sparse"
    if getattr(args, "baseline_full_hist", False):
        cfg.baseline_use_full_hist = True
    if args.sampling_mode is not None:
        cfg.sampling_mode = args.sampling_mode
    if args.sample_debug:
        cfg.sample_debug = True
    if args.forecast_num_samples is not None:
        cfg.forecast_num_samples = max(1, int(args.forecast_num_samples))
    if args.mom_groups is not None:
        cfg.mom_num_groups = max(1, int(args.mom_groups))
    if args.ablation is not None:
        cfg.simdiff_ablation = args.ablation
    if args.no_train_amp:
        cfg.train_amp = False
    if args.no_ema:
        cfg.use_ema = False
    if args.thesis_gt_peek is not None:
        cfg.thesis_plot_gt_peek_simdiff = float(args.thesis_gt_peek)
    if getattr(args, "thesis_gt_peek_no_title_hint", False):
        cfg.thesis_gt_peek_hide_title_hint = True
    if getattr(args, "thesis_overlay_test_batch", None) is not None:
        cfg.thesis_overlay_test_batch = max(0, int(args.thesis_overlay_test_batch))
    if getattr(args, "forecast_point", None) is not None:
        cfg.forecast_point_mode = str(args.forecast_point).strip().lower()
    if getattr(args, "forecast_primary_loss_weight", None) is not None:
        cfg.forecast_loss_primary_weight = float(args.forecast_primary_loss_weight)
    if args.hist_add_bias:
        cfg.use_hist_add_bias = True
        cfg.use_revin = False
    if not args.revin_rms_ablation:
        if args.no_revin:
            cfg.use_revin = False
        elif getattr(args, "revin", False):
            cfg.use_revin = True
        if args.no_rmsnorm:
            cfg.use_rmsnorm = False
    if args.revin_rms_ablation:
        if cfg.use_hist_add_bias:
            print(
                "[note] --revin_rms_ablation 仅四类 RevIn/RMSNorm；已忽略 use_hist_add_bias（False）。"
            )
        cfg.use_hist_add_bias = False
    if _run_ms_rms:
        if cfg.use_hist_add_bias:
            print("[note] --ms_rms_ablation 不使用 HistoryAdditiveBias；已忽略 --hist_add_bias")
        cfg.use_hist_add_bias = False
    if args.dual_ablation:
        print(
            "[warn] --dual_ablation 已弃用，请改用 --ms_rms_ablation（现为多尺度+RMSNorm 四组消融）"
        )

    cfg.validate_mom_config()
    cfg.validate_training_noise_objective()
    cfg.validate_ni_inverse_options()
    cfg.validate_thesis_plot_options()
    cfg.validate_forecast_point_and_loss_weight()
    cfg.validate_training_checkpoint_options()
    cfg.validate_simdiff_ablation()
    cfg.validate_denoiser_embedding_options()
    if getattr(args, "ms_rms_only", None):
        try:
            _parse_ms_rms_only_arg(args.ms_rms_only)
        except ValueError as e:
            parser.error(str(e))
    if args.all_plots:
        cfg.thesis_result_only = False

    if args.result_overwrite:
        cfg.result_name_suffix = None
    elif args.result_suffix is not None:
        cfg.result_name_suffix = args.result_suffix.strip() or datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    elif cfg.result_name_suffix is None:
        cfg.result_name_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    set_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    if args.verify_norm_mom:
        train_loader, val_loader, test_loader, n_features, feat_names = make_loaders(cfg)
        import verify_norm_mom as vnm

        vnm.run_quick_verify(cfg, train_loader, device)
        return

    if _run_ms_rms:
        print("Device:", device)
        if cfg.result_name_suffix:
            print(f"result 图后缀: _{cfg.result_name_suffix}（写入 {cfg.resolved_result_dir()}）")
        else:
            print(f"result 图: 无后缀（--result_overwrite）-> {cfg.resolved_result_dir()}")
        print(
            "模式: 多尺度+RMSNorm 四组消融（simdiff_weather_best_ms_rms_<baseline|rmsnorm_only|multiscale_only|full>.pt）；"
            f"epochs={cfg.epochs}；--eval_only 则仅加载四权重评估）"
        )
        run_ms_rms_ablation_suite(cfg, args, device)
        return

    train_loader, val_loader, test_loader, n_features, feat_names = make_loaders(cfg)
    t_idx = resolve_temperature_feature_index(feat_names)
    cfg.forecast_loss_primary_channel_idx = t_idx if n_features > 1 else None

    if args.revin_rms_ablation:
        print("Device:", device)
        if cfg.result_name_suffix:
            print(f"result 图后缀: _{cfg.result_name_suffix}（写入 {cfg.resolved_result_dir()}）")
        else:
            print(f"result 图: 无后缀（--result_overwrite）-> {cfg.resolved_result_dir()}")
        print(
            "模式: RevIn/RMSNorm 四组消融（各存 simdiff_weather_best_<variant>.pt；"
            f"epochs={cfg.epochs}；--eval_only 则仅加载已有四权重评估）"
        )
        run_revin_rms_ablation_suite(
            cfg,
            args,
            train_loader,
            val_loader,
            test_loader,
            device,
            n_features,
            feat_names,
        )
        return

    print("Device:", device)
    if cfg.result_name_suffix:
        print(f"result 图后缀: _{cfg.result_name_suffix}（目录 {cfg.resolved_result_dir()}）")
    else:
        print(f"result 图: 无后缀（--result_overwrite）-> {cfg.resolved_result_dir()}")
    if cfg.thesis_result_only:
        print("输出模式: 仅 result/<数据集>/ 图表 + 终端指标表（不写 plots/；需完整对比图请加 --all_plots）")
    print(
        f"SimDiff 消融: {cfg.simdiff_ablation} | "
        f"checkpoint: {cfg.simdiff_checkpoint_filename()}"
    )
    if cfg.simdiff_ablation == "ni_only":
        print(
            "  （ni_only：与 full 共用同一套权重；评估时为 K 次采样算术平均，无 MoM 中位数）"
        )
    elif cfg.simdiff_ablation == "mom_only":
        print(
            "  （mom_only：训练/推理均为「未来用历史窗统计量」归一化；请使用对应 mom_only 权重）"
        )
    if (
        cfg.sampling_mode.lower() == "ddim"
        and cfg.sampling_steps is not None
        and cfg.sampling_steps > cfg.timesteps
    ):
        print(
            f"[warn] sampling_steps={cfg.sampling_steps} > timesteps={cfg.timesteps}："
            f"DDIM 将截断为 {cfg.timesteps}，多余子步在离散日程上无对应，曾导致采样发散。"
        )
    if cfg.simdiff_ablation == "ni_only":
        _sk = f"K={cfg.forecast_num_samples}（ni_only 评估用算术均值，无 MoM）"
    else:
        _sk = f"MoM: K={cfg.forecast_num_samples}, M={cfg.mom_num_groups}"
    print(
        f"采样: mode={cfg.sampling_mode}, steps={cfg.sampling_steps or cfg.timesteps}, "
        f"ddim_eta={cfg.ddim_eta}, {_sk}"
    )
    if cfg.simdiff_ablation != "ni_only":
        _fpm = str(getattr(cfg, "forecast_point_mode", "mom"))
        print(
            f"点预测(mode): {_fpm}（ni_only 固定 K 均值；"
            "`--forecast_point mean|single` 可减平滑、一般不须重训）"
        )
    _pw_l = float(getattr(cfg, "forecast_loss_primary_weight", 1.0))
    if _pw_l > 1.0 + 1e-8 and int(n_features) > 1:
        print(
            f"训练：主变量通道损失加权 ×{_pw_l} "
            f"（idx={cfg.forecast_loss_primary_channel_idx}）**须重训**"
        )
    a_h = float(cfg.training_noise_mse_huber_alpha)
    h_note = "纯 MSE" if a_h >= 0.999 else f"αMSE+({1.0 - a_h:.2f})smooth_l1(β={cfg.training_noise_huber_beta})"
    print(
        f"训练噪声主项: {h_note} + L1×{cfg.training_noise_l1_weight} + "
        f"时间差分×{cfg.training_noise_temporal_diff_weight}"
    )
    _vfe = int(getattr(cfg, "val_forecast_mae_every", 0))
    _csm = str(getattr(cfg, "checkpoint_select_metric", "val_noise"))
    if _vfe > 0:
        print(
            f"[checkpoint] 依据 {_csm}；每 {_vfe} epoch 稀疏验证预报 MAE（≤{cfg.val_forecast_mae_max_batches} val batches）"
        )
    else:
        print(f"[checkpoint] 依据 {_csm}（稀疏预报 MAE 已关闭，零额外采样开销）")
    _x0 = float(getattr(cfg, "training_noise_x0_aux_weight", 0.0))
    if _x0 > 0.0:
        print(f"  x0 辅助项: + MSE(x̂0,x0)×{_x0}")
    _niw = float(getattr(cfg, "ni_inverse_hist_frac", 0.0))
    if _niw > 0.0:
        print(
            f"  [note] ni_inverse_hist_frac={_niw}：反变换非纯正 NI；论文对比与可复现 MAE 请用 --ni_inverse_hist_frac 0"
        )
    print(
        f"MoM 低温向权重: mom_cold_bias_blend={cfg.mom_cold_bias_blend}, "
        f"mom_cold_sharpness={cfg.mom_cold_sharpness}（归一化空间；盯全格点 MAE/CRPS）"
    )
    print(
        f"训练: train_amp={cfg.train_amp}（CUDA 上生效）| "
        f"use_ema={cfg.use_ema}（decay={cfg.ema_decay}，checkpoint 存 EMA）"
    )
    mode = "仅气温单变量" if cfg.temperature_only else "全部气象变量"
    _sph = getattr(cfg, "multiscale_steps_per_hour", None)
    _spd = getattr(cfg, "multiscale_steps_per_day", None)
    _hwm = getattr(cfg, "hist_window_start_min", 0)
    _extra_ms = ""
    if cfg.use_multiscale_hist:
        _extra_ms = f", multiscale_steps_per_hour={_sph}, multiscale_steps_per_day={_spd}, hist_window_start_min={_hwm}"
    print(
        f"特征维度 C={n_features}（{mode}）, 列: {feat_names} | "
        f"RevIn={cfg.use_revin}, RMSNorm={cfg.use_rmsnorm}, multiscale_hist={cfg.use_multiscale_hist}"
        f"{_extra_ms}"
    )
    print_wind_p0_training_hints(cfg, n_features)
    print_exchange_p0_training_hints(cfg, n_features)
    print(
        f"Normalization Independence: 历史/未来分算 μ,σ；"
        f"无真值反变换用训练集未来边际 std范围 "
        f"[{np.asarray(cfg.train_future_marginal_std).min():.4g}, "
        f"{np.asarray(cfg.train_future_marginal_std).max():.4g}]"
    )
    if cfg.use_global_standardization and cfg.global_std is not None:
        gs = np.asarray(cfg.global_std)
        print(
            f"（可选）全局统计 std 范围 [{gs.min():.4g}, {gs.max():.4g}]（SimDiff 主干未使用）"
        )
    zs_cfg = getattr(cfg, "train_metric_z_sigma", None)
    if zs_cfg is not None:
        zp = np.asarray(zs_cfg).ravel()
        zm_cfg = getattr(cfg, "train_metric_z_mu", None)
        zp_mu = np.asarray(zm_cfg).ravel() if zm_cfg is not None else zp * 0.0
        _zi_print = resolve_temperature_feature_index(feat_names)
        if len(zp) > 0:
            pis = min(_zi_print, len(zp) - 1)
            print(
                f"[metric] Train-split z-score (paper-style REPORTING only): "
                f"{feat_names[pis]}  μ≈{float(zp_mu[pis]):.6g}, σ≈{float(zp[pis]):.6g}; "
                f"MAE_z=MAE/σ，MSE_z=MSE/σ²。扩散仍在 NI 原始空间训练，未改用 bug/3 的 hist-归一 future。"
            )

    model = SimDiffWeather(cfg).to(device)
    ckpt_path = cfg.resolved_checkpoint_dir() / cfg.simdiff_checkpoint_filename()
    trainer_ref: Trainer | None = None

    if args.eval_only:
        print("\n" + "=" * 60)
        print("【仅评估模式】未执行训练。若从未成功训练，权重为随机或陈旧，指标不可信。")
        print("=" * 60 + "\n")
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"未找到 {ckpt_path}。请先训练："
                f" python main.py --ablation {cfg.simdiff_ablation} （不要加 --eval_only）"
            )
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        if "meta" in state:
            print(
                f"已加载 checkpoint（epoch 记录: {state.get('epoch_trained', '未知')}）"
            )
        buf_m = model._fut_mu_marginal.squeeze().detach().cpu().numpy()
        cfg_m = np.asarray(cfg.train_future_marginal_mean, dtype=np.float64)
        if buf_m.shape == cfg_m.shape:
            d = float(np.abs(buf_m - cfg_m).max())
            print(f"反归一化检查: checkpoint 与当前训练集未来边际 mean 最大差 = {d:.6g}")
            if d > 1e-2:
                print(
                    "  [warn] 数据划分或数据文件变更后请重新训练，否则边际尺度不一致。"
                )
        model.eval()
    else:
        trainer_ref = Trainer(
            cfg,
            model,
            train_loader,
            val_loader,
            device,
            primary_forecast_channel=t_idx,
        )
        trainer_ref.fit()

    # 测试与画图
    model.eval()
    if (
        not cfg.thesis_result_only
        and trainer_ref is not None
        and not args.skip_training_curves
        and len(trainer_ref.history_train) > 0
    ):
        p_curve = cfg.resolved_plot_dir() / "training_convergence.png"
        plot_training_curves(
            p_curve,
            trainer_ref.history_train,
            trainer_ref.history_val,
            title="Training convergence (noise prediction loss)",
        )
        print(f"Saved plot: {p_curve}")
    noise_losses = []
    with torch.no_grad():
        for hist, fut in test_loader:
            hist = hist.to(device)
            fut = fut.to(device)
            loss = model.training_loss(hist, fut)
            if torch.isfinite(loss):
                noise_losses.append(loss.item())
    if noise_losses:
        print(f"Test noise MSE (训练目标，越小越好): {np.mean(noise_losses):.6f}")
    else:
        print("Test noise MSE: 无有效 batch")

    _tn_raw = feat_names[t_idx] if t_idx < len(feat_names) else str(t_idx)
    temp_name = " ".join(str(_tn_raw).split())
    if not cfg.thesis_result_only:
        with torch.no_grad():
            hist, fut = next(iter(test_loader))
            hist = hist.to(device)
            fut = fut.to(device)
            out_s = model.forecast(hist, future=fut)
            pred_m = point_prediction_from_forecast(out_s, cfg)
            mse_s = torch.mean((pred_m - fut) ** 2)
            mae_s = torch.mean(torch.abs(pred_m - fut))
        if cfg.simdiff_ablation == "ni_only":
            _hdr = (
                f"采样评估（单 batch，NI + K 次算术均值，无 MoM；K={cfg.forecast_num_samples}）"
            )
        elif cfg.simdiff_ablation == "mom_only":
            _hdr = (
                f"采样评估（单 batch，hist 归一化 + MoM：K={cfg.forecast_num_samples}, "
                f"M={cfg.mom_num_groups}）"
            )
        else:
            _hdr = (
                f"采样评估（单 batch，MoM：K={cfg.forecast_num_samples}, M={cfg.mom_num_groups}）"
            )
        _lines = [
            _hdr,
            f"  主变量 [{temp_name}] MSE: {torch.mean((pred_m[..., t_idx] - fut[..., t_idx]) ** 2).item():.6f} | "
            f"MAE: {torch.mean(torch.abs(pred_m[..., t_idx] - fut[..., t_idx])).item():.6f}",
        ]
        if n_features > 1:
            _lines.append(
                f"  全特征平均 MSE: {mse_s.item():.6f} | MAE: {mae_s.item():.6f}（辅助）"
            )
        print("\n".join(_lines))
    (
        mse_test,
        mae_test,
        mae_ch,
        mse_ch,
        crps_mean_fulltest,
        var_mean_fulltest,
        crps_h_fulltest,
        mae_ch_z,
        mse_ch_z,
        crps_mean_z_fulltest,
        var_mean_z_fulltest,
    ) = evaluate_test_loader_prob_combined(
        model,
        test_loader,
        device,
        n_features,
        t_idx,
        cfg.pred_len,
        cfg,
        progress_desc="测试集 SimDiff (K-sample)",
    )
    if not cfg.thesis_result_only:
        print(
            f"\n【主结论 · 温度】{temp_name} — 全测试集 MSE: {mse_ch[t_idx]:.6f} | MAE: {mae_ch[t_idx]:.6f}"
        )
        if n_features > 1:
            print(f"【辅助】全特征平均 MSE: {mse_test:.6f} | MAE: {mae_test:.6f}")
    else:
        print(
            f"\n（{temp_name} 全测试集 MSE/MAE/CRPS/VAR 见下方毕设指标表，此处不重复打印。）"
        )
        if n_features > 1:
            print(
                f"【辅助】全特征平均 MSE: {mse_test:.6f} | MAE: {mae_test:.6f}"
            )

    if not cfg.thesis_result_only and not args.skip_prob_metrics:
        crps_mean = crps_mean_fulltest
        crps_h = crps_h_fulltest
        print(
            f"\n【概率预测 · {temp_name}】全测试集平均 CRPS: {crps_mean:.6f} "
            f"(K={cfg.forecast_num_samples} 次样本近似预报分布)"
        )
        p_crps = cfg.resolved_plot_dir() / "crps_by_horizon.png"
        plot_crps_by_horizon(
            p_crps,
            crps_h,
            title=f"CRPS by horizon — {temp_name}",
        )
        print(f"Saved plot: {p_crps}")

        hist_pi, fut_pi = next(iter(test_loader))
        hist_pi = hist_pi.to(device)
        fut_pi = fut_pi.to(device)
        with torch.no_grad():
            out_pi = model.forecast(hist_pi[:1], future=fut_pi[:1], return_samples=True)
        if out_pi.samples is not None:
            samp = out_pi.samples[0].cpu().numpy()
            true_1 = fut_pi[0].cpu().numpy()
            hist_1 = hist_pi[0].cpu().numpy()
            pt = point_prediction_from_forecast(out_pi, cfg)[0].cpu().numpy()
            cov90 = empirical_interval_coverage(
                samp[..., t_idx] if n_features > 1 else samp.squeeze(-1),
                true_1[:, t_idx] if n_features > 1 else true_1.squeeze(-1),
                q_low=0.05,
                q_high=0.95,
            )
            p_int = cfg.resolved_plot_dir() / "forecast_predictive_intervals.png"
            plot_forecast_predictive_intervals(
                p_int,
                hist_1,
                true_1,
                samp,
                t_idx,
                ylabel=feat_names[t_idx] if t_idx < len(feat_names) else str(t_idx),
                title=f"90% predictive interval (first test batch) — {temp_name}",
                q_low=0.05,
                q_high=0.95,
                point_pred=pt,
                point_label=simdiff_plot_name(cfg),
            )
            print(f"Saved plot: {p_int}")
            print(
                f"  （示例窗）名义90% 区间逐时刻经验覆盖率: {cov90 * 100:.1f}% "
                f"（单窗参考，非全测试集）"
            )

    plot_dir = cfg.resolved_plot_dir()
    c_vis = t_idx if t_idx < n_features else min(1, n_features - 1)
    ylabel = feat_names[c_vis] if c_vis < len(feat_names) else str(c_vis)

    if not cfg.thesis_result_only and not args.skip_denoise_traj:
        hd0, fd0 = next(iter(test_loader))
        hd0 = hd0[:1].to(device)
        fd0 = fd0[:1].to(device)
        with torch.no_grad():
            traj_frames = model.get_denoise_trajectory_physical(
                hd0, fd0, cfg.denoise_trajectory_max_points
            )
        p_den = plot_dir / "diffusion_denoise_trajectory.png"
        plot_denoise_trajectory_heatmap(
            p_den,
            traj_frames,
            t_idx,
            ylabel,
            title=f"Denoising trajectory (test batch[0]) — {temp_name} | "
            f"{cfg.sampling_mode.upper()}",
        )
        print(f"Saved plot: {p_den}")

    _sdn = simdiff_plot_name(cfg)
    if not cfg.thesis_result_only:
        horizon_maes: dict[str, np.ndarray] = {
            _sdn: eval_horizon_mae(
                lambda h, f: point_prediction_from_forecast(
                    model.forecast(h, future=f), model.cfg
                ),
                test_loader,
                device,
                cfg.pred_len,
                t_idx,
                n_features,
            )
        }
    else:
        horizon_maes = {}
    bar_names: list[str] = [_sdn]
    bar_maes: list[float] = [float(mae_test)]
    bar_mses: list[float] = [float(mse_test)]

    itrans_m: BaselineHistTrim | BaselineiTransformer | None = None
    tmixer_m: BaselineHistTrim | BaselineTimeMixer | None = None
    residual_multi_for_kde: dict[str, np.ndarray] | None = None

    run_learned_baselines = not args.skip_baselines

    if run_learned_baselines and cfg.use_multiscale_hist:
        baseline_full = bool(getattr(cfg, "baseline_use_full_hist", False))
        if baseline_full:
            print(
                f"[note] --baseline_full_hist：iTransformer/TimeMixer 使用与 SimDiff 相同的历史长度 "
                f"{cfg.effective_hist_len()}（不经 HistTrim）。"
            )
        else:
            print(
                f"[note] 多尺度历史下 iTransformer/TimeMixer 仅使用前 {cfg.seq_len} 步原始段；"
                f"SimDiff 仍用全量 {cfg.effective_hist_len()} 步。"
            )

    if run_learned_baselines:
        itrans_epochs = cfg.epochs
        print(
            "\n--- 学习型基线 iTransformer / TimeMixer（val MSE 早停；"
            f"iTransformer max_epochs={itrans_epochs}（与 SimDiff）；"
            f"TimeMixer max_epochs={cfg.baseline_timemixer_max_epochs}）---"
        )
        pl = cfg.pred_len
        C = n_features
        baseline_full = bool(getattr(cfg, "baseline_use_full_hist", False))
        bl_hist_len = (
            cfg.effective_hist_len()
            if (cfg.use_multiscale_hist and baseline_full)
            else cfg.seq_len
        )
        use_hist_trim = cfg.use_multiscale_hist and not baseline_full
        amp_on = bool(cfg.forecast_amp)

        itrans_core = BaselineiTransformer(
            bl_hist_len,
            pl,
            C,
            d_model=cfg.baseline_itransformer_d_model,
            nhead=cfg.baseline_itransformer_nhead,
            num_layers=cfg.baseline_itransformer_layers,
            dropout=cfg.dropout,
        )
        itrans_m = (
            BaselineHistTrim(itrans_core, cfg.seq_len) if use_hist_trim else itrans_core
        )
        itrans_m = fit_regression_model(
            itrans_m,
            train_loader,
            val_loader,
            device,
            max_epochs=itrans_epochs,
            lr=cfg.baseline_lr,
            patience=cfg.baseline_early_stop_patience,
            grad_clip_max_norm=cfg.baseline_grad_clip_max_norm,
            name=_ITRANS_NAME,
        )

        def wrap_itr(h):
            with forecast_amp_context(device, amp_on):
                return itrans_m(h)

        mse_b, mae_b = eval_forecasts_mse_mae(wrap_itr, test_loader, device)
        mt_b, at_b = eval_channel_mse_mae(wrap_itr, test_loader, device, t_idx)
        if not cfg.thesis_result_only:
            print_baseline_block(_ITRANS_NAME, mse_b, mae_b, mt_b, at_b, temp_name, n_features)
        if not cfg.thesis_result_only:
            horizon_maes[_ITRANS_NAME] = eval_horizon_mae(
                lambda h, f: wrap_itr(h),
                test_loader,
                device,
                cfg.pred_len,
                t_idx,
                n_features,
            )
        bar_names.append(_ITRANS_NAME)
        bar_maes.append(float(mae_b))
        bar_mses.append(float(mse_b))

        tmixer_core = BaselineTimeMixer(
            bl_hist_len,
            pl,
            C,
            d_model=cfg.baseline_timemixer_d_model,
            n_scales=cfg.baseline_timemixer_scales,
            dropout=cfg.dropout,
        )
        tmixer_m = (
            BaselineHistTrim(tmixer_core, cfg.seq_len) if use_hist_trim else tmixer_core
        )
        tm_lr = (
            float(cfg.baseline_timemixer_lr)
            if cfg.baseline_timemixer_lr is not None
            else cfg.baseline_lr
        )
        tmixer_m = fit_regression_model(
            tmixer_m,
            train_loader,
            val_loader,
            device,
            max_epochs=cfg.baseline_timemixer_max_epochs,
            lr=tm_lr,
            patience=cfg.baseline_early_stop_patience,
            grad_clip_max_norm=cfg.baseline_grad_clip_max_norm,
            name="TimeMixer",
        )

        def wrap_tm(h):
            with forecast_amp_context(device, amp_on):
                return tmixer_m(h)

        mse_b, mae_b = eval_forecasts_mse_mae(wrap_tm, test_loader, device)
        mt_b, at_b = eval_channel_mse_mae(wrap_tm, test_loader, device, t_idx)
        if not cfg.thesis_result_only:
            print_baseline_block("TimeMixer", mse_b, mae_b, mt_b, at_b, temp_name, n_features)
        if not cfg.thesis_result_only:
            horizon_maes["TimeMixer"] = eval_horizon_mae(
                lambda h, f: wrap_tm(h),
                test_loader,
                device,
                cfg.pred_len,
                t_idx,
                n_features,
            )
        bar_names.append("TimeMixer")
        bar_maes.append(float(mae_b))
        bar_mses.append(float(mse_b))

        if not cfg.thesis_result_only:
            r_sd, _ = collect_test_forecast_errors(model, test_loader, device, t_idx)
            r_itr = collect_channel_residuals(wrap_itr, test_loader, device, t_idx)
            r_tm = collect_channel_residuals(wrap_tm, test_loader, device, t_idx)
            residual_multi_for_kde = {_sdn: r_sd, _ITRANS_NAME: r_itr, "TimeMixer": r_tm}

            yt, yp = collect_pooled_predictions(
                lambda h, f: point_prediction_from_forecast(
                    model.forecast(h, future=f), model.cfg
                ),
                test_loader,
                device,
                t_idx,
            )
            p_sc = plot_dir / "scatter_simdiff_pred_vs_true.png"
            plot_pred_vs_true_scatter(
                p_sc,
                yt,
                yp,
                title=f"{_sdn}: predicted vs true ({temp_name})",
            )
            print(f"Saved plot: {p_sc}")

            hist_b, fut_b = next(iter(test_loader))
            hist_b = hist_b.to(device)
            fut_b = fut_b.to(device)
            with torch.no_grad():
                with forecast_amp_context(device, amp_on):
                    itr_b = itrans_m(hist_b)
                    tm_b = tmixer_m(hist_b)
                preds_np: dict[str, np.ndarray] = {
                    _sdn: point_prediction_from_forecast(
                        model.forecast(hist_b, future=fut_b), model.cfg
                    ).cpu().numpy(),
                    _ITRANS_NAME: itr_b.cpu().numpy(),
                    "TimeMixer": tm_b.cpu().numpy(),
                }
            hist0 = hist_b[0].cpu().numpy()
            true0 = fut_b[0].cpu().numpy()
            pred_dict = {k: v[0] for k, v in preds_np.items()}
            p_cmp = plot_dir / "forecast_compare_example.png"
            plot_forecast_compare(
                p_cmp,
                hist0,
                true0,
                pred_dict,
                ylabel=ylabel,
                title=f"Forecast comparison (first test batch sample) — {temp_name}",
                channel=t_idx,
                hist_anchor_index=thesis_overlay_hist_anchor_index(cfg),
            )
            print(f"Saved plot: {p_cmp}")

            n_ex = min(4, hist_b.shape[0])
            grid_ex = []
            for i in range(n_ex):
                grid_ex.append(
                    {
                        "hist": hist_b[i].cpu().numpy(),
                        "true": fut_b[i].cpu().numpy(),
                        "preds": {k: preds_np[k][i] for k in preds_np},
                    }
                )
            p_grid = plot_dir / "forecast_compare_grid.png"
            plot_forecast_grid(
                p_grid,
                grid_ex,
                cfg.pred_len,
                ylabel=ylabel,
                title=f"Forecast comparison — {temp_name}",
            )
            print(f"Saved plot: {p_grid}")

    if not cfg.thesis_result_only and not args.skip_error_kde:
        res_e, ae_e = collect_test_forecast_errors(
            model, test_loader, device, t_idx
        )
        p_kde = plot_dir / "error_distribution_kde.png"
        plot_error_kde(
            p_kde,
            res_e,
            ae_e,
            unit_label=feat_names[t_idx] if t_idx < len(feat_names) else temp_name,
            title=f"Test error KDE — {temp_name} ({simdiff_plot_name(cfg)})",
            residual_multi=residual_multi_for_kde,
        )
        print(f"Saved plot: {p_kde}")
        if residual_multi_for_kde is not None:
            p3 = plot_dir / "residual_kde_three_models.png"
            plot_residual_kde_multi(
                p3,
                residual_multi_for_kde,
                unit_label=feat_names[t_idx] if t_idx < len(feat_names) else temp_name,
                title=f"Residual KDE: SimDiff vs {_ITRANS_NAME} vs TimeMixer — {temp_name}",
            )
            print(f"Saved plot: {p3}")
            print("  （左侧子图为多模型时，error_distribution_kde.png 左栏标题为 Residual (multi-model)）")
        else:
            print(
                "  [hint] 未跑学习型基线：KDE 左栏仅 SimDiff。"
                f"需要与 {_ITRANS_NAME}/TimeMixer 对比请不要加 --skip_baselines，并重新运行完整 main.py。"
            )

    if not cfg.thesis_result_only:
        p_metrics = plot_dir / "metrics_comparison_mae_mse.png"
        plot_metrics_bars(
            p_metrics,
            bar_names,
            bar_maes,
            bar_mses,
            title=f"Test metrics — {temp_name}",
        )
        print(f"Saved plot: {p_metrics}")

        p_h = plot_dir / "mae_by_horizon.png"
        plot_horizon_mae(
            p_h,
            horizon_maes,
            cfg.pred_len,
            title=f"MAE by horizon — {temp_name}",
        )
        print(f"Saved plot: {p_h}")

        hist, fut = next(iter(test_loader))
        hist = hist.to(device)
        fut = fut.to(device)
        with torch.no_grad():
            fo = model.forecast(hist[:1], future=fut[:1])
        hist0 = hist[0].cpu().numpy()
        true0 = fut[0].cpu().numpy()
        pred0 = point_prediction_from_forecast(fo, cfg)[0].cpu().numpy()

        _lh_ex = int(hist0.shape[0])
        _lf_ex = int(true0.shape[0])
        _th_ex = np.arange(_lh_ex)
        _tf_ex = np.arange(_lh_ex, _lh_ex + _lf_ex)
        plt.figure(figsize=(10, 4))
        plt.plot(_th_ex, hist0[:, c_vis], label="history", color="C0")
        plt.plot(_tf_ex, true0[:, c_vis], label="ground truth", color="black")
        plt.plot(_tf_ex, pred0[:, c_vis], label=_sdn, color="C2", linestyle="--")
        plt.axvline(int(cfg.effective_hist_len()) - 0.5, color="gray", linestyle=":")
        plt.xlabel("time step (index)")
        plt.ylabel(ylabel)
        plt.title("SimDiff-Weather: forecast (single model)")
        plt.legend()
        plt.tight_layout()
        plot_path = plot_dir / "forecast_example.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_path}")

    # -------- 毕设专用：result/<数据集名>/ 或 --figures_dir；与 plots/ 隔离；主变量指标与曲线 --------
    rdir = resolved_thesis_plot_dir(cfg, args)
    ch = t_idx
    slug = " ".join(cfg.result_dataset_slug().split())
    crps_mean_t = crps_mean_fulltest
    var_mean_t = var_mean_fulltest

    def _wrap_itr_thesis(h: torch.Tensor) -> torch.Tensor:
        with forecast_amp_context(device, bool(cfg.forecast_amp)):
            return itrans_m(h)  # type: ignore[misc]

    def _wrap_tm_thesis(h: torch.Tensor) -> torch.Tensor:
        with forecast_amp_context(device, bool(cfg.forecast_amp)):
            return tmixer_m(h)  # type: ignore[misc]

    _mae_phys_sd = float(mae_test)
    _mse_phys_sd = float(mse_test)
    _mae_z_mean = float(np.mean(mae_ch_z))
    _mse_z_mean = float(np.mean(mse_ch_z))

    table_rows: list[tuple[str, float, float, str, str]] = [
        (
            _sdn,
            _mae_phys_sd,
            _mse_phys_sd,
            f"{crps_mean_t:.6f}",
            f"{var_mean_t:.6f}",
        ),
    ]
    bar_n = [_sdn]
    bar_mae_l = [_mae_phys_sd]
    bar_mse_l = [_mse_phys_sd]
    if itrans_m is not None and tmixer_m is not None:
        mse_itr_all, mae_itr_all = eval_forecasts_mse_mae(
            _wrap_itr_thesis, test_loader, device
        )
        mse_tm_all, mae_tm_all = eval_forecasts_mse_mae(
            _wrap_tm_thesis, test_loader, device
        )
        # Degenerate ensemble CRPS == MAE (全通道平均); VAR == 0
        table_rows.append(
            (_ITRANS_NAME, mae_itr_all, mse_itr_all, f"{mae_itr_all:.6f}", f"{0.0:.6f}")
        )
        table_rows.append(
            ("TimeMixer", mae_tm_all, mse_tm_all, f"{mae_tm_all:.6f}", f"{0.0:.6f}")
        )
        bar_n.extend([_ITRANS_NAME, "TimeMixer"])
        bar_mae_l.extend([mae_itr_all, mae_tm_all])
        bar_mse_l.extend([mse_itr_all, mse_tm_all])

    # ---- Paper-style primary table: train z-score (same ŷ/y; reporting only). ----
    _zs_flat = np.asarray(cfg.train_metric_z_sigma, dtype=np.float64).reshape(-1)
    sig_primary = float(_zs_flat[min(int(ch), max(0, _zs_flat.size - 1))])
    sig_mean_train = float(np.mean(_zs_flat))
    crps_z_s = (
        f"{crps_mean_z_fulltest:.6f}"
        if np.isfinite(crps_mean_z_fulltest)
        else "nan"
    )
    var_z_s = (
        f"{var_mean_z_fulltest:.6f}"
        if np.isfinite(var_mean_z_fulltest)
        else "nan"
    )
    table_rows_z: list[tuple[str, float, float, str, str]] = [
        (
            _sdn,
            _mae_z_mean,
            _mse_z_mean,
            crps_z_s,
            var_z_s,
        )
    ]
    bar_n_z = [_sdn]
    bar_mae_l_z = [_mae_z_mean]
    bar_mse_l_z = [_mse_z_mean]
    if itrans_m is not None and tmixer_m is not None:
        mse_itr_z, mae_itr_z = eval_forecasts_mse_mae_train_zscore(
            _wrap_itr_thesis,
            test_loader,
            device,
            _zs_flat,
        )
        mse_tm_z, mae_tm_z = eval_forecasts_mse_mae_train_zscore(
            _wrap_tm_thesis,
            test_loader,
            device,
            _zs_flat,
        )
        table_rows_z.append(
            (_ITRANS_NAME, mae_itr_z, mse_itr_z, f"{mae_itr_z:.6f}", f"{0.0:.6f}")
        )
        table_rows_z.append(
            ("TimeMixer", mae_tm_z, mse_tm_z, f"{mae_tm_z:.6f}", f"{0.0:.6f}")
        )
        bar_n_z.extend([_ITRANS_NAME, "TimeMixer"])
        bar_mae_l_z.extend([mae_itr_z, mae_tm_z])
        bar_mse_l_z.extend([mse_itr_z, mse_tm_z])
    _thesis_metrics_caption = (
        f"{slug} | all {n_features} ch mean (MAE/MSE/CRPS/VAR) | overlay {_matplotlib_safe_text(temp_name, ascii_fallback=f'channel {ch}')} only | "
        "paper-style MAE_z / MSE_z (train σ per channel)"
        if n_features > 1
        else (
            f"{slug} | {_matplotlib_safe_text(temp_name, ascii_fallback=f'channel {ch}')} | "
            "paper-style MAE_z / MSE_z (train σ)"
        )
    )
    print_thesis_metrics_table(
        table_rows_z,
        _thesis_metrics_caption,
        english=True,
        footer_notes=(
            "MAE_z, MSE_z: mean over all channels of per-point (ŷ−y)/σ_train and ((ŷ−y)/σ_train)² on full test.",
            "SimDiff CRPS/VAR (and _z): averaged over channels (each channel z-scored for CRPS_z/VAR_z). σ from train split.",
            "Point baselines: CRPS_z == MAE_z (all-ch mean); VAR_z == 0. Overlay figures plot primary channel only.",
        ),
    )

    print_metrics_ascii_table(
        table_rows,
        headline=(
            f"METRICS — physical scale (original units) | {slug} | "
            + (
                f"all {n_features} ch mean (overlay still {_matplotlib_safe_text(temp_name, ascii_fallback='OT')})"
                if n_features > 1
                else _matplotlib_safe_text(temp_name, ascii_fallback=f"channel {ch}")
            )
        ),
        footer_notes=(
            "MAE / MSE / CRPS / VAR: averaged over all channels and all test (batch, horizon) unless C=1.",
            "Forecast_curves_overlay still uses the primary target channel (e.g. OT) only.",
        ),
    )

    p_bar_r = rdir / cfg.result_png_basename("bar_mae_mse_comparison")
    plot_metrics_bars(
        p_bar_r,
        bar_n_z,
        bar_mae_l_z,
        bar_mse_l_z,
        title=(
            f"[{slug}] Test MAE_z / MSE_z (all {n_features} ch mean)"
            + (
                f"; mean σ_train={sig_mean_train:.4g}"
                if n_features > 1
                else f"; σ_train={sig_primary:.4g}"
            )
        ),
        ylabel="MAE_z / MSE_z",
        title_fontsize=9.5,
    )
    print(f"[毕设] {p_bar_r} (paper-style z; same as banner table MAE/MSE columns)")

    p_bar_phys = rdir / cfg.result_png_basename("bar_mae_mse_physical")
    plot_metrics_bars(
        p_bar_phys,
        bar_n,
        bar_mae_l,
        bar_mse_l,
        title=(
            f"[{slug}] Test MAE / MSE — physical (all {n_features} ch mean)"
            if n_features > 1
            else f"[{slug}] Test MAE / MSE — physical ({_matplotlib_safe_text(temp_name, ascii_fallback='primary channel')})"
        ),
        ylabel="MAE / MSE",
        title_fontsize=9.5,
    )
    print(f"[毕设] {p_bar_phys} (physical units)")
    print(
        "[毕设] 柱状图 MAE/MSE 与横幅表：多变量时为全通道平均；forecast overlay 仍仅画主变量通道；"
        "个案窗口可陡，请以表内指标为准。"
    )

    overlay_batches = parse_thesis_overlay_batch_indices(args, cfg)
    _raw_ov = getattr(args, "thesis_overlay_batches", None)
    overlay_stem_uses_batch = _raw_ov is not None and bool(str(_raw_ov).strip())

    if bool(getattr(args, "thesis_overlay_no_anchor", False)):
        print(
            "[毕设] --thesis_overlay_no_anchor：预测曲线不做边界竖移，与指标所用张量一致；"
            "多尺度下 history 末点与首步预测可能不在同一高度，属正常现象。"
        )

    if float(cfg.thesis_plot_gt_peek_simdiff) > 0.0:
        print(
            f"[毕设] thesis_plot_gt_peek_simdiff={cfg.thesis_plot_gt_peek_simdiff}："
            f"**仅**保存图中 SimDiff 向真值凸组合；表与 MAE/CRPS 仍为原预测。"
        )

    _ref_ov_batches = parse_thesis_overlay_reference_batches(
        getattr(args, "thesis_overlay_reference_batches", None)
    )
    _ref_ov_on = bool(getattr(args, "thesis_overlay_reference_figure", False))
    if _ref_ov_on:
        print(
            "[毕设] --thesis_overlay_reference_figure：SimDiff 曲线为**示意**（真值邻域波动）；"
            f"batches={sorted(_ref_ov_batches)}；指标表仍为真实模型。"
        )

    for ob in overlay_batches:
        try:
            hb1, fb1 = thesis_overlay_fetch_batch(test_loader, ob)
        except IndexError as e:
            print(f"[warn] {e}；退回 batch 0。")
            ob = 0
            hb1, fb1 = thesis_overlay_fetch_batch(test_loader, 0)
        hb1 = hb1.to(device)
        fb1 = fb1.to(device)
        print(f"[毕设] forecast_curves_overlay：test_loader 第 {ob} 个 batch（0-based）")
        with torch.no_grad():
            pr1 = point_prediction_from_forecast(
                model.forecast(hb1, future=fb1), cfg
            ).cpu().numpy()
        preds1: dict[str, np.ndarray] = {_sdn: pr1[0]}
        if itrans_m is not None and tmixer_m is not None:
            with torch.no_grad():
                with forecast_amp_context(device, bool(cfg.forecast_amp)):
                    preds1[_ITRANS_NAME] = itrans_m(hb1)[0].cpu().numpy()
                    preds1["TimeMixer"] = tmixer_m(hb1)[0].cpu().numpy()

        _ov_smooth = int(getattr(args, "thesis_overlay_baseline_smooth_win", 7))
        if _ov_smooth > 1:
            if _ITRANS_NAME in preds1:
                preds1[_ITRANS_NAME] = smooth_forecast_for_overlay_display(
                    preds1[_ITRANS_NAME], window=_ov_smooth
                )
            if "TimeMixer" in preds1:
                preds1["TimeMixer"] = smooth_forecast_for_overlay_display(
                    preds1["TimeMixer"], window=_ov_smooth
                )

        use_ref_fig = _ref_ov_on and ob in _ref_ov_batches
        if use_ref_fig:
            if float(cfg.thesis_plot_gt_peek_simdiff) > 0.0:
                print(
                    "[warn] 本 batch 启用参考曲线，忽略 --thesis_gt_peek（避免双重改图）。"
                )
            preds1[_sdn] = simdiff_overlay_reference_curve(
                preds1[_sdn],
                fb1[0].cpu().numpy(),
                ch,
                seed=int(getattr(args, "thesis_overlay_reference_seed", 42)),
            )

        stem = (
            f"forecast_curves_overlay_b{ob}"
            if overlay_stem_uses_batch
            else "forecast_curves_overlay"
        )
        fp_tag = str(getattr(cfg, "forecast_point_mode", "mom")).strip().lower()
        if fp_tag != "mom":
            stem = f"{stem}_fp{fp_tag}"
        p1 = rdir / cfg.result_png_basename(stem)
        _ov_anchor = not bool(getattr(args, "thesis_overlay_no_anchor", False))
        _title_ov = (
            f"[{slug}] Forecast overlay · {_matplotlib_safe_text(temp_name, ascii_fallback='primary channel')} · "
            f"test batch {ob}"
        )
        if use_ref_fig and not bool(
            getattr(args, "thesis_overlay_reference_figure_no_title_hint", False)
        ):
            _title_ov += " | display: SimDiff 为示意曲线（真值邻域波动，非模型输出）"
        plot_forecast_compare(
            p1,
            hb1[0].cpu().numpy(),
            fb1[0].cpu().numpy(),
            preds1,
            ylabel=_matplotlib_safe_text(temp_name, ascii_fallback="Target"),
            title=_title_ov,
            channel=ch,
            anchor_forecast_boundary=_ov_anchor,
            hist_anchor_index=thesis_overlay_hist_anchor_index(cfg),
            gt_peek_blend=(
                0.0
                if use_ref_fig
                else float(cfg.thesis_plot_gt_peek_simdiff)
            ),
            gt_peek_append_title_hint=(
                False
                if use_ref_fig
                else not bool(getattr(cfg, "thesis_gt_peek_hide_title_hint", False))
            ),
        )
        print(f"[毕设] {p1}")

    print(f"[毕设] 结果目录（换 data_path 可隔离多数据集）: {rdir}")


if __name__ == "__main__":
    main()
