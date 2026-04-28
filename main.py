#!/usr/bin/env python3
"""
SimDiff-Weather 主入口：默认 **先训练再评估**。
默认仅使用气温 `T (degC)` 单变量序列；`--all_features` 可恢复多变量。
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
    BaselineTimeMixer,
    BaselineiTransformer,
    collect_channel_residuals,
    collect_pooled_predictions,
    eval_channel_mse_mae,
    eval_forecasts_mse_mae,
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
    """单变量气温时恒为 0；多变量时解析气温列索引。"""
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
    return min(1, len(feat_names) - 1)


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
) -> tuple[float, float, np.ndarray, np.ndarray, float, float, np.ndarray]:
    """
    **单次**遍历测试集：`forecast(return_samples=True)` 一次算清
    与 `evaluate_test_loader` 相同的 MAE/MSE、`temp_channel` 上 CRPS（均值 + 按步）、VAR。
    避免对同一模型重复 K 次扩散采样（原先约 3× 墙钟）。
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
        if out.samples is not None:
            s = out.samples[..., temp_channel]
            y = fut[..., temp_channel]
            crps_bl = crps_ensemble_1d(s, y)
            sum_crps += float(crps_bl.sum().item())
            n_crps += crps_bl.numel()
            sum_h += crps_bl.sum(dim=0).detach().cpu().numpy()
            n_h += crps_bl.size(0)
            v = s.var(dim=1, unbiased=False)
            sum_var += float(v.sum().item())
            n_var += v.numel()
    mse_all = sum_sq / max(n_elem, 1)
    mae_all = sum_abs / max(n_elem, 1)
    steps = n_elem // max(n_features, 1)
    mse_ch = (sum_sq_ch / max(steps, 1)).cpu().numpy()
    mae_ch = (sum_abs_ch / max(steps, 1)).cpu().numpy()
    mean_crps = sum_crps / max(n_crps, 1)
    mean_var = sum_var / max(n_var, 1)
    crps_h = sum_h / np.maximum(n_h, 1)
    return mse_all, mae_all, mae_ch, mse_ch, mean_crps, mean_var, crps_h


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
    cfg.use_multiscale_hist = False
    cfg.hist_window_start_min = 0
    cfg.use_revin = True
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
    rdir = cfg.resolved_result_dir()
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
        trainer_v = Trainer(cfg, model_v, train_loader, val_loader, device)
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
                float(mae_ch[t_idx]),
                float(mse_ch[t_idx]),
                f"{crps_mean:.6f}",
                f"{var_mean:.6f}",
            )
        )
        bar_names.append(label)
        bar_maes.append(float(mae_ch[t_idx]))
        bar_mses.append(float(mse_ch[t_idx]))
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

    t_hist = np.arange(cfg.seq_len)
    t_fut = np.arange(cfg.seq_len, cfg.seq_len + cfg.pred_len)
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
        t_hist,
        t_fut,
        hist_plot,
        fb_ref[0].detach().cpu().numpy(),
        preds_overlay,
        ylabel=plot_ylab,
        title=f"[{plot_slug}] ms_rms overlay / ch {t_idx} / batch0 (hist=96h strip)",
        channel=t_idx,
        gt_peek_blend=float(cfg.thesis_plot_gt_peek_simdiff),
        gt_peek_name_prefix="SimDiff",
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
    rdir = cfg.resolved_result_dir()
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
        trainer_v = Trainer(cfg, model_v, train_loader, val_loader, device)
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
    t_hist = np.arange(cfg.seq_len)
    t_fut = np.arange(cfg.seq_len, cfg.seq_len + cfg.pred_len)

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
                float(mae_ch[t_idx]),
                float(mse_ch[t_idx]),
                f"{crps_mean:.6f}",
                f"{var_mean:.6f}",
            )
        )
        bar_names.append(label)
        bar_maes.append(float(mae_ch[t_idx]))
        bar_mses.append(float(mse_ch[t_idx]))
        with torch.no_grad():
            pr = point_prediction_from_forecast(
                model_v.forecast(hb1, future=fb1), cfg
            ).cpu().numpy()
        preds_overlay[label] = pr[0]
        del model_v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _clear_denoiser_ablation_key(cfg)
    cfg.use_revin, cfg.use_rmsnorm = True, True

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
        t_hist,
        t_fut,
        hb1[0].cpu().numpy(),
        fb1[0].cpu().numpy(),
        preds_overlay,
        ylabel=plot_ylab,
        title=f"[{plot_slug}] Denoiser ablation overlay / {plot_ylab} / batch 0",
        channel=t_idx,
        gt_peek_blend=float(cfg.thesis_plot_gt_peek_simdiff),
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
        help="使用 weather.csv 全部数值列（默认仅气温单变量）",
    )
    parser.add_argument(
        "--multiscale_hist",
        action="store_true",
        help="单模型训练：历史输入为 96h 原始 + 7 日均值 + 4 周均值拼接（共 107 token），需更长向左上下文",
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
        help="追加到 SimDiff checkpoint 文件名 stem 与 .pt 之间（如 _pl48）；"
        "不写则仍为 simdiff_weather_best_ms_rms_full.pt 等，可能覆盖已有权重",
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
        "--no_revin",
        action="store_true",
        help="单次训练关闭去噪器 RevIn（与 --revin_rms_ablation 互斥于后者内部设定）",
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
        help="毕设图仍用无后缀旧文件名（如 bar_mae_mse_temperature.png），会覆盖同目录已有文件",
    )
    args = parser.parse_args()
    _run_ms_rms = bool(args.ms_rms_ablation or args.dual_ablation)
    if getattr(args, "dual_reuse_b_only_ckpt", False):
        args.ms_rms_reuse_rmsnorm_ckpt = True
    if args.revin_rms_ablation and _run_ms_rms:
        parser.error("--revin_rms_ablation 与 --ms_rms_ablation（含弃用的 --dual_ablation）不能同时使用")

    cfg = Config()
    if args.hist_add_bias_scale is not None:
        cfg.hist_add_bias_scale = float(args.hist_add_bias_scale)
    if args.hist_add_bias_scale_with_rmsnorm is not None:
        cfg.hist_add_bias_scale_with_rmsnorm = float(args.hist_add_bias_scale_with_rmsnorm)
    if args.all_features:
        cfg.temperature_only = False
    if getattr(args, "multiscale_hist", False):
        cfg.use_multiscale_hist = True
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
    if args.pred_len is not None:
        cfg.pred_len = max(1, int(args.pred_len))
    if args.ckpt_extra_suffix is not None:
        s = str(args.ckpt_extra_suffix).strip()
        cfg.simdiff_checkpoint_extra_suffix = s if s else None
    if args.sampling_steps is not None:
        cfg.sampling_steps = args.sampling_steps
    if args.ddim_eta is not None:
        cfg.ddim_eta = args.ddim_eta
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
    if args.hist_add_bias:
        cfg.use_hist_add_bias = True
        cfg.use_revin = False
    if not args.revin_rms_ablation:
        if args.no_revin:
            cfg.use_revin = False
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
    cfg.validate_thesis_plot_options()
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
    a_h = float(cfg.training_noise_mse_huber_alpha)
    h_note = "纯 MSE" if a_h >= 0.999 else f"αMSE+({1.0 - a_h:.2f})smooth_l1(β={cfg.training_noise_huber_beta})"
    print(
        f"训练噪声主项: {h_note} + L1×{cfg.training_noise_l1_weight} + "
        f"时间差分×{cfg.training_noise_temporal_diff_weight}"
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
    print(f"特征维度 C={n_features}（{mode}）, 列: {feat_names}")
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
        trainer_ref = Trainer(cfg, model, train_loader, val_loader, device)
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

    t_idx = resolve_temperature_feature_index(feat_names)
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
    ) = evaluate_test_loader_prob_combined(
        model, test_loader, device, n_features, t_idx, cfg.pred_len, cfg
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
                np.arange(cfg.effective_hist_len()),
                np.arange(cfg.seq_len, cfg.seq_len + cfg.pred_len),
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
    _ehl = int(cfg.effective_hist_len())
    t_hist = np.arange(_ehl)
    t_fut = np.arange(cfg.seq_len, cfg.seq_len + cfg.pred_len)

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
    bar_maes: list[float] = [float(mae_ch[t_idx])]
    bar_mses: list[float] = [float(mse_ch[t_idx])]

    itrans_m: BaselineiTransformer | None = None
    tmixer_m: BaselineTimeMixer | None = None
    residual_multi_for_kde: dict[str, np.ndarray] | None = None

    run_learned_baselines = not args.skip_baselines and not cfg.use_multiscale_hist
    if cfg.use_multiscale_hist and not args.skip_baselines:
        print(
            "[note] use_multiscale_hist 下历史序列长于 96；iTransformer/TimeMixer 固定 seq_len=96，已跳过学习型基线。"
        )

    if run_learned_baselines:
        itrans_epochs = cfg.epochs
        print(
            "\n--- 学习型基线 iTransformer / TimeMixer（val MSE 早停；"
            f"iTransformer max_epochs={itrans_epochs}（与 SimDiff）；"
            f"TimeMixer max_epochs={cfg.baseline_timemixer_max_epochs}）---"
        )
        pl, sl, C = cfg.pred_len, cfg.seq_len, n_features
        amp_on = bool(cfg.forecast_amp)

        itrans_m = BaselineiTransformer(
            sl,
            pl,
            C,
            d_model=cfg.baseline_itransformer_d_model,
            nhead=cfg.baseline_itransformer_nhead,
            num_layers=cfg.baseline_itransformer_layers,
            dropout=cfg.dropout,
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
        bar_maes.append(float(at_b))
        bar_mses.append(float(mt_b))

        tmixer_m = BaselineTimeMixer(
            sl,
            pl,
            C,
            d_model=cfg.baseline_timemixer_d_model,
            n_scales=cfg.baseline_timemixer_scales,
            dropout=cfg.dropout,
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
        bar_maes.append(float(at_b))
        bar_mses.append(float(mt_b))

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
                t_hist,
                t_fut,
                hist0,
                true0,
                pred_dict,
                ylabel=ylabel,
                title=f"Forecast comparison (first test batch sample) — {temp_name}",
                channel=t_idx,
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
                _ehl,
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

        plt.figure(figsize=(10, 4))
        plt.plot(t_hist, hist0[:, c_vis], label="history", color="C0")
        plt.plot(t_fut, true0[:, c_vis], label="ground truth", color="black")
        plt.plot(t_fut, pred0[:, c_vis], label=_sdn, color="C2", linestyle="--")
        plt.axvline(cfg.seq_len - 0.5, color="gray", linestyle=":")
        plt.xlabel("time step (index)")
        plt.ylabel(ylabel)
        plt.title("SimDiff-Weather: forecast (single model)")
        plt.legend()
        plt.tight_layout()
        plot_path = plot_dir / "forecast_example.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_path}")

    # -------- 毕设专用：result/<数据集名>/，与 plots/ 隔离；温度主变量指标与曲线 --------
    rdir = cfg.resolved_result_dir()
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

    table_rows: list[tuple[str, float, float, str, str]] = [
        (
            _sdn,
            float(mae_ch[ch]),
            float(mse_ch[ch]),
            f"{crps_mean_t:.6f}",
            f"{var_mean_t:.6f}",
        ),
    ]
    bar_n = [_sdn]
    bar_mae_l = [float(mae_ch[ch])]
    bar_mse_l = [float(mse_ch[ch])]
    if itrans_m is not None and tmixer_m is not None:
        # eval_channel_mse_mae 返回 (MSE, MAE)（与全通道 eval 命名一致，勿对调毕设表/图）
        mse_itr, mae_itr = eval_channel_mse_mae(_wrap_itr_thesis, test_loader, device, ch)
        mse_tm, mae_tm = eval_channel_mse_mae(_wrap_tm_thesis, test_loader, device, ch)
        # Degenerate ensemble CRPS == MAE; no sample spread -> VAR 0
        table_rows.append(
            (_ITRANS_NAME, mae_itr, mse_itr, f"{mae_itr:.6f}", f"{0.0:.6f}")
        )
        table_rows.append(("TimeMixer", mae_tm, mse_tm, f"{mae_tm:.6f}", f"{0.0:.6f}"))
        bar_n.extend([_ITRANS_NAME, "TimeMixer"])
        bar_mae_l.extend([mae_itr, mae_tm])
        bar_mse_l.extend([mse_itr, mse_tm])

    print_thesis_metrics_table(table_rows, f"{slug} · {temp_name}")

    p_bar_r = rdir / cfg.result_png_basename("bar_mae_mse_temperature")
    plot_metrics_bars(
        p_bar_r,
        bar_n,
        bar_mae_l,
        bar_mse_l,
        title=f"{slug} · test MAE/MSE",
        ylabel="MAE / MSE",
        title_fontsize=9.5,
    )
    print(f"[毕设] {p_bar_r}")

    hb1, fb1 = next(iter(test_loader))
    hb1 = hb1.to(device)
    fb1 = fb1.to(device)
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

    p1 = rdir / cfg.result_png_basename("forecast_curves_temperature_overlay")
    if float(cfg.thesis_plot_gt_peek_simdiff) > 0.0:
        print(
            f"[毕设] thesis_plot_gt_peek_simdiff={cfg.thesis_plot_gt_peek_simdiff}："
            f"**仅**保存图中 SimDiff 向真值凸组合；表与 MAE/CRPS 仍为原预测。"
        )
    plot_forecast_compare(
        p1,
        t_hist,
        t_fut,
        hb1[0].cpu().numpy(),
        fb1[0].cpu().numpy(),
        preds1,
        ylabel=temp_name,
        title=f"[{slug}] Forecast overlay / {temp_name} / batch 0",
        channel=ch,
        gt_peek_blend=float(cfg.thesis_plot_gt_peek_simdiff),
    )
    print(f"[毕设] {p1}")

    print(f"[毕设] 结果目录（换 data_path 可隔离多数据集）: {rdir}")


if __name__ == "__main__":
    main()
