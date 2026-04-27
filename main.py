#!/usr/bin/env python3
"""
SimDiff-Weather 主入口：默认 **先训练再评估**。
默认仅使用气温 `T (degC)` 单变量序列；`--all_features` 可恢复多变量。
使用 `--eval_only` 将 **跳过训练** 并加载已有权重——若从未训练，指标无意义。
"""
from __future__ import annotations

import argparse
import random
import sys
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
from utils.prob_metrics import (
    empirical_interval_coverage,
    eval_crps_on_test,
    mean_pred_sample_variance_on_test,
)
from utils.result_output import print_thesis_metrics_table
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


def simdiff_plot_name(cfg: Config) -> str:
    """图中/表中 SimDiff 曲线名称，区分消融。"""
    if cfg.simdiff_ablation == "full":
        return "SimDiff"
    if cfg.simdiff_ablation == "ni_only":
        return "SimDiff (NI, K-mean)"
    return "SimDiff (hist-norm, MoM)"


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
  python main.py --epochs 30          # 指定轮数
  python main.py --eval_only          # 仅评估（必须先训练生成 checkpoint）
""",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--all_features",
        action="store_true",
        help="使用 weather.csv 全部数值列（默认仅气温单变量）",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="历史步长（默认读取 Config）；增大有助于长期依赖，需重训",
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
        help="ddim 或 ddpm（默认与 config 一致；ddpm 与训练日程一致，通常更稳）",
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
    args = parser.parse_args()

    cfg = Config()
    if args.all_features:
        cfg.temperature_only = False
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
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

    cfg.validate_mom_config()
    cfg.validate_simdiff_ablation()
    if args.all_plots:
        cfg.thesis_result_only = False

    set_seed(cfg.seed)
    train_loader, val_loader, test_loader, n_features, feat_names = make_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    if args.verify_norm_mom:
        import verify_norm_mom as vnm

        vnm.run_quick_verify(cfg, train_loader, device)
        return

    print("Device:", device)
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
    print(
        f"训练噪声损失: MSE + L1×{cfg.training_noise_l1_weight} + "
        f"时间差分×{cfg.training_noise_temporal_diff_weight}"
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
    mse_test, mae_test, mae_ch, mse_ch = evaluate_test_loader(
        model, test_loader, device, n_features
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
        crps_mean, crps_h = eval_crps_on_test(
            model, test_loader, device, t_idx, cfg.pred_len
        )
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
                np.arange(cfg.seq_len),
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
    t_hist = np.arange(cfg.seq_len)
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

    if not args.skip_baselines:
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
                cfg.seq_len,
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
    crps_mean_t, _ = eval_crps_on_test(model, test_loader, device, ch, cfg.pred_len)
    var_mean_t = mean_pred_sample_variance_on_test(model, test_loader, device, ch)

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
        ma_itr, ms_itr = eval_channel_mse_mae(_wrap_itr_thesis, test_loader, device, ch)
        ma_tm, ms_tm = eval_channel_mse_mae(_wrap_tm_thesis, test_loader, device, ch)
        # Degenerate ensemble CRPS == MAE; no sample spread -> VAR 0
        table_rows.append((_ITRANS_NAME, ma_itr, ms_itr, f"{ma_itr:.6f}", f"{0.0:.6f}"))
        table_rows.append(("TimeMixer", ma_tm, ms_tm, f"{ma_tm:.6f}", f"{0.0:.6f}"))
        bar_n.extend([_ITRANS_NAME, "TimeMixer"])
        bar_mae_l.extend([ma_itr, ma_tm])
        bar_mse_l.extend([ms_itr, ms_tm])

    print_thesis_metrics_table(table_rows, f"{slug} · {temp_name}")

    p_bar_r = rdir / "bar_mae_mse_temperature.png"
    plot_metrics_bars(
        p_bar_r,
        bar_n,
        bar_mae_l,
        bar_mse_l,
        title=f"[{slug}] Test MAE/MSE: {temp_name}",
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

    p1 = rdir / "forecast_curves_temperature_overlay.png"
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
    )
    print(f"[毕设] {p1}")

    print(f"[毕设] 结果目录（换 data_path 可隔离多数据集）: {rdir}")


if __name__ == "__main__":
    main()
