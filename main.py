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
from models.simdiff import SimDiffWeather
from utils.baselines import (
    BaselineLSTM,
    BaselineTransformer,
    DLinearMap,
    collect_pooled_predictions,
    eval_channel_mse_mae,
    eval_forecasts_mse_mae,
    eval_horizon_mae,
    fit_regression_model,
    moving_average_forecast,
    persistence_forecast,
    print_baseline_block,
)
from utils.compare_viz import (
    plot_forecast_compare,
    plot_forecast_grid,
    plot_horizon_mae,
    plot_metrics_bars,
    plot_pred_vs_true_scatter,
)
from utils.data_loader import make_loaders
from utils.trainer import Trainer


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
        pred = out.mom
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
        help="跳过训练，只加载 checkpoints/simdiff_weather_best.pt 做测试。未训练时指标为随机权重，无意义。",
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
        help="跳过 persistence / MA / DLinear 基线评估",
    )
    parser.add_argument(
        "--verify_norm_mom",
        action="store_true",
        help="仅运行归一化/MoM 快速自检后退出（不训练）",
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

    cfg.validate_mom_config()

    set_seed(cfg.seed)
    train_loader, val_loader, test_loader, n_features, feat_names = make_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    if args.verify_norm_mom:
        import verify_norm_mom as vnm

        vnm.run_quick_verify(cfg, train_loader, device)
        return

    print("Device:", device)
    if (
        cfg.sampling_mode.lower() == "ddim"
        and cfg.sampling_steps is not None
        and cfg.sampling_steps > cfg.timesteps
    ):
        print(
            f"[warn] sampling_steps={cfg.sampling_steps} > timesteps={cfg.timesteps}："
            f"DDIM 将截断为 {cfg.timesteps}，多余子步在离散日程上无对应，曾导致采样发散。"
        )
    print(
        f"采样: mode={cfg.sampling_mode}, steps={cfg.sampling_steps or cfg.timesteps}, "
        f"ddim_eta={cfg.ddim_eta}, MoM: K={cfg.forecast_num_samples}, M={cfg.mom_num_groups}"
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
    ckpt_path = cfg.resolved_checkpoint_dir() / "simdiff_weather_best.pt"

    if args.eval_only:
        print("\n" + "=" * 60)
        print("【仅评估模式】未执行训练。若从未成功训练，权重为随机或陈旧，指标不可信。")
        print("=" * 60 + "\n")
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"未找到 {ckpt_path}。请先运行: python main.py   （不要加 --eval_only）"
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
        trainer = Trainer(cfg, model, train_loader, val_loader, device)
        trainer.fit()

    # 测试与画图
    model.eval()
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

    with torch.no_grad():
        hist, fut = next(iter(test_loader))
        hist = hist.to(device)
        fut = fut.to(device)
        out_s = model.forecast(hist, future=fut)
        pred_m = out_s.mom
        mse_s = torch.mean((pred_m - fut) ** 2)
        mae_s = torch.mean(torch.abs(pred_m - fut))
    t_idx = resolve_temperature_feature_index(feat_names)
    temp_name = feat_names[t_idx] if t_idx < len(feat_names) else str(t_idx)
    _lines = [
        f"采样评估（单 batch，MoM：K={cfg.forecast_num_samples}, M={cfg.mom_num_groups}）",
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
    print(
        f"\n【主结论 · 温度】{temp_name} — 全测试集 MSE: {mse_ch[t_idx]:.6f} | MAE: {mae_ch[t_idx]:.6f}"
    )
    if n_features > 1:
        print(f"【辅助】全特征平均 MSE: {mse_test:.6f} | MAE: {mae_test:.6f}")

    plot_dir = cfg.resolved_plot_dir()
    c_vis = t_idx if t_idx < n_features else min(1, n_features - 1)
    ylabel = feat_names[c_vis] if c_vis < len(feat_names) else str(c_vis)
    t_hist = np.arange(cfg.seq_len)
    t_fut = np.arange(cfg.seq_len, cfg.seq_len + cfg.pred_len)

    horizon_maes: dict[str, np.ndarray] = {
        "SimDiff": eval_horizon_mae(
            lambda h, f: model.forecast(h, future=f).mom,
            test_loader,
            device,
            cfg.pred_len,
            t_idx,
            n_features,
        )
    }
    bar_names: list[str] = ["SimDiff"]
    bar_maes: list[float] = [float(mae_ch[t_idx])]
    bar_mses: list[float] = [float(mse_ch[t_idx])]

    dlm: DLinearMap | None = None
    lstm_m: BaselineLSTM | None = None
    trans_m: BaselineTransformer | None = None

    if not args.skip_baselines:
        print("\n--- 基线（原始尺度，全测试集）---")
        pl, sl, C = cfg.pred_len, cfg.seq_len, n_features

        def wrap_persist(h):
            return persistence_forecast(h, pl)

        def wrap_ma(h):
            return moving_average_forecast(h, pl, window=min(24, sl))

        mse_b, mae_b = eval_forecasts_mse_mae(wrap_persist, test_loader, device)
        mt_b, at_b = eval_channel_mse_mae(wrap_persist, test_loader, device, t_idx)
        print_baseline_block("Persistence", mse_b, mae_b, mt_b, at_b, temp_name, n_features)
        horizon_maes["Persistence"] = eval_horizon_mae(
            lambda h, f: persistence_forecast(h, pl),
            test_loader,
            device,
            cfg.pred_len,
            t_idx,
            n_features,
        )
        bar_names.append("Persistence")
        bar_maes.append(float(at_b))
        bar_mses.append(float(mt_b))

        mse_b, mae_b = eval_forecasts_mse_mae(wrap_ma, test_loader, device)
        mt_b, at_b = eval_channel_mse_mae(wrap_ma, test_loader, device, t_idx)
        print_baseline_block(
            f"Moving-avg (w={min(24, sl)})", mse_b, mae_b, mt_b, at_b, temp_name, n_features
        )
        horizon_maes[f"MA(w={min(24, sl)})"] = eval_horizon_mae(
            lambda h, f: moving_average_forecast(h, pl, window=min(24, sl)),
            test_loader,
            device,
            cfg.pred_len,
            t_idx,
            n_features,
        )

        print("\n--- 学习型基线（验证集 MSE 早停，与 SimDiff 同 epoch 上限）---")
        dlm = DLinearMap(sl, pl, C)
        dlm = fit_regression_model(
            dlm,
            train_loader,
            val_loader,
            device,
            max_epochs=cfg.baseline_max_epochs,
            lr=cfg.baseline_lr,
            patience=cfg.baseline_early_stop_patience,
            grad_clip_max_norm=cfg.baseline_grad_clip_max_norm,
            name="DLinear",
        )

        def wrap_dl(h):
            return dlm(h)

        mse_b, mae_b = eval_forecasts_mse_mae(wrap_dl, test_loader, device)
        mt_b, at_b = eval_channel_mse_mae(wrap_dl, test_loader, device, t_idx)
        print_baseline_block("DLinear", mse_b, mae_b, mt_b, at_b, temp_name, n_features)
        horizon_maes["DLinear"] = eval_horizon_mae(
            lambda h, f: dlm(h),
            test_loader,
            device,
            cfg.pred_len,
            t_idx,
            n_features,
        )
        bar_names.append("DLinear")
        bar_maes.append(float(at_b))
        bar_mses.append(float(mt_b))

        lstm_m = BaselineLSTM(C, pl, cfg.baseline_lstm_hidden, cfg.baseline_lstm_layers, cfg.dropout)
        lstm_m = fit_regression_model(
            lstm_m,
            train_loader,
            val_loader,
            device,
            max_epochs=cfg.baseline_max_epochs,
            lr=cfg.baseline_lr,
            patience=cfg.baseline_early_stop_patience,
            grad_clip_max_norm=cfg.baseline_grad_clip_max_norm,
            name="LSTM",
        )

        def wrap_lstm(h):
            return lstm_m(h)

        mse_b, mae_b = eval_forecasts_mse_mae(wrap_lstm, test_loader, device)
        mt_b, at_b = eval_channel_mse_mae(wrap_lstm, test_loader, device, t_idx)
        print_baseline_block("LSTM", mse_b, mae_b, mt_b, at_b, temp_name, n_features)
        horizon_maes["LSTM"] = eval_horizon_mae(
            lambda h, f: lstm_m(h),
            test_loader,
            device,
            cfg.pred_len,
            t_idx,
            n_features,
        )
        bar_names.append("LSTM")
        bar_maes.append(float(at_b))
        bar_mses.append(float(mt_b))

        trans_m = BaselineTransformer(
            sl,
            pl,
            C,
            d_model=cfg.baseline_transformer_d_model,
            nhead=cfg.baseline_transformer_nhead,
            num_layers=cfg.baseline_transformer_layers,
            dropout=cfg.dropout,
        )
        trans_m = fit_regression_model(
            trans_m,
            train_loader,
            val_loader,
            device,
            max_epochs=cfg.baseline_max_epochs,
            lr=cfg.baseline_lr,
            patience=cfg.baseline_early_stop_patience,
            grad_clip_max_norm=cfg.baseline_grad_clip_max_norm,
            name="Plain Transformer",
        )

        def wrap_tr(h):
            return trans_m(h)

        mse_b, mae_b = eval_forecasts_mse_mae(wrap_tr, test_loader, device)
        mt_b, at_b = eval_channel_mse_mae(wrap_tr, test_loader, device, t_idx)
        print_baseline_block(
            "Plain Transformer", mse_b, mae_b, mt_b, at_b, temp_name, n_features
        )
        horizon_maes["Plain Transformer"] = eval_horizon_mae(
            lambda h, f: trans_m(h),
            test_loader,
            device,
            cfg.pred_len,
            t_idx,
            n_features,
        )
        bar_names.append("Plain Transformer")
        bar_maes.append(float(at_b))
        bar_mses.append(float(mt_b))

        yt, yp = collect_pooled_predictions(
            lambda h, f: model.forecast(h, future=f).mom,
            test_loader,
            device,
            t_idx,
        )
        p_sc = plot_dir / "scatter_simdiff_pred_vs_true.png"
        plot_pred_vs_true_scatter(
            p_sc,
            yt,
            yp,
            title=f"SimDiff: predicted vs true ({temp_name})",
        )
        print(f"Saved plot: {p_sc}")

        hist_b, fut_b = next(iter(test_loader))
        hist_b = hist_b.to(device)
        fut_b = fut_b.to(device)
        with torch.no_grad():
            preds_np: dict[str, np.ndarray] = {
                "SimDiff": model.forecast(hist_b, future=fut_b).mom.cpu().numpy(),
                "DLinear": dlm(hist_b).cpu().numpy(),
                "LSTM": lstm_m(hist_b).cpu().numpy(),
                "Plain Transformer": trans_m(hist_b).cpu().numpy(),
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
    pred0 = fo.mom[0].cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(t_hist, hist0[:, c_vis], label="history", color="C0")
    plt.plot(t_fut, true0[:, c_vis], label="ground truth", color="C1")
    plt.plot(t_fut, pred0[:, c_vis], label="SimDiff (MoM)", color="C2", linestyle="--")
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


if __name__ == "__main__":
    main()
