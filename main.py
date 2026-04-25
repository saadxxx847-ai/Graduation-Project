#!/usr/bin/env python3
"""
SimDiff-Weather 主入口：默认 **先训练再评估**。
默认仅使用气温 `T (degC)` 单变量序列；`--all_features` 可恢复多变量。
使用 `--eval_only` 将 **跳过训练** 并加载已有权重——若从未训练，指标无意义。
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections.abc import Callable
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
    BaselineLSTM,
    ITransformer,
    DLinearMap,
    collect_channel_residuals,
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
from utils.prob_metrics import empirical_interval_coverage, eval_crps_on_test
from utils.data_loader import make_loaders
from utils.trainer import Trainer


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
    """图中/表中 SimDiff / mr-Diff 曲线名称，区分改进与消融。"""
    if bool(getattr(cfg, "mrdiff_denoiser", False)):
        return "mr-Diff"
    if cfg.simdiff_ablation == "ni_only":
        return "SimDiff (NI, K-mean)"
    if cfg.simdiff_ablation == "mom_only":
        return "SimDiff (hist-norm, MoM)"
    if not bool(getattr(cfg, "use_patch", False)) and not bool(
        getattr(cfg, "use_rope", False)
    ):
        return "SimDiff"
    if bool(getattr(cfg, "use_patch", False)) and bool(getattr(cfg, "use_rope", False)):
        # 图例/曲线名用纯英文，避免「中文 + SimDiff」混排时 DejaVu 与 CJK 回退字体接缝处变宽像多空格
        return "Improved SimDiff"
    if bool(getattr(cfg, "use_patch", False)):
        return "SimDiff (+Patch)"
    return "SimDiff (+RoPE)"


def _paper_diff_variant_id(cfg: Config) -> str:
    """与论文 3 种扩散主对比对齐：mrdiff / 基线 simdiff / ours；其余 (仅 Patch 或仅 RoPE) 为 other。"""
    if cfg.mrdiff_denoiser:
        return "mrdiff"
    if cfg.use_patch and cfg.use_rope:
        return "ours"
    if not cfg.use_patch and not cfg.use_rope:
        return "simdiff"
    return "other"


def _load_sibling_diffusion(
    base_cfg: Config, device: torch.device, *, mrdiff: bool, use_patch: bool, use_rope: bool
) -> SimDiffWeather | None:
    c = copy.deepcopy(base_cfg)
    c.mrdiff_denoiser = mrdiff
    c.use_patch = use_patch
    c.use_rope = use_rope
    p = c.resolved_checkpoint_dir() / c.simdiff_checkpoint_filename()
    if not p.is_file():
        return None
    st = torch.load(p, map_location=device, weights_only=False)
    meta = st.get("meta") or {}
    if "timesteps" in meta:
        c.timesteps = int(meta["timesteps"])
    if meta.get("sampling_steps") is not None:
        c.sampling_steps = int(meta["sampling_steps"])
    m = SimDiffWeather(c).to(device)
    m.load_state_dict(st["model"], strict=True)
    m.eval()
    return m


def _paper_diffusion_stack(
    base_cfg: Config, device: torch.device, main: SimDiffWeather
) -> list[tuple[str, SimDiffWeather]]:
    """
    论文主对比：mr-Diff、SimDiff(基线)、Improved SimDiff(ours) 同序；
    当前 run 用 main，其余从 checkpoint 补全；缺权重则少一条。
    """
    if base_cfg.simdiff_ablation != "full" or _paper_diff_variant_id(base_cfg) == "other":
        return [(simdiff_plot_name(base_cfg), main)]
    vid = _paper_diff_variant_id(base_cfg)
    out: list[tuple[str, SimDiffWeather]] = []
    slots: list[tuple[str, str, bool, bool, bool]] = [
        ("mr-Diff", "mrdiff", True, False, False),
        ("SimDiff", "simdiff", False, False, False),
        ("Improved SimDiff", "ours", False, True, True),
    ]
    for label, key, md, up, ro in slots:
        if vid == key:
            out.append((label, main))
        else:
            sib = _load_sibling_diffusion(base_cfg, device, mrdiff=md, use_patch=up, use_rope=ro)
            if sib is not None:
                out.append((label, sib))
            else:
                ck = copy.deepcopy(base_cfg)
                ck.mrdiff_denoiser = md
                ck.use_patch, ck.use_rope = up, ro
                miss = base_cfg.resolved_checkpoint_dir() / ck.simdiff_checkpoint_filename()
                print(f"  [hint] 未找到 {miss.name}，同图/柱图不显示 {label}；请先训练该变体。")
    return out


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
  python main.py                      # 训练+测试；默认 **不写** 诊断图到 plots/，见 --full_plots
  python main.py --full_plots         # 同时保存旧版散点/对比/KDE/指标等到 plots/
  python main.py --epochs 30          # 指定轮数
  python main.py --eval_only          # 仅评估（必须先训练生成 checkpoint）
  论文用图/表见：python generate_paper_artifacts.py（表与图在 outputs/paper/，调试图在 plots/diagnostics/）
""",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="优化器学习率，默认见 config（当前默认 0.003）",
    )
    parser.add_argument(
        "--no_fp16_predict",
        action="store_true",
        help="预测/全测试集评估时不用半精度 autocast（默认 CUDA 上开启以加速采样）",
    )
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
        help="跳过 persistence / MA / DLinear 基线评估",
    )
    parser.add_argument(
        "--include_stat_baselines",
        action="store_true",
        help="将 Persistence / 移动平均 纳入柱图与按步 MAE（默认仅 DLinear、iTransformer 与扩散类主对比）",
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
        "--data_preset",
        type=str,
        default=None,
        help="weather | etth1 | ettm1 | exchange | wind（将自动设置数据路径与默认目标列）",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=None,
        help="预测步长（主实验 48/72/168/192；消融 ETTh1(OT) 建议 168）",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="单变量目标列名（如 OT、0、ture_w_speed）；覆盖 preset 默认",
    )
    parser.add_argument("--use_patch", action="store_true", help="启用 Patch 分块嵌入")
    parser.add_argument("--use_rope", action="store_true", help="显式 RoPE（不用可学习绝对位置）")
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--patch_stride", type=int, default=None)
    parser.add_argument(
        "--mrdiff",
        action="store_true",
        help="使用 1D 卷积去噪网作为 mr-Diff 对照（含同一扩散与 NI/MoM）",
    )
    parser.add_argument(
        "--multivariate",
        action="store_true",
        help="使用 CSV 全部数值列（多变量）；默认单变量仅目标列",
    )
    parser.add_argument(
        "--skip_lstm",
        action="store_true",
        help="不在基线对比中训练 LSTM（论文五模型表可省略 LSTM）",
    )
    parser.add_argument(
        "--print_result_json",
        action="store_true",
        help="测试结束后多打一行 RESULT_JSON 便于记录主变量 MSE/MAE 到表",
    )
    parser.add_argument(
        "--save_run_metrics_dir",
        type=str,
        default=None,
        help="将本 run 的 DLinear / iTransformer / 当前扩散 指标写入该目录下 JSON，供 build_paper 合并成总表",
    )
    parser.add_argument(
        "--full_plots",
        action="store_true",
        help="保存全部历史诊断 PNG 到 plot_dir；默认不保存。论文用图见 generate_paper_artifacts.py / utils.paper_output",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.data_preset:
        cfg.data_preset = args.data_preset
    if args.pred_len is not None:
        cfg.pred_len = int(args.pred_len)
    if args.target is not None:
        cfg.target_column = args.target.strip()
    if args.use_patch:
        cfg.use_patch = True
    if args.use_rope:
        cfg.use_rope = True
    if args.patch_size is not None:
        cfg.patch_size = int(args.patch_size)
    if args.patch_stride is not None:
        cfg.patch_stride = int(args.patch_stride)
    if args.mrdiff:
        cfg.mrdiff_denoiser = True
    if args.multivariate:
        cfg.univariate = False
    if args.all_features:
        cfg.temperature_only = False
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.learning_rate is not None:
        cfg.learning_rate = float(args.learning_rate)
    if bool(getattr(args, "no_fp16_predict", False)):
        cfg.infer_fp16 = False
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
    cfg.validate_arch()

    set_seed(cfg.seed)
    train_loader, val_loader, test_loader, n_features, feat_names = make_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    if args.verify_norm_mom:
        import verify_norm_mom as vnm

        vnm.run_quick_verify(cfg, train_loader, device)
        return

    print("Device:", device)
    print(
        f"数据: preset={cfg.data_preset} | 路径: {cfg.data_path} | "
        f"seq_len={cfg.seq_len} pred_len={cfg.pred_len}"
    )
    print(
        f"网络: use_patch={cfg.use_patch} use_rope={cfg.use_rope} "
        f"patch=({cfg.patch_size},{cfg.patch_stride}) mrdiff={getattr(cfg, 'mrdiff_denoiser', False)}"
    )
    print(
        f"训练: lr={cfg.learning_rate} epochs={cfg.epochs} | "
        f"预测半精度(仅采样): infer_fp16={getattr(cfg, 'infer_fp16', False)} "
        f"（关: --no_fp16_predict）"
    )
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

    # 测试与画图（仅 --full_plots 时写入 plot_dir 下历史诊断图）
    do_plots = bool(getattr(args, "full_plots", False))
    model.eval()
    if (
        do_plots
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

    with torch.no_grad():
        hist, fut = next(iter(test_loader))
        hist = hist.to(device)
        fut = fut.to(device)
        out_s = model.forecast(hist, future=fut)
        pred_m = point_prediction_from_forecast(out_s, cfg)
        mse_s = torch.mean((pred_m - fut) ** 2)
        mae_s = torch.mean(torch.abs(pred_m - fut))
    t_idx = resolve_temperature_feature_index(feat_names)
    temp_name = feat_names[t_idx] if t_idx < len(feat_names) else str(t_idx)
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
    print(
        f"\n【主结论 · 温度】{temp_name} — 全测试集 MSE: {mse_ch[t_idx]:.6f} | MAE: {mae_ch[t_idx]:.6f}"
    )
    if n_features > 1:
        print(f"【辅助】全特征平均 MSE: {mse_test:.6f} | MAE: {mae_test:.6f}")

    if not args.skip_prob_metrics and do_plots:
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
    elif not args.skip_prob_metrics and not do_plots:
        print("（已跳过 CRPS/预测区间，未开 --full_plots）")

    plot_dir = cfg.resolved_plot_dir()
    c_vis = t_idx if t_idx < n_features else min(1, n_features - 1)
    ylabel = feat_names[c_vis] if c_vis < len(feat_names) else str(c_vis)
    t_hist = np.arange(cfg.seq_len)
    t_fut = np.arange(cfg.seq_len, cfg.seq_len + cfg.pred_len)

    if do_plots and not args.skip_denoise_traj:
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
    paper_stack = _paper_diffusion_stack(cfg, device, model)

    horizon_maes: dict[str, np.ndarray] = {}
    bar_names: list[str] = []
    bar_maes: list[float] = []
    bar_mses: list[float] = []

    dlm: DLinearMap | None = None
    lstm_m: BaselineLSTM | None = None
    residual_multi_for_kde: dict[str, np.ndarray] | None = None
    dlinear_mse: float | None = None
    dlinear_mae: float | None = None
    itrans_mse: float | None = None
    itrans_mae: float | None = None
    preds_np: dict[str, np.ndarray] = {}

    if args.skip_baselines:
        for lab, mm in paper_stack:
            horizon_maes[lab] = eval_horizon_mae(
                lambda h, f, mmm=mm: point_prediction_from_forecast(
                    mmm.forecast(h, future=f), mmm.cfg
                ),
                test_loader,
                device,
                cfg.pred_len,
                t_idx,
                n_features,
            )
            _mse_a, _mae_a, mae_chx, mse_chx = evaluate_test_loader(
                mm, test_loader, device, n_features
            )
            bar_names.append(lab)
            bar_maes.append(float(mae_chx[t_idx]))
            bar_mses.append(float(mse_chx[t_idx]))
    else:
        pl, sl, C = cfg.pred_len, cfg.seq_len, n_features

        if args.include_stat_baselines:
            print("\n--- 统计基线（Persistence / 移动平均，与论文主表无关时可省略）---")

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
        print("\n--- 学习型基线：DLinear(2023)、iTransformer(2024)；扩散类见下节 ---")
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
        dlinear_mse, dlinear_mae = float(mt_b), float(at_b)

        wrap_lstm: Callable | None
        if not bool(getattr(args, "skip_lstm", False)):
            lstm_m = BaselineLSTM(
                C, pl, cfg.baseline_lstm_hidden, cfg.baseline_lstm_layers, cfg.dropout
            )
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

            def _wrap_lstm(h: torch.Tensor) -> torch.Tensor:
                return lstm_m(h)

            wrap_lstm = _wrap_lstm
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
        else:
            lstm_m = None
            wrap_lstm = None

        itrans = ITransformer(
            sl,
            pl,
            C,
            d_model=cfg.baseline_transformer_d_model,
            nhead=cfg.baseline_transformer_nhead,
            num_layers=cfg.baseline_transformer_layers,
            dropout=cfg.dropout,
        )
        itrans = fit_regression_model(
            itrans,
            train_loader,
            val_loader,
            device,
            max_epochs=cfg.baseline_max_epochs,
            lr=cfg.baseline_lr,
            patience=cfg.baseline_early_stop_patience,
            grad_clip_max_norm=cfg.baseline_grad_clip_max_norm,
            name="iTransformer",
        )

        def wrap_itr(h):
            return itrans(h)

        mse_b, mae_b = eval_forecasts_mse_mae(wrap_itr, test_loader, device)
        mt_b, at_b = eval_channel_mse_mae(wrap_itr, test_loader, device, t_idx)
        print_baseline_block("iTransformer", mse_b, mae_b, mt_b, at_b, temp_name, n_features)
        horizon_maes["iTransformer"] = eval_horizon_mae(
            lambda h, f: itrans(h),
            test_loader,
            device,
            cfg.pred_len,
            t_idx,
            n_features,
        )
        bar_names.append("iTransformer")
        bar_maes.append(float(at_b))
        bar_mses.append(float(mt_b))
        itrans_mse, itrans_mae = float(mt_b), float(at_b)

        print(
            "\n--- 扩散主对比（mr-Diff / SimDiff(基线) / Improved；缺权重已跳过，见上 [hint]）---"
        )
        for lab, mm in paper_stack:
            _mse_b, _mae_b, mae_chx, mse_chx = evaluate_test_loader(
                mm, test_loader, device, n_features
            )
            print(
                f"  [{lab}] {temp_name} MSE={float(mse_chx[t_idx]):.6f} | "
                f"MAE={float(mae_chx[t_idx]):.6f}"
            )
            horizon_maes[lab] = eval_horizon_mae(
                lambda h, f, mmm=mm: point_prediction_from_forecast(
                    mmm.forecast(h, future=f), mmm.cfg
                ),
                test_loader,
                device,
                cfg.pred_len,
                t_idx,
                n_features,
            )
            bar_names.append(lab)
            bar_maes.append(float(mae_chx[t_idx]))
            bar_mses.append(float(mse_chx[t_idx]))

        r_dl = collect_channel_residuals(wrap_dl, test_loader, device, t_idx)
        residual_multi_for_kde = {"DLinear": r_dl}
        for lab, mm in paper_stack:
            r_k, _ = collect_test_forecast_errors(mm, test_loader, device, t_idx)
            residual_multi_for_kde[lab] = r_k
        if wrap_lstm is not None and lstm_m is not None:
            r_ls = collect_channel_residuals(wrap_lstm, test_loader, device, t_idx)
            residual_multi_for_kde["LSTM"] = r_ls

        yt, yp = collect_pooled_predictions(
            lambda h, f: point_prediction_from_forecast(
                model.forecast(h, future=f), model.cfg
            ),
            test_loader,
            device,
            t_idx,
        )
        if do_plots:
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
            _pd: dict[str, np.ndarray] = {
                "DLinear": dlm(hist_b).cpu().numpy(),
                "iTransformer": itrans(hist_b).cpu().numpy(),
            }
            for lab, mm in paper_stack:
                _pd[lab] = point_prediction_from_forecast(
                    mm.forecast(hist_b, future=fut_b), mm.cfg
                ).cpu().numpy()
            if lstm_m is not None:
                _pd["LSTM"] = lstm_m(hist_b).cpu().numpy()
            # 图例顺序：DLinear、iTransformer、三扩散、LSTM 置后
            _plot_order = (
                "DLinear",
                "iTransformer",
                "mr-Diff",
                "SimDiff",
                "Improved SimDiff",
                "LSTM",
            )
            preds_np = {k: _pd[k] for k in _plot_order if k in _pd}
        if do_plots:
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

    if do_plots and not args.skip_error_kde:
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
                title=f"Residual KDE (DLinear, iTransformer, LSTM, 扩散) — {temp_name}",
            )
            print(f"Saved plot: {p3}")
            print("  （左侧子图多模型时，error_distribution_kde.png 左栏为 Residual (multi-model)）")
        else:
            print(
                "  [hint] 未跑学习型基线或 residual 为空，KDE 多通道对比受限。"
                "需要完整主对比请不要加 --skip_baselines 并开 --full_plots。"
            )

    if do_plots:
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
        plt.plot(t_fut, true0[:, c_vis], label="ground truth", color="C1")
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
    else:
        print("（未开 --full_plots：不写入 plot_dir 下散点/对比/指标/单窗预报等图）")

    if bool(getattr(args, "print_result_json", False)):
        rec = {
            "data_preset": str(cfg.data_preset),
            "pred_len": int(cfg.pred_len),
            "seq_len": int(cfg.seq_len),
            "model": simdiff_plot_name(cfg),
            "target_mse": float(mse_ch[t_idx]),
            "target_mae": float(mae_ch[t_idx]),
        }
        print("RESULT_JSON " + json.dumps(rec, ensure_ascii=False))

    _mdir = getattr(args, "save_run_metrics_dir", None)
    if _mdir:
        mdir = Path(_mdir)
        mdir.mkdir(parents=True, exist_ok=True)
        tag = cfg.simdiff_checkpoint_tag()
        mrec: dict = {
            "data_preset": str(cfg.data_preset),
            "pred_len": int(cfg.pred_len),
            "seq_len": int(cfg.seq_len),
            "checkpoint_tag": tag,
            "diffusion_label": simdiff_plot_name(cfg),
            "diffusion_mse": float(mse_ch[t_idx]),
            "diffusion_mae": float(mae_ch[t_idx]),
            "DLinear_mse": dlinear_mse,
            "DLinear_mae": dlinear_mae,
            "iTransformer_mse": itrans_mse,
            "iTransformer_mae": itrans_mae,
        }
        mpath = mdir / f"{cfg.data_preset}_p{cfg.pred_len}_{tag}.json"
        mpath.write_text(json.dumps(mrec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已保存本 run 指标: {mpath}")


if __name__ == "__main__":
    main()
