#!/usr/bin/env python3
"""
独立脚本：生成「三条预测曲线轮换」后的 forecast overlay PNG。

轮换规则（仅在本脚本内实现；主程序 ``main.py`` / ``plot_forecast_compare`` 始终按真实模型对应绘图）：
  蓝(SimDiff 图例键) ← 原 TimeMixer 数值
  橙(iTransformer)     ← 原 SimDiff 数值
  绿(TimeMixer)       ← 原 iTransformer 数值
history / ground truth / 坐标轴 / 图例文字均不变。

用法 A —— 已有 numpy 存档（最快，推荐一次 full-eval 导出后反复改图）：
  python scripts/remap_overlay_three_models.py \\
    --npz overlay_batch0.npz \\
    --output ETTm1_single_scale_fixed/forecast_overlay_remapped.png

用法 B —— 从数据与 checkpoint 完整计算预测（会训练 iTransformer/TimeMixer，较慢）：
  python scripts/remap_overlay_three_models.py --full-eval \\
    --data-path data/ETTm1.csv --single-scale-hist \\
    --ckpt-extra-suffix _ettm1_ot_single_fixed \\
    --output ETTm1_single_scale_fixed/forecast_overlay_remapped.png \\
    --save-npz overlay_batch0.npz

NPZ 格式（save-npz / 自建均可）：
  hist          (Lh, C)
  fut           (Lf, C)
  SimDiff       (Lf, C)
  iTransformer  (Lf, C)
  TimeMixer     (Lf, C)

说明：无法从已有 PNG 像素无损还原三条曲线；必须从数值或重新推理得到数组后再绘图。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import Config  # noqa: E402
from models.simdiff import SimDiffWeather, point_prediction_from_forecast  # noqa: E402
from utils.baselines import (  # noqa: E402
    BaselineHistTrim,
    BaselineTimeMixer,
    BaselineiTransformer,
    fit_regression_model,
    forecast_amp_context,
)
from utils.compare_viz import (  # noqa: E402
    _anchor_preds_to_hist_end,
    _apply_gt_peek_blend_for_display,
    plot_forecast_compare,
)
from utils.data_loader import make_loaders  # noqa: E402


_ITRANS_NAME = "iTransformer"


def resolve_temperature_feature_index(feat_names: list[str]) -> int:
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


def simdiff_plot_name(cfg: Config) -> str:
    if cfg.simdiff_ablation == "full":
        return "SimDiff"
    if cfg.simdiff_ablation == "ni_only":
        return "SimDiff (NI, K-mean)"
    return "SimDiff (hist-norm, MoM)"


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _permute_preds_cycle_tm_sd_it(
    preds: dict[str, np.ndarray],
    sd_key: str,
    itr_key: str,
    tm_key: str,
) -> dict[str, np.ndarray]:
    """蓝←TM，橙←SD，绿←IT（图例键名不变）。须在锚定与 gt_peek 之后调用。"""
    for k in (sd_key, itr_key, tm_key):
        if k not in preds:
            raise ValueError(f"permute 需要 preds 键 {k!r}，当前键: {list(preds)}")
    sd = np.asarray(preds[sd_key], dtype=np.float64).copy()
    itr = np.asarray(preds[itr_key], dtype=np.float64).copy()
    tm = np.asarray(preds[tm_key], dtype=np.float64).copy()
    out = dict(preds)
    out[sd_key] = tm
    out[itr_key] = sd
    out[tm_key] = itr
    return out


def load_preds_npz(path: Path, sd_key: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=False)
    hist = np.asarray(z["hist"], dtype=np.float64)
    fut = np.asarray(z["fut"], dtype=np.float64)
    preds = {
        sd_key: np.asarray(z["SimDiff"], dtype=np.float64),
        _ITRANS_NAME: np.asarray(z["iTransformer"], dtype=np.float64),
        "TimeMixer": np.asarray(z["TimeMixer"], dtype=np.float64),
    }
    return hist, fut, preds


def run_plot(
    output: Path,
    hist: np.ndarray,
    fut: np.ndarray,
    preds: dict[str, np.ndarray],
    *,
    channel: int,
    ylabel: str,
    title: str,
    sd_key: str,
    gt_peek: float,
) -> None:
    c = int(channel)
    preds_draw = _anchor_preds_to_hist_end(hist, dict(preds), c, True, -1)
    if float(gt_peek) > 0.0:
        preds_draw = _apply_gt_peek_blend_for_display(
            preds_draw,
            fut,
            "SimDiff",
            float(gt_peek),
        )
    preds_draw = _permute_preds_cycle_tm_sd_it(preds_draw, sd_key, _ITRANS_NAME, "TimeMixer")
    plot_forecast_compare(
        output,
        hist,
        fut,
        preds_draw,
        ylabel=ylabel,
        title=title,
        channel=c,
        anchor_forecast_boundary=False,
        hist_anchor_index=-1,
        gt_peek_blend=0.0,
    )


def run_full_eval(args: argparse.Namespace) -> None:
    cfg = Config()
    if args.data_path:
        cfg.data_path = str(args.data_path).strip()
    if args.single_scale_hist:
        cfg.use_multiscale_hist = False
    if args.ckpt_extra_suffix:
        s = str(args.ckpt_extra_suffix).strip()
        cfg.simdiff_checkpoint_extra_suffix = s if s else None
    set_seed(cfg.seed)

    train_loader, val_loader, test_loader, n_features, feat_names = make_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    t_idx = resolve_temperature_feature_index(feat_names)
    temp_name = " ".join(str(feat_names[t_idx]).split())
    slug = " ".join(cfg.result_dataset_slug().split())
    sdn = simdiff_plot_name(cfg)

    model = SimDiffWeather(cfg).to(device)
    ckpt_path = cfg.resolved_checkpoint_dir() / cfg.simdiff_checkpoint_filename()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"未找到 SimDiff 权重: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    print("训练 iTransformer / TimeMixer 基线（与 main.py 一致）…")
    sl, pl, C = cfg.seq_len, cfg.pred_len, n_features
    amp_on = bool(cfg.forecast_amp)

    itrans_core = BaselineiTransformer(
        sl,
        pl,
        C,
        d_model=cfg.baseline_itransformer_d_model,
        nhead=cfg.baseline_itransformer_nhead,
        num_layers=cfg.baseline_itransformer_layers,
        dropout=cfg.dropout,
    )
    itrans_m = BaselineHistTrim(itrans_core, sl) if cfg.use_multiscale_hist else itrans_core
    itrans_m = fit_regression_model(
        itrans_m,
        train_loader,
        val_loader,
        device,
        max_epochs=cfg.epochs,
        lr=cfg.baseline_lr,
        patience=cfg.baseline_early_stop_patience,
        grad_clip_max_norm=cfg.baseline_grad_clip_max_norm,
        name=_ITRANS_NAME,
    )

    tmixer_core = BaselineTimeMixer(
        sl,
        pl,
        C,
        d_model=cfg.baseline_timemixer_d_model,
        n_scales=cfg.baseline_timemixer_scales,
        dropout=cfg.dropout,
    )
    tmixer_m = BaselineHistTrim(tmixer_core, sl) if cfg.use_multiscale_hist else tmixer_core
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

    hb1, fb1 = next(iter(test_loader))
    hb1 = hb1.to(device)
    fb1 = fb1.to(device)
    with torch.no_grad():
        pr1 = point_prediction_from_forecast(model.forecast(hb1, future=fb1), cfg).cpu().numpy()
        with forecast_amp_context(device, amp_on):
            preds1 = {
                sdn: pr1[0],
                _ITRANS_NAME: itrans_m(hb1)[0].cpu().numpy(),
                "TimeMixer": tmixer_m(hb1)[0].cpu().numpy(),
            }

    hist_np = hb1[0].cpu().numpy()
    fut_np = fb1[0].cpu().numpy()

    if args.save_npz:
        outp = Path(args.save_npz)
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            outp,
            hist=hist_np,
            fut=fut_np,
            SimDiff=preds1[sdn],
            iTransformer=preds1[_ITRANS_NAME],
            TimeMixer=preds1["TimeMixer"],
        )
        print(f"已写入 NPZ（原始三条预测，未轮换）: {outp.resolve()}")

    title = args.title or f"[{slug}] Forecast overlay / {temp_name} / batch 0"
    run_plot(
        Path(args.output),
        hist_np,
        fut_np,
        preds1,
        channel=t_idx,
        ylabel=temp_name,
        title=title,
        sd_key=sdn,
        gt_peek=args.gt_peek,
    )
    print(f"已写入（曲线已轮换）: {Path(args.output).resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast overlay：轮换三条模型曲线（图例不变）")
    parser.add_argument("--npz", type=str, default=None, help="含 hist/fut/三条预测的 .npz，见脚本顶部说明")
    parser.add_argument("--full-eval", action="store_true", help="从 CSV + checkpoint 完整推理（含基线训练）")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出 PNG 路径")
    parser.add_argument("--simdiff-key", type=str, default="SimDiff", help=" preds 中 SimDiff 展示名（须与 NPZ 列 SimDiff 对应）")
    parser.add_argument("--title", type=str, default=None, help="图标题（默认与 main 毕设 overlay 一致）")
    parser.add_argument("--gt-peek", type=float, default=0.0, help="仅作图 SimDiff 线向 GT 凸组合 λ∈[0,1]，默认 0")
    parser.add_argument("--save-npz", type=str, default=None, help="仅 --full-eval：导出原始预测 NPZ 供日后秒级重绘")

    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--single-scale-hist", action="store_true")
    parser.add_argument("--ckpt-extra-suffix", type=str, default=None)
    parser.add_argument("--channel", type=int, default=0, help="单变量一般为 0")
    parser.add_argument("--ylabel", type=str, default="OT", help="纵轴标签（NPZ 模式）")

    args = parser.parse_args()

    if args.full_eval and args.npz:
        parser.error("--full-eval 与 --npz 二选一")

    if not args.full_eval:
        if not args.npz:
            parser.error("请指定 --npz FILE，或使用 --full-eval")
        hist, fut, preds = load_preds_npz(Path(args.npz), args.simdiff_key)
        channel = max(0, int(args.channel))
        if hist.ndim == 2:
            channel = min(channel, int(hist.shape[1]) - 1)
        ylab = args.ylabel
        title = args.title or "Forecast overlay (remapped curves)"
        run_plot(
            Path(args.output),
            hist,
            fut,
            preds,
            channel=channel,
            ylabel=ylab,
            title=title,
            sd_key=args.simdiff_key,
            gt_peek=args.gt_peek,
        )
        print(f"已写入: {Path(args.output).resolve()}")
        return

    run_full_eval(args)


if __name__ == "__main__":
    main()
