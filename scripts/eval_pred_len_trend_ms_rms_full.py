#!/usr/bin/env python3
"""
在不同预测长度 pred_len 上评估「SimDiff multiscale + RMSNorm」（ms_rms full）的测试集精度并绘制趋势图。

重要：`pred_len` 参与网络构造（未来 token 数），**每个长度必须使用单独训练得到的权重**。
仅持有 `simdiff_weather_best_ms_rms_full.pt`（对应训练时的某个 pred_len，默认 24）时，
无法在同一权重下合法评估 pred_len=48/72/…。

默认权重路径模板（训练完成后请将对应 checkpoint 复制或命名为）：
  checkpoints/simdiff_weather_best_ms_rms_full_pl{pred_len}.pt

示例（默认写出**一张**双轴折线图到项目根下 **length/**，与 main 训练自带的柱状图/overlay 无关）：
  python scripts/eval_pred_len_trend_ms_rms_full.py --pred_lens 48,72,168,192
  python scripts/eval_pred_len_trend_ms_rms_full.py --out_png length/custom.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from config.config import Config
from main import (
    _apply_ms_rms_key,
    _clear_ms_rms_key,
    evaluate_test_loader,
    resolve_temperature_feature_index,
)
from models.simdiff import SimDiffWeather
from utils.compare_viz import plot_pred_len_accuracy_trend
from utils.data_loader import make_loaders


def _parse_pred_lens(s: str) -> list[int]:
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return [max(1, int(x)) for x in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="MAE/MSE vs pred_len for ms_rms full")
    parser.add_argument(
        "--pred_lens",
        type=str,
        default="48,72,168,192",
        help="逗号分隔，如 48,72,168,192",
    )
    parser.add_argument(
        "--ckpt_format",
        type=str,
        default="simdiff_weather_best_ms_rms_full_pl{}.pt",
        help="相对 --ckpt_dir；{} 替换为 pred_len",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints",
        help="checkpoint 目录（相对项目根）",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default=None,
        help="输出 png；默认 length/mae_mse_vs_pred_len_ms_rms_full_<时间戳>.png（项目根下 length/）",
    )
    args = parser.parse_args()

    pred_lens = _parse_pred_lens(args.pred_lens)
    cfg = Config()
    cfg.validate_mom_config()
    cfg.validate_training_noise_objective()
    cfg.validate_simdiff_ablation()
    cfg.validate_denoiser_embedding_options()

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    ckpt_root = ROOT / args.ckpt_dir

    rows: list[tuple[int, float, float]] = []
    for pl in pred_lens:
        cfg.pred_len = pl
        cfg.validate_mom_config()
        _apply_ms_rms_key(cfg, "full")
        ckpt_path = ckpt_root / args.ckpt_format.format(pl)
        if not ckpt_path.is_file():
            print(f"[skip] pred_len={pl}: 缺少权重 {ckpt_path}")
            _clear_ms_rms_key(cfg)
            continue
        train_loader, val_loader, test_loader, n_features, feat_names = make_loaders(cfg)
        model = SimDiffWeather(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        _mse, _mae, mae_ch, mse_ch = evaluate_test_loader(model, test_loader, device, n_features)
        t_idx = resolve_temperature_feature_index(feat_names)
        rows.append((pl, float(mae_ch[t_idx]), float(mse_ch[t_idx])))
        print(f"pred_len={pl}: MAE={mae_ch[t_idx]:.6f} MSE={mse_ch[t_idx]:.6f} ({ckpt_path.name})")
        _clear_ms_rms_key(cfg)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        print("无任何可用权重，退出。")
        sys.exit(1)

    xs = [r[0] for r in rows]
    maes = [r[1] for r in rows]
    mses = [r[2] for r in rows]

    slug = cfg.result_dataset_slug()
    length_dir = ROOT / "length"
    out = (
        length_dir / cfg.result_png_basename("mae_mse_vs_pred_len_ms_rms_full")
        if args.out_png is None
        else Path(args.out_png)
    )
    if not out.is_absolute():
        out = ROOT / out
    plot_pred_len_accuracy_trend(
        out,
        xs,
        maes,
        mses,
        curve_label="SimDiff",
        title=f"[{slug}] Test MAE / MSE vs prediction length",
    )
    print(f"[saved] {out}")

    print("\n--- pred_len sweep (test) ---")
    print(f"{'pred_len':>8} {'MAE':>14} {'MSE':>14}")
    print("-" * 40)
    for pl, ma, ms in zip(xs, maes, mses):
        print(f"{pl:>8} {ma:>14.6f} {ms:>14.6f}")
    print()


if __name__ == "__main__":
    main()
