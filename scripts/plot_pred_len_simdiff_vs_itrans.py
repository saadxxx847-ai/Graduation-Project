#!/usr/bin/env python3
"""
SimDiff（ms_rms full）与 iTransformer 在同一 pred_len 网格上的 MAE/MSE 双轴对比图。
不 import 模型；仅 matplotlib。数据与 `plot_pred_len_trend_manual.py` 默认表及
`length/itrans_pred_len_sweep_temp.csv` 一致。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# pred_len 自上而下与终端表一致
PRED_LENS = np.array([48, 72, 168, 192], dtype=np.float64)
# SimDiff multiscale + RMSNorm（weather · T degC，1-fold）
SIMDIFF_MAE = np.array([0.512093, 0.654007, 1.043893, 1.120749])
SIMDIFF_MSE = np.array([0.588631, 0.916522, 2.190638, 2.521685])
# iTransformer pred_len sweep（test，气温通道）
ITRANS_MAE = np.array([1.370049, 1.578696, 1.885164, 2.049172])
ITRANS_MSE = np.array([3.760047, 4.594125, 6.487474, 7.630030])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    length_dir = root / "length"
    p = argparse.ArgumentParser(description="SimDiff vs iTransformer · MAE/MSE vs pred_len")
    p.add_argument(
        "--out",
        type=str,
        default=str(length_dir / "mae_mse_vs_pred_len_simdiff_vs_itrans_v1.png"),
        help="输出 png",
    )
    p.add_argument(
        "--title",
        type=str,
        default="[weather] Test MAE / MSE vs pred_len · SimDiff vs iTransformer (T degC)",
    )
    args = p.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(9.0, 4.3))
    (l1,) = ax1.plot(
        PRED_LENS,
        SIMDIFF_MAE,
        marker="o",
        linewidth=1.7,
        label="SimDiff · MAE",
        color="C0",
    )
    (l2,) = ax1.plot(
        PRED_LENS,
        ITRANS_MAE,
        marker="^",
        linewidth=1.7,
        label="iTransformer · MAE",
        color="C2",
    )
    ax1.set_xlabel("Prediction length (steps)")
    ax1.set_ylabel("MAE")
    ax2 = ax1.twinx()
    (l3,) = ax2.plot(
        PRED_LENS,
        SIMDIFF_MSE,
        marker="s",
        linewidth=1.45,
        linestyle="--",
        label="SimDiff · MSE",
        color="C1",
    )
    (l4,) = ax2.plot(
        PRED_LENS,
        ITRANS_MSE,
        marker="d",
        linewidth=1.45,
        linestyle="--",
        label="iTransformer · MSE",
        color="C3",
    )
    ax2.set_ylabel("MSE")
    ax1.set_xticks(PRED_LENS)
    ax1.grid(True, alpha=0.28)
    ax1.set_title(args.title, fontsize=10)
    ax1.legend(handles=[l1, l2, l3, l4], loc="upper left", fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
