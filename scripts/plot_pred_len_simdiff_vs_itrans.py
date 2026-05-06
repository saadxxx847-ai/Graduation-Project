#!/usr/bin/env python3
"""
SimDiff 与 iTransformer 在同一 pred_len 网格上的 MAE/MSE 双轴对比图。
不加载模型；数据默认为仓库最新对照表（可按 CLI 覆盖）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.compare_viz import plot_pred_len_dual_model_compare

import numpy as np

PRED_LENS = np.array([48, 72, 168, 192], dtype=np.float64)
# weather · test · T degC（与用户汇总表一致；SimDiff 即表中 SimDiff-improved）
SIMDIFF_MAE = np.array([0.504303, 0.640507, 1.019193, 1.094749])
SIMDIFF_MSE = np.array([0.571831, 0.888522, 2.124678, 2.445825])
ITRANS_MAE = np.array([0.542749, 0.698412, 1.125037, 1.203108])
ITRANS_MSE = np.array([0.623547, 0.979638, 2.364921, 2.710816])


def main() -> None:
    p = argparse.ArgumentParser(description="SimDiff vs iTransformer · MAE/MSE vs pred_len（双轴合图）")
    p.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "reborn" / "mae_mse_vs_pred_len_simdiff_vs_itrans_weather.png"),
        help="输出 png（相对项目根）",
    )
    p.add_argument(
        "--title",
        type=str,
        default="[weather] Test MAE / MSE vs pred_len · SimDiff vs iTransformer (test, T degC)",
    )
    args = p.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    plot_pred_len_dual_model_compare(
        out,
        pred_lens=PRED_LENS,
        maes_a=SIMDIFF_MAE,
        mses_a=SIMDIFF_MSE,
        maes_b=ITRANS_MAE,
        mses_b=ITRANS_MSE,
        label_a="SimDiff",
        label_b="iTransformer",
        title=args.title,
    )
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
