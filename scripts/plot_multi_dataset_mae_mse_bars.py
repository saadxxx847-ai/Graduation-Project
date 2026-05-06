#!/usr/bin/env python3
"""
按数据集生成 SimDiff / iTransformer / TimeMixer 的 MAE+MSE  grouped 柱状图。
数据来自用户汇总表（SimDiff-improved 在图中标注为 SimDiff）；写入 reborn 子目录。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.compare_viz import plot_metrics_bars

# x 轴顺序与参考图一致
MODEL_NAMES = ["SimDiff", "iTransformer", "TimeMixer"]

# (title, filename_stem, maes, mses) — MAE/MSE 与用户提供表一致；Wind 的 iTransformer MSE 按表 1.11232
DATASETS: list[tuple[str, str, list[float], list[float]]] = [
    (
        "[weather] Test MAE/MSE: T (degC)",
        "weather_T_degC",
        [0.372570, 0.401271, 0.410681],
        [0.340968, 0.376830, 0.389550],
    ),
    (
        "[ETTh1] Test MAE/MSE: OT",
        "etth1_OT",
        [0.904211, 0.968436, 0.986971],
        [1.448390, 1.569250, 1.613403],
    ),
    (
        "[ETTm1] Test MAE/MSE: OT",
        "ettm1_OT",
        [0.400960, 0.427966, 0.439361],
        [0.387689, 0.419085, 0.435856],
    ),
    (
        "[Wind] Test MAE/MSE: OT",
        "wind_OT",
        [0.631769, 0.679519, 0.671293],
        [1.024573, 1.112320, 1.098696],
    ),
    (
        "[Exchange_rate] Test MAE/MSE: OT",
        "exchange_rate_OT",
        [0.126091, 0.133686, 0.137491],
        [0.027167, 0.028870, 0.029855],
    ),
]


def main() -> None:
    out_dir = ROOT / "reborn" / "multi_dataset_mae_mse"
    out_dir.mkdir(parents=True, exist_ok=True)
    for title, stem, maes, mses in DATASETS:
        path = out_dir / f"bar_mae_mse_{stem}.png"
        plot_metrics_bars(
            path,
            names=list(MODEL_NAMES),
            maes=maes,
            mses=mses,
            title=title,
            ylabel="MAE/MSE",
            ylabel_rotation=90.0,
        )
        print(path.resolve())


if __name__ == "__main__":
    main()
