"""
毕设「result/」输出：终端指标表、与 plots/ 隔离的数据集子目录。
"""
from __future__ import annotations

from typing import Sequence


def print_thesis_metrics_table(
    rows: Sequence[tuple[str, float, float, str, str]],
    dataset_label: str,
) -> None:
    """
    rows: (模型名, MAE, MSE, CRPS 字符串, VAR 字符串)。
    点预测模型：CRPS 与 MAE 相同（退化预报），VAR=0。
    """
    print()
    print(f"========== 毕设指标表（{dataset_label}）==========")
    hdr = f"{'Model':<22} {'MAE':>12} {'MSE':>12} {'CRPS':>12} {'VAR':>12}"
    print(hdr)
    print("-" * max(len(hdr), 72))
    for name, mae, mse, crps_s, var_s in rows:
        print(f"{name:<22} {mae:>12.6f} {mse:>12.6f} {crps_s:>12} {var_s:>12}")
    print("-" * max(len(hdr), 72))
    print("  SimDiff: CRPS from K-sample ensemble; VAR = mean sample variance over (batch, horizon).")
    print("  iTransformer/TimeMixer: point forecasts; CRPS equals MAE; VAR = 0.")
    print()
