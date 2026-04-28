"""
毕设「result/」输出：终端指标表、与 plots/ 隔离的数据集子目录。
"""
from __future__ import annotations

from typing import Sequence


def print_metrics_ascii_table(
    rows: Sequence[tuple[str, float, float, str, str]],
    *,
    headline: str | None = None,
    footer_notes: Sequence[str] | None = None,
) -> None:
    """
    与 FiLM / dual 等消融终端一致的简易表：Model | MAE | MSE | CRPS | VAR（无「毕设指标表」横幅）。
    """
    print()
    if headline:
        print(headline)
    w_model = max(38, max((len(str(r[0])) for r in rows), default=0) + 2)
    hdr = f"{'Model':<{w_model}} {'MAE':>12} {'MSE':>12} {'CRPS':>12} {'VAR':>12}"
    print(hdr)
    sep_len = max(len(hdr), 72)
    print("-" * sep_len)
    for name, mae, mse, crps_s, var_s in rows:
        print(f"{str(name):<{w_model}} {mae:>12.6f} {mse:>12.6f} {crps_s:>12} {var_s:>12}")
    print("-" * sep_len)
    if footer_notes:
        for line in footer_notes:
            print(f"  {line}")
    print()


def print_thesis_metrics_table(
    rows: Sequence[tuple[str, float, float, str, str]],
    dataset_label: str,
    footer_notes: Sequence[str] | None = None,
) -> None:
    """
    rows: (模型名, MAE, MSE, CRPS 字符串, VAR 字符串)。
    点预测模型：CRPS 与 MAE 相同（退化预报），VAR=0。
    footer_notes: 若给定则只打印这些说明行；否则打印默认 SimDiff vs 基线说明。
    """
    print()
    print(f"========== 毕设指标表（{dataset_label}）==========")
    w_model = max(38, max((len(str(r[0])) for r in rows), default=0) + 2)
    hdr = f"{'Model':<{w_model}} {'MAE':>12} {'MSE':>12} {'CRPS':>12} {'VAR':>12}"
    print(hdr)
    print("-" * max(len(hdr), 72))
    for name, mae, mse, crps_s, var_s in rows:
        print(f"{str(name):<{w_model}} {mae:>12.6f} {mse:>12.6f} {crps_s:>12} {var_s:>12}")
    print("-" * max(len(hdr), 72))
    if footer_notes:
        for line in footer_notes:
            print(f"  {line}")
    else:
        print("  SimDiff: CRPS from K-sample ensemble; VAR = mean sample variance over (batch, horizon).")
        print("  iTransformer/TimeMixer: point forecasts; CRPS equals MAE; VAR = 0.")
    print()
