#!/usr/bin/env python3
"""
独立脚本：根据手工整理的测试集指标表绘制「pred_len vs MAE/MSE」双轴折线图。
不 import 项目内模型、data_loader、main；仅 matplotlib + pathlib。

数据来源：ms_rms full、weather · T(degC)、各 pred_len 训练后终端 1-fold 表（48→72→168→192 自上而下）。
修改下方常量或传 --csv 即可重画，不影响训练/评估流程。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 与用户提供终端表一致（pred_len 48, 72, 168, 192）
DEFAULT_PRED_LENS = (48, 72, 168, 192)
DEFAULT_MAES = (0.512093, 0.654007, 1.043893, 1.120749)
DEFAULT_MSES = (0.588631, 0.916522, 2.190638, 2.521685)


def _load_csv(path: Path) -> tuple[list[int], list[float], list[float]]:
    """CSV 三列无表头：pred_len,mae,mse"""
    rows: list[tuple[int, float, float]] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        a, b, c = [x.strip() for x in line.split(",")]
        rows.append((int(a), float(b), float(c)))
    rows.sort(key=lambda t: t[0])
    xs = [r[0] for r in rows]
    ma = [r[1] for r in rows]
    ms = [r[2] for r in rows]
    return xs, ma, ms


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    length_dir = root / "length"
    parser = argparse.ArgumentParser(description="Standalone MAE/MSE vs pred_len plot → length/")
    parser.add_argument(
        "--out",
        type=str,
        default=str(length_dir / "mae_mse_vs_pred_len_manual.png"),
        help="输出 png 路径（默认 length/mae_mse_vs_pred_len_manual.png）",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="可选：三列 pred_len,mae,mse（无表头），覆盖默认内置表",
    )
    parser.add_argument(
        "--curve-label",
        type=str,
        default="SimDiff",
        help="图例前缀（默认 SimDiff；如 iTransformer 基线曲线）",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="[weather] Test MAE / MSE vs prediction length (ms_rms full, T degC)",
        help="图标题",
    )
    args = parser.parse_args()

    if args.csv:
        xs, maes, mses = _load_csv(Path(args.csv))
    else:
        xs = list(DEFAULT_PRED_LENS)
        maes = list(DEFAULT_MAES)
        mses = list(DEFAULT_MSES)

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)

    x_arr = np.asarray(xs, dtype=np.float64)
    ma = np.asarray(maes, dtype=np.float64)
    ms = np.asarray(mses, dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    (ln1,) = ax1.plot(
        x_arr, ma, marker="o", linewidth=1.6, label=f"{args.curve_label} · MAE", color="C0"
    )
    ax1.set_xlabel("Prediction length (steps)")
    ax1.set_ylabel("MAE", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    (ln2,) = ax2.plot(
        x_arr,
        ms,
        marker="s",
        linewidth=1.4,
        linestyle="--",
        label=f"{args.curve_label} · MSE",
        color="C1",
    )
    ax2.set_ylabel("MSE", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_xticks(x_arr)
    ax1.grid(True, alpha=0.28)
    ax1.set_title(args.title, fontsize=10)
    ax1.legend(handles=[ln1, ln2], loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
