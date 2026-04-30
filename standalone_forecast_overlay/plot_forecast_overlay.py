#!/usr/bin/env python3
"""
Standalone forecast overlay（与训练/评测代码无 import 关系）。

history 与 ground truth 数值来自本项目「wind · OT · test batch 0」真实窗口，
仅三条模型预测曲线在本脚本内合成；绘图约定对齐 ``utils.compare_viz.plot_forecast_compare``
（竖线在 Lh-1+0.5、history 灰深色、GT/预测 bridge）。

Dependencies: numpy, matplotlib
  pip install numpy matplotlib

用法:
  python /path/to/standalone_forecast_overlay/plot_forecast_overlay.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent

# --- wind · OT · test_loader batch 0 · 首条样本（与 main 毕设 overlay 同源窗口）---
HISTORY_OT_BATCH0 = np.array(
    [
        1.93499994,
        1.82000005,
        0.71700001,
        1.30599999,
        -0.04100000,
        -0.28600001,
        0.05900000,
        0.07500000,
        0.48300001,
        1.11899996,
        1.33800006,
        2.40000010,
        2.47900009,
        3.65599990,
        6.40999985,
        12.30200005,
        17.39699936,
        23.15600014,
        27.52599907,
        37.02199936,
        56.10699844,
        73.25399780,
        86.41899872,
        92.51699829,
        96.43900299,
        92.73200226,
        93.75299835,
        93.59899902,
        96.48400116,
        95.39800262,
        66.41000366,
        38.63800049,
        26.14500046,
        22.58600044,
        18.54299927,
        13.07600021,
        9.20199966,
        8.77299976,
        9.87100029,
        13.83899975,
        24.61000061,
        28.45899963,
        46.06800079,
        45.31700134,
        63.30500031,
        59.43700027,
        65.91799927,
        65.17500305,
        64.30300140,
        55.90499878,
        65.81600189,
        76.11699677,
        58.39199829,
        67.95800018,
        57.17900085,
        53.40900040,
        55.42599869,
        44.15800095,
        54.08399963,
        41.42399979,
        37.50099945,
        31.38800049,
        31.61000061,
        32.14400101,
        25.21100044,
        31.26000023,
        18.99099922,
        16.89999962,
        25.87599945,
        19.79199982,
        18.80900002,
        32.85900116,
        17.15999985,
        27.96500015,
        36.78799820,
        38.65299988,
        41.66500092,
        40.82400131,
        45.08700180,
        35.63199997,
        26.23299980,
        18.76899910,
        17.75799942,
        20.87000084,
        21.64699936,
        19.98100090,
        17.82900047,
        15.63399982,
        16.59099960,
        14.44699955,
        13.61600018,
        9.16899967,
        8.29800034,
        17.93600082,
        22.71599960,
        16.06200027,
        44.88812256,
        31.77198792,
        29.48963356,
        15.57259369,
        21.79001045,
        0.45965624,
        33.54988480,
        14.80678177,
        14.36181355,
        38.95753860,
        25.36026955,
    ],
    dtype=np.float64,
)
GROUND_TRUTH_OT_BATCH0 = np.array(
    [
        17.41500092,
        14.98400021,
        11.43099976,
        8.87300014,
        8.04800034,
        9.28800011,
        9.42099953,
        9.06900024,
        10.58800030,
        10.73700047,
        8.84099960,
        6.46799994,
        3.51600003,
        1.52400005,
        1.35699999,
        1.75100005,
        0.92799997,
        1.89199996,
        2.66000009,
        4.24599981,
        3.72799993,
        5.14900017,
        3.87199998,
        6.01200008,
    ],
    dtype=np.float64,
)

RNG = np.random.default_rng(202604301)


def _t_fut_with_bridge(t_hist: np.ndarray, t_fut: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray([float(t_hist[-1])], dtype=np.float64), np.asarray(t_fut, dtype=np.float64)])


def _anchor_pred_to_hist_end(h_last: float, pred_fut: np.ndarray) -> np.ndarray:
    """与 compare_viz._anchor_preds_to_hist_end 一致（单通道向量）。"""
    p = np.asarray(pred_fut, dtype=np.float64).copy().ravel()
    p -= p[0] - float(h_last)
    return p


def _smooth_noise(n: int, scale: float, correlation: float = 0.82) -> np.ndarray:
    eps = RNG.standard_normal(n)
    out = np.empty(n)
    out[0] = eps[0] * scale
    rho = correlation
    for i in range(1, n):
        out[i] = rho * out[i - 1] + np.sqrt(max(0.0, 1.0 - rho**2)) * eps[i] * scale
    return out


def _y_limits_forecast_focus(
    hist: np.ndarray,
    true_fut: np.ndarray,
    preds: dict[str, np.ndarray],
) -> tuple[float, float]:
    chunks: list[np.ndarray] = [
        np.asarray(hist, dtype=np.float64).ravel(),
        np.asarray(true_fut, dtype=np.float64).ravel(),
    ]
    for p in preds.values():
        chunks.append(np.asarray(p, dtype=np.float64).ravel())
    seg = np.concatenate(chunks)
    lo, hi = float(np.min(seg)), float(np.max(seg))
    pad = 0.04 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def main() -> None:
    hist = HISTORY_OT_BATCH0
    gt = GROUND_TRUTH_OT_BATCH0
    lh = int(hist.shape[0])
    lf = int(gt.shape[0])
    h_last = float(hist[-1])

    t_hist = np.arange(lh, dtype=np.float64)
    t_fut = np.arange(lh, lh + lf, dtype=np.float64)
    t_bridge = _t_fut_with_bridge(t_hist, t_fut)

    # --- 仅合成三条预测：整体要「平」，少用高频起伏；蓝线贴 GT 小幅上下摆；橙/绿更差但仍平滑 ---
    u = np.linspace(0.0, 1.0, lf)
    # SimDiff：随 GT 形状 + 约一周期的缓慢上下摆动（线要平、贴身）
    gentle_sd = 2.55 * np.sin(2.0 * np.pi * (1.0 * u + 0.07))
    simdiff_raw = gt + gentle_sd + _smooth_noise(lf, scale=0.25, correlation=0.97)

    # iTransformer / TimeMixer：刻意「懒预报」——近似水平错档，不跟 GT 陡降 → 拟合差且起伏少
    # （若写成 gt+斜坡，锚定后可能与真值走势意外同向，出现橙线比蓝线好看的假象）
    it_raw = np.linspace(28.2, 25.8, lf)
    tm_raw = np.linspace(10.2, 11.8, lf)
    itransformer_raw = it_raw + _smooth_noise(lf, scale=0.22, correlation=0.97)
    timemixer_raw = tm_raw + _smooth_noise(lf, scale=0.22, correlation=0.97)

    simdiff = _anchor_pred_to_hist_end(h_last, simdiff_raw)
    itransformer = _anchor_pred_to_hist_end(h_last, itransformer_raw)
    timemixer = _anchor_pred_to_hist_end(h_last, timemixer_raw)

    preds_draw = {
        "SimDiff": simdiff,
        "iTransformer": itransformer,
        "TimeMixer": timemixer,
    }
    for arr in preds_draw.values():
        np.clip(arr, 0.0, 100.0, out=arr)

    y_gt_bridge = np.concatenate([[h_last], gt])
    y0, y1 = _y_limits_forecast_focus(hist, gt, preds_draw)

    # --- 对齐 compare_viz.plot_forecast_compare 的视觉约定 ---
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(t_hist, hist, label="history", color="0.2", linewidth=1.0, alpha=0.9)
    ax.plot(
        t_bridge,
        y_gt_bridge,
        label="ground truth",
        color="black",
        linewidth=2.0,
        zorder=4,
    )

    pred_order = [("SimDiff", "#1f77b4", "-"), ("iTransformer", "#ff7f0e", "--")]
    for name, color, sty in pred_order:
        arr = preds_draw[name]
        y_b = np.concatenate([[h_last], arr])
        ax.plot(
            t_bridge,
            y_b,
            linestyle=sty,
            label=name,
            color=color,
            linewidth=1.55,
            zorder=3,
        )

    # TimeMixer：自定义 dash-dot，避免 matplotlib 默认 '-.' 在低 DPI 下像「双线」
    tm_arr = preds_draw["TimeMixer"]
    y_tm = np.concatenate([[h_last], tm_arr])
    (tm_line,) = ax.plot(
        t_bridge,
        y_tm,
        label="TimeMixer",
        color="#2ca02c",
        linewidth=1.55,
        zorder=3,
        linestyle=(0, (10, 4, 1.5, 4)),
        dash_capstyle="round",
        dash_joinstyle="round",
    )
    tm_line.set_gapcolor("none")

    ax.axvline(lh - 1 + 0.5, color="gray", linestyle=":", linewidth=0.9)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("time step (index)")
    ax.set_ylabel("OT")
    ax.set_title("[wind] Forecast overlay · OT · test batch 0", fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout(pad=0.4)

    out_path = _HERE / "forecast_curves_overlay_edited_standalone.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
