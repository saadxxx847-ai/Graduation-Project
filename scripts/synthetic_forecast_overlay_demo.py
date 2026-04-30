#!/usr/bin/env python3
"""
独立演示脚本：**不导入 `main.py`**，仅依赖 `Config`、`make_loaders`、`plot_forecast_compare`，
与训练/评估入口解耦。

从真实 CSV 读出与项目一致的滑动窗（history + GT），再给三条合成预测曲线，用于示意配图。

用法（在项目根目录）：
  python scripts/synthetic_forecast_overlay_demo.py
  python scripts/synthetic_forecast_overlay_demo.py --out wind/forecast_overlay_synthetic_demo.png --seed 42

说明：
  - GT/history 的数据管线与仓库 `make_loaders` 一致；三线预测为人工合成，不是模型输出。
  - 正文/答辩中须标注 synthetic / illustration。
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

from config.config import Config
from utils.compare_viz import plot_forecast_compare
from utils.data_loader import make_loaders

# 与 `main.py` 中 overlay 图示约定一致：`plot_forecast_compare` 的 hist 锚在最后 token。
_HIST_ANCHOR_FOR_OVERLAY = -1


def _resolve_primary_channel(feat_names: list[str]) -> int:
    """与 `main.resolve_temperature_feature_index` 等价逻辑（本地化，避免 import main）。"""
    if len(feat_names) <= 1:
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


def _fetch_loader_batch(test_loader: torch.utils.data.DataLoader, batch_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    bi = max(0, int(batch_index))
    last_j = -1
    for j, (h, f) in enumerate(test_loader):
        last_j = j
        if j == bi:
            return h, f
    raise IndexError(f"batch_index={bi} out of range: test_loader only has {last_j + 1} batches")


def _ar1_noise(
    lf: int, rng: np.random.Generator, scale: float, rho: float
) -> np.ndarray:
    eps = np.zeros(lf, dtype=np.float64)
    z = rng.standard_normal(lf).astype(np.float64)
    s = scale * np.sqrt(max(1.0 - rho * rho, 1e-6))
    eps[0] = scale * z[0]
    for t in range(1, lf):
        eps[t] = rho * eps[t - 1] + s * z[t]
    return eps


def build_synthetic_preds(
    gt: np.ndarray,
    hist_tail: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lf = int(gt.shape[0])
    t = np.linspace(0.0, 1.0, lf, dtype=np.float64)

    err_sd = (
        _ar1_noise(lf, rng, scale=2.3, rho=0.76)
        + 0.45 * np.sin(np.linspace(0.0, 3.1 * np.pi, lf))
    )
    pred_simdiff = np.clip(gt.astype(np.float64) + err_sd, 0.0, 100.0)

    k = np.array([0.22, 0.56, 0.22], dtype=np.float64)
    smoothed = np.convolve(gt, k, mode="same")
    edge_bias = np.clip(hist_tail * 0.12 + smoothed.mean() * 0.08, 0.5, 8.0)
    systematic_tm = edge_bias + 14.0 * (1.0 - t) ** 1.65
    pred_tm = (
        smoothed
        + systematic_tm
        + rng.normal(0.0, 2.9, lf)
        + 0.85 * np.sin(np.linspace(0.0, 8.0, lf))
    )
    pred_tm = np.clip(pred_tm, 0.0, 100.0)

    chirp = 5.8 * np.sin(np.linspace(0.0, 6.8 * np.pi, lf))
    lag = np.concatenate([[gt[0]], gt[:-1]])
    blended = (
        0.58 * gt + 0.32 * lag + 0.10 * pred_tm + 10.5 * np.exp(-3.9 * t) + chirp * 0.35
    )
    pred_itr = np.clip(blended + rng.normal(0.0, 3.2, lf), 0.0, 100.0)

    return pred_simdiff, pred_itr, pred_tm


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic overlay demo (standalone from main.py)")
    ap.add_argument("--data_path", type=str, default="data/wind.csv", help="CSV 路径")
    ap.add_argument("--batch", type=int, default=0, metavar="J", help="test_loader batch index (0-based)")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed for synthetic noise")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG (default: <resolved_result_dir>/forecast_curves_overlay_synthetic_demo.png)",
    )
    args = ap.parse_args()

    cfg = Config()
    cfg.data_path = str(args.data_path).strip()

    _, _, test_loader, _nfeat, feat_names = make_loaders(cfg)
    hb, fb = _fetch_loader_batch(test_loader, int(args.batch))
    hist_np = hb[0].detach().cpu().numpy()
    true_np = fb[0].detach().cpu().numpy()
    lf, C = int(true_np.shape[0]), int(true_np.shape[1])
    ch = _resolve_primary_channel(feat_names)

    rng = np.random.default_rng(int(args.seed))
    gt = np.asarray(true_np[:, ch], dtype=np.float64).ravel()
    hist_tail = float(hist_np[-1, ch])

    sd, itr, tm = build_synthetic_preds(gt, hist_tail, rng)
    preds: dict[str, np.ndarray] = {}
    for nm, vec in (("SimDiff", sd), ("iTransformer", itr), ("TimeMixer", tm)):
        full = np.zeros((lf, C), dtype=np.float64)
        full[:, ch] = vec
        preds[nm] = full

    slug = cfg.result_dataset_slug()
    out = Path(args.out) if args.out else (cfg.resolved_result_dir() / "forecast_curves_overlay_synthetic_demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    ylab = str(feat_names[ch]) if ch < len(feat_names) else "OT"
    plot_forecast_compare(
        out,
        hist_np.copy(),
        true_np.copy(),
        preds,
        ylabel=ylab,
        title=f"[{slug}] Forecast overlay · {ylab} · synthetic preds (illus.) · batch {args.batch}",
        channel=ch,
        hist_anchor_index=_HIST_ANCHOR_FOR_OVERLAY,
    )

    maes = {name: float(np.mean(np.abs(preds[name][:, ch] - gt))) for name in preds}
    print(f"Wrote {out.resolve()} | synthetic preds only; hist/GT from data pipeline.")
    print("Mean absolute error vs GT:")
    for k in sorted(maes, key=lambda x: maes[x]):
        print(f"  {k:14s} {maes[k]:.4f}")


if __name__ == "__main__":
    main()
