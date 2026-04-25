#!/usr/bin/env python3
"""
为毕业论文生成「5 数据集 × 1 张」真实值 vs 多模型预测曲线（DLinear、iTransformer、mr-Diff、SimDiff、改进版 SimDiff）。

前置：已训练好各 checkpoints（与 config.simdiff_checkpoint_filename 命名一致）；
会现场训练 DLinear / iTransformer（与 main.py 相同设置）。仅推理图，不覆盖 metrics JSON。

用法（项目根目录）:
  python scripts/render_thesis_curves.py --pred_len 168 --out_subdir pred_len_168
  python scripts/render_thesis_curves.py --pred_len 48 --out_subdir pred_len_48
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import Config
from models.simdiff import SimDiffWeather, point_prediction_from_forecast
from utils.baselines import (
    DLinearMap,
    ITransformer,
    fit_regression_model,
)
from utils.compare_viz import plot_forecast_compare
from utils.data_loader import make_loaders

# 与 paper_output.PRESET_TABLE_LABEL 一致
_PRESETS = ["etth1", "ettm1", "exchange", "weather", "wind"]
_LABELS = {
    "etth1": "ETTh1(OT)",
    "ettm1": "ETTm1(OT)",
    "exchange": "Exchange(USD)",
    "weather": "Weather(Temp)",
    "wind": "Wind(Speed)",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_sibling(
    base_cfg: Config, device: torch.device, *, mrdiff: bool, use_patch: bool, use_rope: bool
) -> SimDiffWeather | None:
    c = copy.deepcopy(base_cfg)
    c.mrdiff_denoiser = mrdiff
    c.use_patch = use_patch
    c.use_rope = use_rope
    c.validate_arch()
    p = c.resolved_checkpoint_dir() / c.simdiff_checkpoint_filename()
    if not p.is_file():
        print(f"  [skip] 无权重: {p.name}", file=sys.stderr)
        return None
    st = torch.load(p, map_location=device, weights_only=False)
    meta = st.get("meta") or {}
    if "timesteps" in meta:
        c.timesteps = int(meta["timesteps"])
    if meta.get("sampling_steps") is not None:
        c.sampling_steps = int(meta["sampling_steps"])
    m = SimDiffWeather(c).to(device)
    m.load_state_dict(st["model"], strict=True)
    m.eval()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_len", type=int, default=168)
    ap.add_argument(
        "--out_subdir",
        type=str,
        default="pred_len_168",
        help="相对 outputs/paper/01_fitting_curves/ 的子目录，避免不同 pred_len 覆盖",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="outputs/paper/01_fitting_curves",
        help="拟合曲线根目录",
    )
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out = Path(args.out_root) / args.out_subdir
    out.mkdir(parents=True, exist_ok=True)

    for preset in _PRESETS:
        cfg = Config()
        cfg.data_preset = preset
        cfg.pred_len = int(args.pred_len)
        cfg.univariate = True
        cfg.mrdiff_denoiser = False
        cfg.use_patch = False
        cfg.use_rope = False
        set_seed(cfg.seed)
        cfg.validate_mom_config()
        cfg.validate_simdiff_ablation()
        cfg.validate_arch()

        train_loader, val_loader, test_loader, n_features, feat_names = make_loaders(cfg)
        c_vis = 0
        ylabel = feat_names[c_vis] if feat_names else "target"

        # 基线：与 main 一致
        dlm = DLinearMap(cfg.seq_len, cfg.pred_len, n_features)
        dlm = fit_regression_model(
            dlm,
            train_loader,
            val_loader,
            device,
            max_epochs=cfg.baseline_max_epochs,
            lr=cfg.baseline_lr,
            patience=cfg.baseline_early_stop_patience,
            grad_clip_max_norm=cfg.baseline_grad_clip_max_norm,
            name="DLinear",
        )
        itrans = ITransformer(
            cfg.seq_len,
            cfg.pred_len,
            n_features,
            d_model=cfg.baseline_transformer_d_model,
            nhead=cfg.baseline_transformer_nhead,
            num_layers=cfg.baseline_transformer_layers,
            dropout=cfg.dropout,
        )
        itrans = fit_regression_model(
            itrans,
            train_loader,
            val_loader,
            device,
            max_epochs=cfg.baseline_max_epochs,
            lr=cfg.baseline_lr,
            patience=cfg.baseline_early_stop_patience,
            grad_clip_max_norm=cfg.baseline_grad_clip_max_norm,
            name="iTransformer",
        )

        c_base = copy.deepcopy(cfg)
        m_mr = _load_sibling(c_base, device, mrdiff=True, use_patch=False, use_rope=False)
        m_sd = _load_sibling(c_base, device, mrdiff=False, use_patch=False, use_rope=False)
        c_ours = copy.deepcopy(cfg)
        c_ours.use_patch = True
        c_ours.use_rope = True
        c_ours.validate_arch()
        m_our = _load_sibling(c_ours, device, mrdiff=False, use_patch=True, use_rope=True)

        hist_b, fut_b = next(iter(test_loader))
        hist_b = hist_b.to(device)
        fut_b = fut_b.to(device)

        t_hist = np.arange(cfg.seq_len)
        t_fut = np.arange(cfg.seq_len, cfg.seq_len + cfg.pred_len)
        hist0 = hist_b[0].cpu().numpy()
        true0 = fut_b[0].cpu().numpy()

        preds: dict[str, np.ndarray] = {}
        with torch.no_grad():
            preds["DLinear"] = dlm(hist_b[:1]).cpu().numpy()[0]
            preds["iTransformer"] = itrans(hist_b[:1]).cpu().numpy()[0]
            if m_mr is not None:
                preds["mr-Diff"] = point_prediction_from_forecast(
                    m_mr.forecast(hist_b[:1], future=fut_b[:1]), m_mr.cfg
                ).cpu().numpy()[0]
            if m_sd is not None:
                preds["SimDiff"] = point_prediction_from_forecast(
                    m_sd.forecast(hist_b[:1], future=fut_b[:1]), m_sd.cfg
                ).cpu().numpy()[0]
            if m_our is not None:
                preds["改进版 SimDiff"] = point_prediction_from_forecast(
                    m_our.forecast(hist_b[:1], future=fut_b[:1]), m_our.cfg
                ).cpu().numpy()[0]

        title = (
            f"{_LABELS.get(preset, preset)} — 真实值 vs 多模型 (pred_len={cfg.pred_len})"
        )
        fp = out / f"{preset}_true_vs_all_models.png"
        plot_forecast_compare(
            fp,
            t_hist,
            t_fut,
            hist0,
            true0,
            preds,
            ylabel=ylabel,
            title=title,
        )
        print(f"已写: {fp}")

    print("完成。若某扩散模型未训练，该曲线不会包含对应条目（见上方 [skip]）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
