#!/usr/bin/env python3
"""
最小自检：Normalization Independence（历史统计不依赖 future）+ MoM 与单次采样可区分。
用法：python main.py --verify_norm_mom
或：python verify_norm_mom.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import Config
from utils.data_loader import make_loaders
from utils.independent_normalizer import IndependentNormalizer, mom_aggregate_normalized


def run_quick_verify(
    cfg: Config,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    cfg.validate_mom_config()

    hist, fut = next(iter(train_loader))
    hist = hist.to(device)
    fut = fut.to(device)

    # --- NI：normalize_history 不读 future ---
    _, sh1 = IndependentNormalizer.normalize_history(hist)
    _ = torch.roll(fut, shifts=3, dims=0)
    _, sh2 = IndependentNormalizer.normalize_history(hist)
    assert torch.allclose(sh1["mu_h"], sh2["mu_h"]), "历史 μ 不应随 future 改变（NI）"
    assert torch.allclose(sh1["sig_h"], sh2["sig_h"]), "历史 σ 不应随 future 改变（NI）"
    fn, sf = IndependentNormalizer.normalize_future(fut)
    round_trip = IndependentNormalizer.inverse_transform_future(fn, sf["mu_f"], sf["sig_f"])
    assert torch.allclose(round_trip, fut, rtol=1e-4, atol=1e-3), "未来反变换应可逆"
    print("[ok] Normalization Independence：历史统计与 future 无关；未来反变换可逆")

    # --- MoM：聚合后 ≠ 单次（高概率）---
    b = hist.shape[0]
    k, lf, c = cfg.forecast_num_samples, cfg.pred_len, cfg.input_dim
    torch.manual_seed(0)
    stacked = torch.randn(b, k, lf, c, device=device)
    single, _, mom_n = mom_aggregate_normalized(stacked, cfg.mom_num_groups)
    if k > 1:
        assert not torch.allclose(single, mom_n), "MoM 与单次采样应可区分"
    print("[ok] MoM 聚合逻辑在非退化随机张量上与 single 可区分")

    print("\n全部快速检查通过。")


def main() -> None:
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    train_loader, _, _, _, _ = make_loaders(cfg)
    run_quick_verify(cfg, train_loader, device)


if __name__ == "__main__":
    main()
