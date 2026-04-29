#!/usr/bin/env python3
"""
Minimal end-to-end smoke audit for NI + multiscale + SimDiff forecast.
Run from repo root:  python scripts/smoke_audit_simdiff_etth1.py --data_path data/ETTh1.csv

Requires GPU optional; uses one batch when CUDA available else CPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from config.config import Config
from models.simdiff import SimDiffWeather, point_prediction_from_forecast
from utils.data_loader import make_loaders
from utils.independent_normalizer import IndependentNormalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/ETTh1.csv")
    args = parser.parse_args()

    cfg = Config()
    cfg.data_path = args.data_path
    cfg.batch_size = 4
    cfg.test_batch_size = 4
    cfg.num_workers = 0

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _val_loader, test_loader, n_feat, feat_names = make_loaders(cfg)
    print(f"feat_names={feat_names} input_dim(C)={n_feat}")
    hist, fut = next(iter(test_loader))
    hist = hist.to(device)
    fut = fut.to(device)
    lh, lf = hist.shape[1], fut.shape[1]
    print(f"[shapes] hist={tuple(hist.shape)} fut={tuple(fut.shape)} seq_len(cfg)={cfg.seq_len} ehl={cfg.effective_hist_len()} pred_len={lf}")

    # NI stats scope
    h_cpu = hist[:1].detach().cpu()
    f_cpu = fut[:1].detach().cpu()
    hn, st_h = IndependentNormalizer.normalize_history(
        h_cpu, hist_stats_span=int(cfg.seq_len)
    )
    fn, st_f = IndependentNormalizer.normalize_future(f_cpu)
    print(
        f"[NI history] mu_h shape={tuple(st_h['mu_h'].shape)} "
        f"(stats from hist[:, :seq_len], seq_len={cfg.seq_len}); normalized full hist Lh={lh})"
    )
    print(f"[NI future] mu_f={st_f['mu_f'].squeeze().tolist()} sig_f={st_f['sig_f'].squeeze().tolist()}")

    # Multiscale: show fine tail vs pooled tail magnitude (informal)
    h0 = h_cpu.numpy()[0, :, 0]
    print(f"[fine 96] OT min/max={h0[: cfg.seq_len].min():.4f}/{h0[: cfg.seq_len].max():.4f}")
    if lh > cfg.seq_len:
        print(f"[daily+weekly tail last 11] min/max={h0[-11:].min():.4f}/{h0[-11:].max():.4f}")

    model = SimDiffWeather(cfg).to(device)
    model.train()
    loss = model.training_loss(hist[:2], fut[:2])
    loss.backward()
    g_in = (
        torch.norm(model.net.in_proj.weight.grad).item()
        if model.net.in_proj.weight.grad is not None
        else float("nan")
    )
    g_pos = (
        torch.norm(model.net.pos_h.grad).item()
        if model.net.pos_h.grad is not None
        else float("nan")
    )
    rev_g = ""
    if model.net.use_revin and getattr(model.net, "revin", None) is not None and model.net.revin is not None:
        aw = getattr(model.net.revin, "affine_weight", None)
        if aw is not None and getattr(aw, "grad", None) is not None:
            rev_g = f" revin.affine.grad_norm={torch.norm(aw.grad).item():.4g}"
    print(
        f"[grad] backward ok | in_proj.grad_norm={g_in:.4g} "
        f"pos_h.grad_norm={g_pos:.4g}{rev_g} (use_revin={cfg.use_revin})"
    )

    model.eval()
    with torch.no_grad():
        cfg.forecast_num_samples = 10
        cfg.mom_num_groups = 5
        out = model.forecast(hist[:1], future=fut[:1], return_samples=True, num_samples=10, num_groups=5)
        pred = point_prediction_from_forecast(out, cfg)
        if out.samples is not None:
            s = out.samples[0].cpu().numpy()
            v_k = float(np.var(s[:, 0, 0], axis=0))
            m_k = float(np.mean(s[:, 0, 0], axis=0))
            print(
                f"[samples K=10] phys ch0 fut step0 mean over K={m_k:.6f} "
                f"var across K (step 0 ch0)={v_k:.6g}"
            )
        pred0 = pred[0].cpu().numpy()
        gt0 = fut[0].cpu().numpy()
        print(f"[point pred MoM] pred[0][:4]={pred0.flatten()[:4]} vs gt[0][:4]={gt0.flatten()[:4]}")


if __name__ == "__main__":
    main()
