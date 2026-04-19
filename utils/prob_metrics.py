"""
概率预测指标：基于 K 次蒙特卡洛样本的 CRPS（有限集合公式）。
"""
from __future__ import annotations

import numpy as np
import torch


def crps_ensemble_1d(samples: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """
    samples: (B, K, L)；obs: (B, L)。返回 (B, L) 逐时刻 CRPS。

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|；K 个样本近似：
    term1 = mean_i |x_i - y|，term2 = 0.5 * mean_{ij} |x_i - x_j|。
    """
    if samples.dim() == 4:
        raise ValueError("请先对通道切片为 (B, K, L)")
    obs_e = obs.unsqueeze(1)
    term1 = (samples - obs_e).abs().mean(dim=1)
    diff = samples.unsqueeze(2) - samples.unsqueeze(1)
    term2 = 0.5 * diff.abs().mean(dim=(1, 2))
    return term1 - term2


@torch.no_grad()
def eval_crps_on_test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    channel: int,
    pred_len: int,
) -> tuple[float, np.ndarray]:
    """
    全测试集平均 CRPS 及按预报步 CRPS（单通道 channel）。
    model 须为 SimDiffWeather，forward 经 forecast(..., return_samples=True)。
    """
    sum_all = 0.0
    n_all = 0
    sum_h = np.zeros(pred_len, dtype=np.float64)
    n_h = np.zeros(pred_len, dtype=np.int64)

    for hist, fut in test_loader:
        hist = hist.to(device)
        fut = fut.to(device)
        out = model.forecast(hist, future=fut, return_samples=True)
        if out.samples is None:
            continue
        s = out.samples[..., channel]
        y = fut[..., channel]
        crps_bl = crps_ensemble_1d(s, y)
        sum_all += float(crps_bl.sum().item())
        n_all += crps_bl.numel()
        sum_h += crps_bl.sum(dim=0).detach().cpu().numpy()
        n_h += crps_bl.size(0)

    mean_crps = sum_all / max(n_all, 1)
    per_h = sum_h / np.maximum(n_h, 1)
    return mean_crps, per_h


def empirical_interval_coverage(
    samples: np.ndarray,
    obs: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> float:
    """
    samples: (K, L) 单窗口单通道；obs: (L,)
    返回该窗上逐时刻区间是否覆盖 obs 的比例（再对 L 平均已在内部）。
    """
    lo = np.quantile(samples, q_low, axis=0)
    hi = np.quantile(samples, q_high, axis=0)
    cover = (obs >= lo) & (obs <= hi)
    return float(np.mean(cover))
