"""
归一化与反归一化。

默认模式（推荐，训练/推理一致）：
历史与未来都用「历史窗口」的 mean/std 做标准化：
  - hist_n = (hist - μ_h) / σ_h
  - fut_n  = (future - μ_h) / σ_h
  这样扩散在「以历史尺度为基准」的空间里学习；推理时用同一组 μ_h, σ_h 反变换，
  与训练目标空间严格一致。

可选「独立未来统计量」模式（更贴近 SimDiff 文中「历史/未来统计解耦」的叙述）：
  训练时 fut_n 使用未来窗口自身的 mean/std。
  此时若推理仍用历史 μ/σ 反归一化，会与训练坐标系不一致，导致生成指标极差。
  评估时请使用本 batch 的真值未来统计量反归一化（见 simdiff.validation_mse）；
  无真值时的 forecast() 只能近似（用历史统计量），会有分布偏移。
"""
from __future__ import annotations

import torch


class Normalizer:
    """按 batch 归一化，并返回反归一化所需的统计量。"""

    @staticmethod
    def normalize_pair(
        hist: torch.Tensor,
        future: torch.Tensor,
        independent_future: bool = False,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        hist: (B, Lh, C), future: (B, Lf, C)

        independent_future=False（默认）:
            历史与未来共用 (μ_h, σ_h)，保证 forecast 反归一化与训练一致。
        independent_future=True:
            历史用 (μ_h, σ_h)，未来用 (μ_f, σ_f)（论文式「独立归一化」，
            但推理无真值时反归一化需另行处理）。
        """
        h_mean = hist.mean(dim=1, keepdim=True)
        h_std = hist.std(dim=1, keepdim=True).clamp_min(eps)
        hist_n = (hist - h_mean) / h_std

        if independent_future:
            f_mean = future.mean(dim=1, keepdim=True)
            f_std = future.std(dim=1, keepdim=True).clamp_min(eps)
            fut_n = (future - f_mean) / f_std
        else:
            f_mean = h_mean
            f_std = h_std
            fut_n = (future - h_mean) / h_std

        stats = {
            "h_mean": h_mean,
            "h_std": h_std,
            "f_mean": f_mean,
            "f_std": f_std,
        }
        return hist_n, fut_n, stats

    @staticmethod
    def denormalize_future(future_norm: torch.Tensor, f_mean: torch.Tensor, f_std: torch.Tensor) -> torch.Tensor:
        """future_norm * f_std + f_mean；f_mean/f_std 可为 (B,1,C) 或已与 future_norm 广播一致。"""
        return future_norm * f_std + f_mean

    @staticmethod
    def infer_future_stats_from_hist(hist: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor]:
        """
        无未来真值时：用历史窗口沿时间维的 per-channel 均值、标准差，
        在默认训练设置下与 normalize_pair(..., independent_future=False) 中的 (f_mean,f_std) 一致。
        """
        f_mean = hist.mean(dim=1, keepdim=True)
        f_std = hist.std(dim=1, keepdim=True).clamp_min(eps)
        return f_mean, f_std
