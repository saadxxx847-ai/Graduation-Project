"""
Normalization Independence（SimDiff）：历史段与未来段各自仅用本段数据估计统计量，互不混用。

* normalize_history：μ_h, σ_h 仅由 hist 在时间维上计算，**不得**依赖 future。
* normalize_future：μ_f, σ_f 仅由 future 在时间维上计算，**不得**依赖 hist。
* 推理阶段若无真值 future，反归一化使用训练集上仅由「未来窗口」聚合得到的边际 μ,σ（见 data_loader.fit_future_marginal_stats）。
"""
from __future__ import annotations

import torch


class IndependentNormalizer:
    """窗口级独立归一化：历史与未来统计量严格分离。"""

    @staticmethod
    def normalize_history(hist: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        hist: (B, Lh, C)。仅使用 hist 计算 μ_h, σ_h。
        """
        if hist.ndim != 3:
            raise ValueError(f"hist 期望 (B,Lh,C)，得到 {tuple(hist.shape)}")
        mu_h = hist.mean(dim=1, keepdim=True)
        sig_h = hist.std(dim=1, keepdim=True).clamp_min(eps)
        hist_n = (hist - mu_h) / sig_h
        stats = {"mu_h": mu_h, "sig_h": sig_h}
        return hist_n, stats

    @staticmethod
    def normalize_future(future: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        future: (B, Lf, C)。仅使用 future 计算 μ_f, σ_f。
        """
        if future.ndim != 3:
            raise ValueError(f"future 期望 (B,Lf,C)，得到 {tuple(future.shape)}")
        mu_f = future.mean(dim=1, keepdim=True)
        sig_f = future.std(dim=1, keepdim=True).clamp_min(eps)
        fut_n = (future - mu_f) / sig_f
        stats = {"mu_f": mu_f, "sig_f": sig_f}
        return fut_n, stats

    @staticmethod
    def inverse_transform_future(
        future_norm: torch.Tensor,
        mu_f: torch.Tensor,
        sig_f: torch.Tensor,
    ) -> torch.Tensor:
        """
        future_norm: (B, Lf, C)；mu_f, sig_f: (B, 1, C) 或与 batch 广播一致。
        """
        lf = future_norm.shape[1]
        return future_norm * sig_f.expand(-1, lf, -1) + mu_f.expand(-1, lf, -1)

    @staticmethod
    def debug_assert_shapes_and_idempotent_history(
        hist: torch.Tensor,
        future: torch.Tensor,
        seq_len: int,
        pred_len: int,
    ) -> None:
        """调试用：时间步对齐；对同一 hist 重复 normalize_history 结果一致。"""
        if hist.shape[1] != seq_len:
            raise AssertionError(f"hist 长度应为 seq_len={seq_len}，得到 {hist.shape[1]}")
        if future.shape[1] != pred_len:
            raise AssertionError(f"future 长度应为 pred_len={pred_len}，得到 {future.shape[1]}")
        _, s1 = IndependentNormalizer.normalize_history(hist)
        _, s2 = IndependentNormalizer.normalize_history(hist.clone())
        if not torch.allclose(s1["mu_h"], s2["mu_h"]) or not torch.allclose(s1["sig_h"], s2["sig_h"]):
            raise AssertionError("normalize_history 对相同 hist 非幂等")


def mom_aggregate_normalized(
    stacked: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    stacked: (B, K, L, C) 归一化空间中的 K 条轨迹。
    分成 M=num_groups 组，每组 S=K/M 条求均值，再对 M 个组均值取逐元素中位数（Median-of-Means）。

    返回：(single, sample_mean, mom) 均在归一化空间，shape (B, L, C)。
    """
    b, k, l, c = stacked.shape
    m = int(num_groups)
    if m <= 0 or k % m != 0:
        raise ValueError(f"要求 K={k} 能被 M={m} 整除，便于 MoM 分组")
    s = k // m
    grouped = stacked.reshape(b, m, s, l, c).mean(dim=2)
    mom = grouped.median(dim=1).values
    sample_mean = stacked.mean(dim=1)
    single = stacked[:, 0].contiguous()
    return single, sample_mean, mom
