"""
兼容层：核心实现见 `utils/independent_normalizer.py`（Normalization Independence）。
"""
from __future__ import annotations

from utils.independent_normalizer import IndependentNormalizer


class Normalizer:
    """旧 API 映射到 IndependentNormalizer。"""

    @staticmethod
    def normalize_pair(
        hist,
        future,
        independent_future: bool = True,
        eps: float = 1e-5,
    ):
        if not independent_future:
            raise ValueError(
                "已统一为 SimDiff NI：请使用 IndependentNormalizer.normalize_history / normalize_future"
            )
        hist_n, sh = IndependentNormalizer.normalize_history(hist, eps=eps)
        fut_n, sf = IndependentNormalizer.normalize_future(future, eps=eps)
        stats = {**sh, **sf}
        return hist_n, fut_n, stats

    @staticmethod
    def denormalize_future(future_norm, f_mean, f_std):
        return IndependentNormalizer.inverse_transform_future(future_norm, f_mean, f_std)

    @staticmethod
    def infer_future_stats_from_hist(hist, eps: float = 1e-5):
        raise NotImplementedError(
            "NI 下未来反变换应使用 batch 真值 future 的统计量，或模型内 _fut_mu_marginal / _fut_sig_marginal"
        )
