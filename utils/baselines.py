"""
简单确定性基线：persistence、滑动均值、DLinear（单线性层），用于与扩散模型对比。
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def persistence_forecast(hist: torch.Tensor, pred_len: int) -> torch.Tensor:
    """每通道复制最后一个历史步。"""
    return hist[:, -1:, :].expand(-1, pred_len, -1)


@torch.no_grad()
def moving_average_forecast(hist: torch.Tensor, pred_len: int, window: int = 24) -> torch.Tensor:
    """每通道用历史最后 window 步的均值作为常数未来轨迹。"""
    m = min(int(window), hist.shape[1])
    mu = hist[:, -m:, :].mean(dim=1, keepdim=True)
    return mu.expand(-1, pred_len, -1)


class DLinearMap(nn.Module):
    """最小 DLinear：展平历史 -> 展平未来（无季节-趋势分解，作轻量对照）。"""

    def __init__(self, seq_len: int, pred_len: int, channels: int):
        super().__init__()
        d_in = seq_len * channels
        d_out = pred_len * channels
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        b, lh, c = hist.shape
        x = hist.reshape(b, lh * c)
        y = self.lin(x)
        return y.reshape(b, self.lin.out_features // c, c)


def fit_dlinear(
    train_loader: torch.utils.data.DataLoader,
    seq_len: int,
    pred_len: int,
    channels: int,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
) -> DLinearMap:
    model = DLinearMap(seq_len, pred_len, channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for hist, fut in train_loader:
            hist = hist.to(device)
            fut = fut.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(hist)
            loss = torch.mean((pred - fut) ** 2)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def eval_forecasts_mse_mae(
    predict_fn,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """predict_fn(hist) -> (B, pred_len, C)，与 batch 真值同设备。"""
    sum_sq = 0.0
    sum_abs = 0.0
    n_elem = 0
    for hist, fut in test_loader:
        hist = hist.to(device)
        fut = fut.to(device)
        pred = predict_fn(hist)
        diff = pred - fut
        sum_sq += float((diff**2).sum().item())
        sum_abs += float(diff.abs().sum().item())
        n_elem += diff.numel()
    mse = sum_sq / max(n_elem, 1)
    mae = sum_abs / max(n_elem, 1)
    return mse, mae


@torch.no_grad()
def eval_channel_mse_mae(
    predict_fn,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    channel: int,
) -> tuple[float, float]:
    sum_sq = 0.0
    sum_abs = 0.0
    n = 0
    for hist, fut in test_loader:
        hist = hist.to(device)
        fut = fut.to(device)
        pred = predict_fn(hist)
        d = pred[..., channel] - fut[..., channel]
        sum_sq += float((d**2).sum().item())
        sum_abs += float(d.abs().sum().item())
        n += d.numel()
    return sum_sq / max(n, 1), sum_abs / max(n, 1)


def print_baseline_block(
    name: str,
    mse_all: float,
    mae_all: float,
    mse_t: float,
    mae_t: float,
    temp_name: str,
) -> None:
    print(
        f"  [{name}] 全特征平均 MSE={mse_all:.6f} MAE={mae_all:.6f} | "
        f"{temp_name} MSE={mse_t:.6f} MAE={mae_t:.6f}"
    )
