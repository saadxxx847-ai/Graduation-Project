"""
简单确定性基线：persistence、滑动均值、DLinear、LSTM、Plain Transformer；
与 SimDiff 共用原始尺度 (hist, fut)，在验证集 MSE 上早停训练。
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm


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
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        d_in = seq_len * channels
        d_out = pred_len * channels
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        b, lh, c = hist.shape
        x = hist.reshape(b, lh * c)
        y = self.lin(x)
        return y.reshape(b, self.pred_len, c)


class BaselineLSTM(nn.Module):
    """历史序列 LSTM，最后隐状态线性映射到整段未来 (Lf, C)。"""

    def __init__(
        self,
        channels: int,
        pred_len: int,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.channels = channels
        drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            channels,
            hidden,
            num_layers,
            batch_first=True,
            dropout=drop,
        )
        self.fc = nn.Linear(hidden, pred_len * channels)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        b = hist.shape[0]
        out, _ = self.lstm(hist)
        last = out[:, -1, :]
        y = self.fc(last)
        return y.reshape(b, self.pred_len, self.channels)


class BaselineTransformer(nn.Module):
    """Encoder-only Transformer：历史 token + 可学习未来 query token，一次读出 Lf 步。"""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        channels: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.d_model = d_model
        self.in_proj = nn.Linear(channels, d_model)
        self.fut_queries = nn.Parameter(torch.zeros(1, pred_len, d_model))
        self.pos = nn.Parameter(torch.zeros(1, seq_len + pred_len, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.out_proj = nn.Linear(d_model, channels)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        b, lh, _ = hist.shape
        h = self.in_proj(hist)
        q = self.fut_queries.expand(b, -1, -1)
        x = torch.cat([h, q], dim=1)
        x = x + self.pos[:, : lh + self.pred_len, :]
        out = self.encoder(x)
        fut_part = out[:, lh:, :]
        return self.out_proj(fut_part)


@torch.no_grad()
def _val_mse(model: nn.Module, val_loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for hist, fut in val_loader:
        hist = hist.to(device)
        fut = fut.to(device)
        pred = model(hist)
        losses.append(torch.mean((pred - fut) ** 2).item())
    return float(np.mean(losses)) if losses else float("nan")


def fit_regression_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_epochs: int,
    lr: float,
    patience: int,
    grad_clip_max_norm: float,
    name: str = "model",
) -> nn.Module:
    """AdamW + 验证 MSE 早停，取最优权重。"""
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.99))
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for hist, fut in tqdm(train_loader, desc=f"{name} ep{epoch}", leave=False):
            hist = hist.to(device)
            fut = fut.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(hist)
            loss = torch.mean((pred - fut) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
            opt.step()
            losses.append(loss.item())
        va = _val_mse(model, val_loader, device)
        print(
            f"  [{name}] epoch {epoch:03d} train_mse {float(np.mean(losses)):.6f} | val_mse {va:.6f}"
        )
        if np.isfinite(va) and va < best_val - 1e-9:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  [{name}] early stop at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model


def fit_dlinear(
    train_loader: torch.utils.data.DataLoader,
    seq_len: int,
    pred_len: int,
    channels: int,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    val_loader: torch.utils.data.DataLoader | None = None,
    max_epochs: int | None = None,
    patience: int = 10,
    grad_clip_max_norm: float = 1.0,
) -> DLinearMap:
    """兼容旧接口：未传 val_loader 时固定训 epochs轮；传 val_loader 时用早停。"""
    model = DLinearMap(seq_len, pred_len, channels).to(device)
    if val_loader is None:
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
    return fit_regression_model(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=max_epochs or 50,
        lr=lr,
        patience=patience,
        grad_clip_max_norm=grad_clip_max_norm,
        name="DLinear",
    )


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


@torch.no_grad()
def eval_horizon_mae(
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    pred_len: int,
    channel: int | None,
    n_channels: int,
) -> np.ndarray:
    """
    每一预报步的平均 MAE。channel 为 None 时对 C 维取平均（分母 B*C）；
    否则仅该通道（分母 B）。
    """
    sum_abs = np.zeros(pred_len, dtype=np.float64)
    total_w = 0.0
    for hist, fut in test_loader:
        hist = hist.to(device)
        fut = fut.to(device)
        pred = forward(hist, fut)
        diff = (pred - fut).abs()
        if channel is not None:
            diff = diff[:, :, channel]
            sum_abs += diff.sum(dim=0).detach().cpu().numpy()
            total_w += hist.size(0)
        else:
            sum_abs += diff.sum(dim=(0, 2)).detach().cpu().numpy()
            total_w += hist.size(0) * n_channels
    return sum_abs / max(total_w, 1.0)


@torch.no_grad()
def collect_pooled_predictions(
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    channel: int,
) -> tuple[np.ndarray, np.ndarray]:
    """展平所有 batch 的真值与预测（单通道），用于散点图。"""
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    for hist, fut in test_loader:
        hist = hist.to(device)
        fut = fut.to(device)
        pred = forward(hist, fut)
        ys.append(fut[..., channel].reshape(-1).detach().cpu().numpy())
        ps.append(pred[..., channel].reshape(-1).detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def print_baseline_block(
    name: str,
    mse_all: float,
    mae_all: float,
    mse_t: float,
    mae_t: float,
    temp_name: str,
    num_channels: int = 1,
) -> None:
    if num_channels <= 1:
        print(f"  [{name}] {temp_name} MSE={mse_t:.6f} MAE={mae_t:.6f}")
    else:
        print(
            f"  [{name}] 全特征平均 MSE={mse_all:.6f} MAE={mae_all:.6f} | "
            f"{temp_name} MSE={mse_t:.6f} MAE={mae_t:.6f}"
        )
