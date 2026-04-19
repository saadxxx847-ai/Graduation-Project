"""
毕设超参数集中配置（避免在代码中硬编码）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_path: str = "data/weather.csv"
    # True：只保留气温一列（毕设单变量预测）；False：使用 CSV 全部数值列
    temperature_only: bool = True
    processed_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"

    seq_len: int = 96
    pred_len: int = 24
    train_ratio: float = 0.70
    val_ratio: float = 0.15

    input_dim: int = -1
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1

    timesteps: int = 200
    cosine_s: float = 5.0
    sampling_mode: str = "ddpm"
    sampling_steps: int | None = None
    ddim_eta: float = 0.0

    sample_clip_pred_x0: bool = True
    sample_debug: bool = False
    sample_debug_every: int = 20
    # 去噪轨迹图：沿反向过程均匀保留的帧数（含初始纯噪声）
    denoise_trajectory_max_points: int = 36

    training_noise_l1_weight: float = 0.08
    training_noise_temporal_diff_weight: float = 0.05

    batch_size: int = 64
    learning_rate: float = 5e-5
    epochs: int = 50
    num_workers: int = 0
    seed: int = 42
    early_stop_patience: int = 10
    grad_clip_max_norm: float = 0.5
    z_clip: float = 4.0

    device: str = "cuda"

    # DLinear / LSTM / Plain Transformer：与 SimDiff 同数据划分，在验证集 MSE 上早停
    baseline_max_epochs: int = 50
    baseline_early_stop_patience: int = 10
    baseline_lr: float = 1e-3
    baseline_grad_clip_max_norm: float = 1.0
    baseline_transformer_d_model: int = 128
    baseline_transformer_nhead: int = 4
    baseline_transformer_layers: int = 3
    baseline_lstm_hidden: int = 128
    baseline_lstm_layers: int = 2

    # MoM：K 次独立采样 → 分 M 组组内均值 → M 个均值再逐元中位数
    forecast_num_samples: int = 20
    mom_num_groups: int = 5

    # SimDiff 消融：full=NI+MoM；ni_only=保留 NI，评估用 K 次算术均值（无 MoM），与 full 共用权重；
    # mom_only=未来用历史窗 μ_h,σ_h 归一化（非 NI）+ MoM，需单独训练，见 simdiff_checkpoint_filename()
    simdiff_ablation: str = "full"

    # 训练集「仅未来窗口」聚合的边际 μ/σ，供无真值时反归一化（make_loaders 写入）
    train_future_marginal_mean: Any = field(default=None, repr=False)
    train_future_marginal_std: Any = field(default=None, repr=False)

    # True时在 training_loss 内做独立归一化幂等检查（略慢）
    debug_norm_assert: bool = False

    # 保留字段仅兼容旧 checkpoint meta；逻辑上始终为窗口级 NI（历史/未来分离）
    independent_future_normalization: bool = False
    # False：不在 SimDiff 中使用；仅当 True 时 data_loader 仍写入 global_mean/std（旧工具）
    use_global_standardization: bool = False
    global_mean: Any = field(default=None, repr=False)
    global_std: Any = field(default=None, repr=False)

    def resolved_data_path(self) -> Path:
        return self.project_root / self.data_path

    def resolved_processed(self) -> Path:
        p = self.project_root / self.processed_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolved_checkpoint_dir(self) -> Path:
        p = self.project_root / self.checkpoint_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolved_plot_dir(self) -> Path:
        p = self.project_root / self.plot_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def simdiff_checkpoint_filename(self) -> str:
        if self.simdiff_ablation == "mom_only":
            return "simdiff_weather_best_mom_only.pt"
        return "simdiff_weather_best.pt"

    def validate_simdiff_ablation(self) -> None:
        allowed = ("full", "ni_only", "mom_only")
        if self.simdiff_ablation not in allowed:
            raise ValueError(f"simdiff_ablation 须为 {allowed}，当前为 {self.simdiff_ablation!r}")

    def validate_mom_config(self) -> None:
        k, m = int(self.forecast_num_samples), int(self.mom_num_groups)
        if k < 1 or m < 1:
            raise ValueError("forecast_num_samples 与 mom_num_groups 必须 >= 1")
        if k == 1 and m != 1:
            raise ValueError("K=1 时 MoM 分组数必须为 1")
        if k > 1 and k % m != 0:
            raise ValueError(f"forecast_num_samples={k} 必须能被 mom_num_groups={m} 整除")
