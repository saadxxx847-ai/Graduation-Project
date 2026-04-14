"""
毕设超参数集中配置（避免在代码中硬编码）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # 路径（相对项目根目录 Simdiff_weather）
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_path: str = "data/weather.csv"
    processed_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"

    # 数据：weather.csv 除 date 外为数值列，运行时自动得到 n_features
    # 更长历史有助于长期依赖；需重训且与旧 checkpoint 的 seq_len 不一致时无法直接加载权重
    seq_len: int = 96
    pred_len: int = 24
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio = 1 - train - val

    # 模型
    input_dim: int = -1  # -1 表示由数据加载器写入
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1

    # 扩散
    timesteps: int = 200
    cosine_s: float = 5.0
    # 采样：默认 ddpm 与训练噪声日程一致，最稳；ddim 若步数 > timesteps 会产生大量重复时间步，易数值发散
    sampling_mode: str = "ddpm"  # "ddpm" | "ddim"
    # 推理子步数（仅 ddim）：None 表示与 timesteps 相同；不得超过 timesteps（超过会截断并告警）
    sampling_steps: int | None = None
    ddim_eta: float = 0.0  # 0=确定性 DDIM；>0 易锯齿，一般保持 0

    # 采样数值稳定：在归一化空间裁剪预测的 x0（与 z_clip 同量级）
    sample_clip_pred_x0: bool = True
    sample_debug: bool = False
    sample_debug_every: int = 20

    # 训练损失：纯 MSE 易学成条件均值 → 曲线偏平滑；加 L1 与时间差分可强调高频/突变
    training_noise_l1_weight: float = 0.08
    training_noise_temporal_diff_weight: float = 0.05

    # 训练（默认偏保守，减轻梯度爆炸）
    batch_size: int = 64
    learning_rate: float = 5e-5
    epochs: int = 50
    num_workers: int = 0
    seed: int = 42
    early_stop_patience: int = 10
    grad_clip_max_norm: float = 0.5
    # 全局标准化后的 Z 分数裁剪到 [-z_clip, z_clip]，避免极端值灌入 Transformer
    z_clip: float = 4.0

    # 设备
    device: str = "cuda"

    # True：未来用自身 mean/std 归一化（论文叙述更贴，但推理反归一化易与训练不一致）；
    # False（默认）：未来与历史共用历史窗口统计量，训练/采样/反归一化一致。
    independent_future_normalization: bool = False

    # 使用「训练集全局」per-channel 均值/方差标准化（强烈推荐）。
    # 若 False，则退回按窗口统计量；近常数列易使标准差趋于 0，归一化爆炸（Loss 极高、MSE 可达 1e12 量级）。
    use_global_standardization: bool = True
    # 由 data_loader 在读取训练段后写入 (C,) float32；未设置时模型使用窗口归一化
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
