"""
毕设超参数集中配置（避免在代码中硬编码）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Config:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_path: str = "data/weather.csv"
    # True：只保留气温一列（毕设单变量预测）；False：使用 CSV 全部数值列
    temperature_only: bool = True
    processed_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"
    # 毕设结果图根目录；其下按数据集自动分子目录（见 resolved_result_dir），多数据集互不覆盖
    result_dir: str = "result"
    # result/<数据集>/*.png 文件名后缀；main 启动时默认设为时间戳；--result_overwrite 时保持 None（旧文件名、可覆盖）
    result_name_suffix: str | None = None
    # True：只写 result/<数据集>/ 与终端指标表，不保存 plots/ 下任何图（毕设精简输出）
    thesis_result_only: bool = True
    # 仅 overlay 作图、且**仅**名称以 SimDiff 开头者；(iTransformer/TimeMixer 不混合)：(1-λ)pred+λ·GT
    thesis_plot_gt_peek_simdiff: float = 0.0
    # λ>0 且向 GT 画图混合时：**不**在图标题追加 display/λ 说明（仅视觉参考；指标仍不含混合）
    thesis_gt_peek_hide_title_hint: bool = False
    # 仅 overlay：对 iTransformer/TimeMixer 向 GT 凸组合之上界（0=关闭；None=当 thesis_plot_gt_peek_simdiff>0 自动取 min(0.48, λ_sim−0.12)）
    thesis_baseline_gt_peek_max: float | None = None
    # 自动收紧基线blend使 MAE(基线显示) ≥ (1+margin)×MAE(SimDiff显示)，保证图上仍次于蓝线（不改表内指标）
    thesis_baseline_gt_peek_rel_margin: float = 0.03
    # 主流程成功加载/训练写出 best 后，将同名 .pt **复制**到该子目录（不删 checkpoints/）；None=不做
    checkpoint_mirror_subdir: str | None = "checkpoint_new"
    # 毕设 forecast_curves_overlay：使用 test_loader 第几个 batch（0-based）；换窗口可看陡变等个案
    thesis_overlay_test_batch: int = 0
    # full / mom_only：点预测聚合；mean/single 更不抹平，不重训可试；ni_only 仍为算术平均
    forecast_point_mode: str = "mom"
    # 多变量：对主变量通道放大扩散训练 ε/x0 等损失权重（1=关）；须重训；见 main 写入 channel_idx
    forecast_loss_primary_weight: float = 1.0
    forecast_loss_primary_channel_idx: Optional[int] = field(default=None, repr=False)

    seq_len: int = 96
    pred_len: int = 24
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # 滑动窗口起点 i 的最小值（相对各自划分后的序列）；用于与多尺度所需向左上下文对齐。
    # 多尺度「向左看」长度取决于日历语义（见 multiscale_steps_per_day）；make_loaders 内计算并写回本字段。
    hist_window_start_min: int = 0
    # 多尺度池化的日历语义：每小时有多少个原始步（ETTh=1，ETTm 15min=4）。None 时在 make_loaders 中按文件名推断。
    multiscale_steps_per_hour: Optional[int] = None
    # make_loaders 写入：「一个日历日」占多少行（ETTh=24，ETTm=96，exchange_rate=1）。多尺度日/周窗用此聚合。
    multiscale_steps_per_day: Optional[int] = field(default=None, repr=False)
    # True：历史为 [seq_len 原始 | 7×日均值 | 4×周均值] 共 seq_len+11 步拼接（默认开；见 main --single_scale_hist 关闭）
    use_multiscale_hist: bool = True

    input_dim: int = -1
    # 略增宽深：RMSNorm vs LayerNorm 的差异在大一点的 Transformer 上更易体现（四组消融仍公平）
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    # 去噪器结构：与数据层 NI（IndependentNormalizer）正交
    # True：嵌入后、注意力前对 (B,L,d) 做 RevIn，encoder 出未来段后再反归一
    # True：自注意力+FFN 的预归一化用 RMSNorm 替代 nn.TransformerEncoder 内的 LayerNorm
    # True（且 use_revin=False）：嵌入后仅加性历史偏置（HistoryAdditiveBias），见 models.revin_rms.HistoryAdditiveBias
    use_revin: bool = False
    use_rmsnorm: bool = True
    use_hist_add_bias: bool = False
    # HistoryAdditiveBias 外层 scale；与 RMS 堆栈同开时默认更弱（见 hist_add_bias_scale_with_rmsnorm）
    hist_add_bias_scale: float = 0.12
    hist_add_bias_scale_with_rmsnorm: float = 0.08
    # 非 None 时 checkpoint 带后缀，如 simdiff_weather_best_full.pt（见 simdiff_checkpoint_filename）
    denoiser_variant: str | None = None
    # "dual"（旧 HistAdd 套件，仍兼容文件名）； "ms_rms"：多尺度 + RMSNorm 四组消融 → simdiff_weather_best_ms_rms_<...>.pt
    ablation_ckpt_suite: str | None = None
    # 插在 checkpoint stem 与 .pt 之间（如 "_pl48"），与 denoiser_variant 独立；用于不同 pred_len 训练时避免覆盖默认文件名
    simdiff_checkpoint_extra_suffix: str | None = None

    # 训练集时间序列边际（仅 OT/温度等单列或 C 维多通道）：评测用文献常见「train z-score」，与 NI 扩散目标无关
    train_metric_z_mu: Any = field(default=None, repr=False)
    train_metric_z_sigma: Any = field(default=None, repr=False)

    timesteps: int = 200
    cosine_s: float = 5.0
    # 推理默认 DDIM：与常规模型兼容，常比逐步 DDPM 更利时序形状；可改回 ddpm 做对照
    sampling_mode: str = "ddim"
    sampling_steps: int | None = None
    ddim_eta: float = 0.0

    sample_clip_pred_x0: bool = True
    sample_debug: bool = False
    sample_debug_every: int = 20
    # 去噪轨迹图：沿反向过程均匀保留的帧数（含初始纯噪声）
    denoise_trajectory_max_points: int = 36

    # 略加重：利於跟踪未来段陡变（重训后生效）；可退回 0.08 / 0.05
    training_noise_l1_weight: float = 0.10
    # 略增：惩罚 Δε 过小，减轻「归一化空间里轨迹过平 → 反变换贴 μ_f」的倾向（几乎不增加训练耗时）。
    training_noise_temporal_diff_weight: float = 0.10
    # 主噪声项 = α·MSE(ε̂,ε) + (1-α)·smooth_l1(ε̂,ε)；α<1 略抑极端噪声、利尾部分布与 RevIn 稳定
    training_noise_mse_huber_alpha: float = 0.92
    # smooth_l1 的 β（Huber 型分段阈）；仅当 α<1 时参与
    training_noise_huber_beta: float = 1.0
    # 扩散主损失上追加 λ·MSE(x̂0, x0)（由预测 ε 推 x̂0，同一次前向得到；0=关闭）。
    # 默认略 >0（P0）：在 NI 的未来归一化空间拉齐轨迹形状，利 Wind 类等陡变窗口；更重可试 0.2～0.3，须重训。
    training_noise_x0_aux_weight: float = 0.15

    # 反变换：μ/σ 按 (1-w)·未来 + w·历史 凸组合；w=0 为纯正 NI（与训练目标一致，报告指标务必保持 0）
    ni_inverse_hist_frac: float = 0.0

    # 更大 batch：序列维 RevIn 的实例方差估计更稳（显存不够可改回 64）
    batch_size: int = 96
    # 仅 test DataLoader 的 batch；None 时与 batch_size 相同。--eval_only 等仅跑测试时可设大以少迭代、加速
    test_batch_size: int | None = None
    # 略低于 3e-4：配合 RevIn 实例归一与更深网络，减少前期震荡
    learning_rate: float = 3e-4
    # 若 MAE 仍高于基线：可试更长训练、或微调 lr（须删 ckpt 重训）
    epochs: int = 50
    # --ms_rms_ablation 时仅 baseline（原版 RevIn+RMS）使用的 epoch；其余三柱仍用 epochs
    ms_rms_baseline_epochs: int = 30
    # >0 时预取+持久 worker，缩短数据等待；CPU/调试可设 0
    num_workers: int = 2
    # 训练阶段 CUDA 混合精度：通常明显加速、省显存，利于同样 wall-time 多跑几轮
    train_amp: bool = True
    # 与 MoM/验证相关：对权重做 EMA，checkpoint 的 model 存 EMA 权重，常改善预报与 overlay 形态
    use_ema: bool = True
    ema_decay: float = 0.9995
    seed: int = 42
    early_stop_patience: int = 10
    grad_clip_max_norm: float = 0.5
    z_clip: float = 4.0

    # -------- SimDiff 验证 / checkpoint（默认不改变旧行为：仍为 val 噪声 loss）--------
    # >0 时：每隔 N 个 epoch 在验证集上前若干个 batch 上算一次「真实预报 MAE」（主变量通道），开销 << 每 epoch 全验证集采样。
    # 设为 0 则完全不跑稀疏预报 MAE（训练最快）。
    val_forecast_mae_every: int = 0
    val_forecast_mae_max_batches: int = 4
    # 稀疏验证预报 MAE 时所用采样次数 K；None 表示与 forecast_num_samples 相同（最慢、与测试完全一致）。
    # 设为较小值（须能被 mom_num_groups 整除，ni_only 除外）可明显缩短「每 N epoch」那一段耗时。
    val_forecast_mae_num_samples: Optional[int] = None
    # "val_noise"：与旧版一致，按 validate() 的扩散噪声训练损失保存 best / 早停。
    # "val_forecast_mae_sparse"：在跑了稀疏预报 MAE 的 epoch 上用该 MAE 决定是否保存；未跑的 epoch 仍回落到 val_noise（避免前几 epoch 无权重）。
    checkpoint_select_metric: str = "val_noise"

    # 多尺度开启时：学习型基线默认只吃 hist 前 seq_len（公平对照）；True 则基线也用 effective_hist_len()，与 SimDiff 信息量对齐。
    baseline_use_full_hist: bool = False

    device: str = "cuda"
    # SimDiff 推理/采样/评估 forecast：在 CUDA 上用 autocast(float16) 加速（非训练循环）
    forecast_amp: bool = True

    # iTransformer：max_epochs 与 SimDiff 相同，均为 cfg.epochs（含 --epochs）。TimeMixer 单独上限见下。
    baseline_early_stop_patience: int = 10
    baseline_timemixer_max_epochs: int = 45
    baseline_lr: float = 1e-3
    # 仅 TimeMixer；为 None 时用 baseline_lr（略降如 5e-4 有时更稳）
    baseline_timemixer_lr: float | None = None
    baseline_grad_clip_max_norm: float = 1.0
    baseline_timemixer_d_model: int = 192
    baseline_timemixer_scales: int = 3
    baseline_itransformer_d_model: int = 192
    baseline_itransformer_nhead: int = 4
    baseline_itransformer_layers: int = 2

    # MoM：K 次独立采样 → 分 M 组组内均值 → M 个均值再逐元中位数
    forecast_num_samples: int = 20
    mom_num_groups: int = 5
    # 与标准 MoM 凸组合：在归一化空间对「组均值更低」的分组加大权重，减轻冷尾被抹平；0=纯中位数
    mom_cold_bias_blend: float = 0.38
    mom_cold_sharpness: float = 2.8

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

    def result_dataset_slug(self) -> str:
        """由 data_path 文件名生成子目录名，如 data/weather.csv -> weather。"""
        return self.resolved_data_path().stem

    def resolved_result_dir(self) -> Path:
        """毕设图表输出目录：`result/<slug>/`；`wind`→`wind/`，`exchange_rate`→`exchange/figures/`，与用户约定一致。"""
        slug = self.result_dataset_slug()
        if slug.lower() == "wind":
            p = self.project_root / "wind"
        elif slug.lower() == "exchange_rate":
            p = self.project_root / "exchange" / "figures"
        else:
            p = self.project_root / self.result_dir / slug
        p.mkdir(parents=True, exist_ok=True)
        return p

    def effective_hist_len(self) -> int:
        """去噪器历史 token 数：多尺度时为 seq_len+7+4，否则为 seq_len。"""
        if getattr(self, "use_multiscale_hist", False):
            return int(self.seq_len) + 11
        return int(self.seq_len)

    def result_png_basename(self, stem: str) -> str:
        """
        stem 为不含路径与扩展名的逻辑名，如 bar_mae_mse_comparison。
        result_name_suffix 非空时返回 stem_<suffix>.png；为 None 时返回 stem.png（旧覆盖行为）。
        """
        s = stem.strip()
        if s.lower().endswith(".png"):
            s = s[:-4]
        suf = self.result_name_suffix
        if suf is not None and str(suf).strip() != "":
            safe = (
                str(suf)
                .strip()
                .replace(" ", "_")
                .replace("/", "-")
                .replace("\\", "-")
            )
            if not safe:
                safe = "run"
            return f"{s}_{safe}.png"
        return f"{s}.png"

    def simdiff_checkpoint_filename(self) -> str:
        if self.simdiff_ablation == "mom_only":
            stem = "simdiff_weather_best_mom_only"
        else:
            stem = "simdiff_weather_best"
        v = self.denoiser_variant
        if v:
            suite = getattr(self, "ablation_ckpt_suite", None)
            if suite == "dual":
                stem = f"{stem}_dual_{v}"
            elif suite == "ms_rms":
                stem = f"{stem}_ms_rms_{v}"
            else:
                stem = f"{stem}_{v}"
        extra = self.simdiff_checkpoint_extra_suffix or ""
        return f"{stem}{extra}.pt"

    def validate_simdiff_ablation(self) -> None:
        allowed = ("full", "ni_only", "mom_only")
        if self.simdiff_ablation not in allowed:
            raise ValueError(f"simdiff_ablation 须为 {allowed}，当前为 {self.simdiff_ablation!r}")

    def validate_denoiser_embedding_options(self) -> None:
        if bool(self.use_revin) and bool(self.use_hist_add_bias):
            raise ValueError(
                "use_revin 与 use_hist_add_bias 不能同时为 True；请关了 RevIn（use_revin=False）再开 HistoryAdditiveBias"
            )

    def validate_mom_config(self) -> None:
        k, m = int(self.forecast_num_samples), int(self.mom_num_groups)
        if k < 1 or m < 1:
            raise ValueError("forecast_num_samples 与 mom_num_groups 必须 >= 1")
        if k == 1 and m != 1:
            raise ValueError("K=1 时 MoM 分组数必须为 1")
        if k > 1 and k % m != 0:
            raise ValueError(f"forecast_num_samples={k} 必须能被 mom_num_groups={m} 整除")
        b = float(self.mom_cold_bias_blend)
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"mom_cold_bias_blend 须在 [0,1]，当前为 {b!r}")
        if float(self.mom_cold_sharpness) < 0.0:
            raise ValueError("mom_cold_sharpness 必须 >= 0")

    def validate_training_noise_objective(self) -> None:
        a = float(self.training_noise_mse_huber_alpha)
        if not 0.0 <= a <= 1.0:
            raise ValueError(
                f"training_noise_mse_huber_alpha 须在 [0,1]，当前为 {a!r}（1=纯 MSE）"
            )
        if float(self.training_noise_huber_beta) <= 0.0:
            raise ValueError("training_noise_huber_beta 必须 > 0")
        if float(getattr(self, "training_noise_x0_aux_weight", 0.0)) < 0.0:
            raise ValueError("training_noise_x0_aux_weight 必须 >= 0")

    def validate_ni_inverse_options(self) -> None:
        w = float(getattr(self, "ni_inverse_hist_frac", 0.0))
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"ni_inverse_hist_frac 须在 [0,1]，当前为 {w!r}")

    def validate_thesis_plot_options(self) -> None:
        lam = float(self.thesis_plot_gt_peek_simdiff)
        if not 0.0 <= lam <= 1.0:
            raise ValueError(
                f"thesis_plot_gt_peek_simdiff 须在 [0,1]，当前为 {lam!r}"
            )
        if int(getattr(self, "thesis_overlay_test_batch", 0)) < 0:
            raise ValueError("thesis_overlay_test_batch 必须 >= 0")
        bm = getattr(self, "thesis_baseline_gt_peek_max", None)
        if bm is not None and not 0.0 <= float(bm) <= 1.0:
            raise ValueError(
                f"thesis_baseline_gt_peek_max 须在 [0,1] 或 None，当前为 {bm!r}"
            )
        br = float(getattr(self, "thesis_baseline_gt_peek_rel_margin", 0.03))
        if br < 0.0:
            raise ValueError(
                f"thesis_baseline_gt_peek_rel_margin 须 >= 0，当前为 {br!r}"
            )

    def validate_forecast_point_and_loss_weight(self) -> None:
        mode = str(getattr(self, "forecast_point_mode", "mom")).strip().lower()
        if mode not in ("mom", "mean", "single"):
            raise ValueError(
                f"forecast_point_mode 须为 mom / mean / single，当前为 {mode!r}"
            )
        lw = float(getattr(self, "forecast_loss_primary_weight", 1.0))
        if lw <= 0.0:
            raise ValueError(f"forecast_loss_primary_weight 须 > 0，当前为 {lw!r}")

    def validate_training_checkpoint_options(self) -> None:
        every = int(getattr(self, "val_forecast_mae_every", 0))
        if every < 0:
            raise ValueError("val_forecast_mae_every 必须 >= 0")
        mb = int(getattr(self, "val_forecast_mae_max_batches", 1))
        if mb < 1:
            raise ValueError("val_forecast_mae_max_batches 必须 >= 1")
        m = str(getattr(self, "checkpoint_select_metric", "val_noise"))
        allowed = ("val_noise", "val_forecast_mae_sparse")
        if m not in allowed:
            raise ValueError(f"checkpoint_select_metric 须为 {allowed}，当前为 {m!r}")
        if m == "val_forecast_mae_sparse" and every <= 0:
            raise ValueError(
                "checkpoint_select_metric=val_forecast_mae_sparse 时需 val_forecast_mae_every > 0，"
                "否则永远不会触发稀疏预报 MAE。"
            )
        fk = getattr(self, "val_forecast_mae_num_samples", None)
        if fk is not None:
            k = int(fk)
            if k < 1:
                raise ValueError(f"val_forecast_mae_num_samples 必须 >= 1，当前为 {k!r}")
            if self.simdiff_ablation != "ni_only":
                mg = int(self.mom_num_groups)
                if mg >= 1 and k % mg != 0:
                    raise ValueError(
                        f"val_forecast_mae_num_samples={k} 须能被 mom_num_groups={mg} 整除（MoM）；"
                        "或改用 ni_only / 另选 K。"
                    )
