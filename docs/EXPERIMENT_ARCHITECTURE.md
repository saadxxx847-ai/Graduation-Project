# SimDiff-Weather：架构与消融（供问答引用）

本文档与**当前仓库实现**一致。**HistConditionalFiLM（FiLM）已移除**；嵌入后可选 **HistoryAdditiveBias（仅加性历史偏置，模块 A）**；编码器可选用 **LN 原版 Transformer** 或 **RMSNorm 堆栈（模块 B）**。

---

## 1. 数据与 NI

- 默认 **`data/weather.csv`**，`seq_len=96`，`pred_len=24`。  
- **IndependentNormalizer**：历史与未来 **各自在时间维** 估计 `μ/σ`，**互不混用**（`utils/independent_normalizer.py`）。

---

## 2. 扩散与 `SimDiffWeather`

- 高斯扩散；训练在 **NI 后的未来序列**上预测噪声；条件为标准化历史。
- **`simdiff_ablation`=`full`** 表示 NI + MoM 等主线；与去噪「A/B」消融正交。
- **`DenoiserTransformer`**（`models/network.py`）：`in_proj` 共享、`pos_h/f`、**扩散时刻 `t` 的 sinusoidal+MPE** broadcast 加到历史与未来 token，`concat → [可选嵌入模块] → Transformer 栈 → 取未来段 → out_proj`。  
- **`use_revin` 与 `use_hist_add_bias` 不能同时为 True。**

---

## 3. 嵌入后路径（三者之一）

| 开关 | 行为 |
|------|------|
| **RevIN** (`use_revin=True`) | **`RevINPatch`**：`forward_norm(seq)`，`encoder` 后对 **未来子序列** `forward_denorm` 再 **`out_proj`**。 |
| **HistoryAdditiveBias** (`use_hist_add_bias=True`) | **仅加性**：历史 token embedding 均值 → 小 MLP → `seq += scale · bias` broadcast；末层线性 **零初始化**（`models/revin_rms.py`：`HistoryAdditiveBias`）。 |
| **都不开** | 直入编码器栈。|

**缩放**：`hist_add_bias_scale`（默认 0.12）；若 **`use_hist_add_bias` 且 `use_rmsnorm`**（≈full）则用 **`hist_add_bias_scale_with_rmsnorm`（默认 0.08）**（`models/simdiff.py`）。

---

## 4. 编码器栈

- **`use_rmsnorm=True`**：`ModuleList[DenoiserEncoderLayerRMSPre]`（pre-RMSNorm + MHA + FFN）。  
- **`use_rmsnorm=False`**：`nn.TransformerEncoder`（LN，`norm_first=True`）。

---

## 5. `--dual_ablation`（A/B 四套）

全程 **`use_revin=False`**。**A** = additive bias / **B** = RMS encoder。

| stem | `use_hist_add_bias`（HistAdd） | `use_rmsnorm`（RMS） | 图/终端展示名 |
|------|---------------------------|---------------------|----------------|
| `full` | True | True | SimDiff HistAdd+RMSNorm |
| `vanilla` | False | False | SimDiff vanilla |
| `a_only` | True | False | SimDiff HistAdd_only |
| `b_only` | False | True | SimDiff RMSNorm_only |

- Checkpoints：**`simdiff_weather_best_dual_<stem>.pt`**，`ablation_ckpt_suite=dual`。  
- **`--dual_reuse_b_only_ckpt`**：优先用 **`simdiff_weather_best_rmsnorm_only.pt`**，若没有则用 **`simdiff_weather_best_film_rmsnorm_only.pt`**（二者均为「仅 RMS、无 HistAdd」），复制为 **`…_dual_b_only.pt`** 并跳过 `b_only` 训练。
- 产出：**`bar_mae_mse_dual_ablation_*.png`**、**`forecast_dual_ablation_overlay_*.png`**（标题与纵轴在 `plot_metrics_bars` 中已缩短以防重叠）。

另：**`--revin_rms_ablation`**（RevIN+RMS 四套文件名 `best_<stem>.pt`，无 `_dual_`）与 **`--dual_ablation`** **互斥**。

---

## 6. 训练默认值（节选）

见 `config/config.py`：`learning_rate`、`epochs`、`train_amp`、`use_ema`、MoM、`forecast_num_samples` 等。

---

## 关键文件

| 内容 | 路径 |
|------|------|
| Config | `config/config.py` |
| NI | `utils/independent_normalizer.py` |
| RMS / RevIN / **HistoryAdditiveBias** | `models/revin_rms.py` |
| 去噪器 | `models/network.py` |
| SimDiff | `models/simdiff.py` |
| 双消融入口 | `main.py`：`run_dual_ablation_suite` |
