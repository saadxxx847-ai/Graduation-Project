# SimDiff-Weather（毕设项目）

基于 **SimDiff**思路的 **气象多变量长序列预测** 简化实现：使用 **Weather** 表格数据，在 `data/weather.csv` 上训练条件扩散模型，预测未来 `pred_len` 步。

> 论文参考：[SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting](https://arxiv.org/abs/2511.19256)  
> 官方代码框架说明见上级目录 `SimDiff/README.md`（本仓库为独立、可运行的课程/毕设版本，非官方仓库完整拷贝）。

---

## 项目结构

```
Simdiff_weather/
├── data/
│   ├── weather.csv          # 原始数据（已提供）
│   └── processed/           # 预留：可存放预处理 .npy
├── models/
│   ├── diffusion.py         # 余弦日程、加噪、DDPM 反向一步
│   ├── network.py           # 时间步嵌入 + Transformer 去噪网络
│   └── simdiff.py           # 归一化独立性 + 训练损失 + 采样预测
├── utils/
│   ├── data_loader.py       # CSV 读取、滑动窗口、划分 train/val/test
│   ├── normalizer.py        # 历史/未来独立归一化与推理用尺度估计
│   └── trainer.py           # 训练 / 验证 / 早停 / 保存权重
├── config/
│   └── config.py            # 超参数
├── main.py                  # 入口：训练 + 测试 + 示例曲线
├── requirements.txt
└── README.md
```

---

## 环境与依赖

```bash
cd Simdiff_weather
pip install -r requirements.txt
```

需要本机已安装 **PyTorch**（建议 2.x，有 GPU 则装 CUDA 版）。若使用虚拟环境：`python -m venv .venv && source .venv/bin/activate` 后再执行 `pip install`。

---

## 如何运行（修改后的项目）

按顺序在项目根目录 **`Simdiff_weather/`** 下执行（路径按你本机调整）。

**1）首次训练（推荐改完归一化逻辑后重新训一版权重）**

```bash
cd /path/to/Simdiff_weather
pip install -r requirements.txt          # 仅需一次
python main.py --epochs 50 --batch_size 64
```

- 默认配置：`config/config.py` 中 **`independent_future_normalization=False`**（历史与未来共用历史窗口 \(\mu,\sigma\)，训练与推理反归一化一致）。  
- 若曾用旧版代码训练过，请删除旧权重后再训，避免分布不一致：  
  `rm -f checkpoints/simdiff_weather_best.pt`

**2）快速试跑（调试用）**

```bash
python main.py --epochs 2 --batch_size 32
```

**3）仅评估（不训练，需已有训练好的 checkpoint）**

```bash
python main.py --eval_only
```

**重要**：`--eval_only` **不会更新权重**。若从未成功跑完训练，加载的仍是随机初始化或旧文件，**测试指标无意义**（MSE 可能极大）。正常实验请用 **`python main.py`**，不要用 `--eval_only` 代替训练。

旧参数名 `--skip_train` 已移除，请改用 `--eval_only`。

**4）输出位置**

| 输出 | 说明 |
|------|------|
| `checkpoints/simdiff_weather_best.pt` | 验证集最优模型 |
| `plots/forecast_example.png` | 测试集一条样本的预测曲线 |
| 终端日志 | `train_loss`、`val_mse`（噪声预测）、学习率；`Test noise MSE`；采样 MSE/MAE（多特征混合尺度，见 `main.py` 说明） |

**5）可选：独立未来归一化（论文叙述向 SimDiff「统计解耦」靠拢时）**

在 `config/config.py` 中设置 `independent_future_normalization = True`，再重新训练。此时无真值的 `forecast()` 与训练空间仍有近似误差；有标签评估以 `validation_mse` 为准（已用真值未来的统计量反归一化）。

---

## 问题记录与修改说明（维护日志）

以下为开发/调试过程中出现过的问题及对应改法，便于毕设答辩与后续维护。

### 1. `utils` 循环导入（ImportError）

- **现象**：`from models.simdiff import SimDiffWeather` 时报 `partially initialized module` /循环导入。  
- **原因**：`utils/__init__.py` 中同时 `import Trainer`，而 `trainer` 又引用 `models.simdiff`，与 `simdiff` 引用 `utils.normalizer` 形成环。  
- **修改**：`utils/__init__.py` 只导出 `make_loaders`、`Normalizer`，不再在包初始化时导入 `Trainer`。`main.py` 中直接 `from utils.trainer import Trainer`。

### 2. Test noise MSE 很低，但 Test generative MSE 极高

- **现象**：噪声预测损失很小（如 ~0.09），而「完整采样 + 原始尺度」的 MSE 达到数千。  
- **原因**：旧版训练用**未来窗口自身**的 \(\mu_f,\sigma_f\) 将 `future` 归一化，但 `forecast()` 用**历史窗口**的 \(\mu_h,\sigma_h\) 反归一化。二者不是互逆变换，采样结果在物理尺度上整体错位；噪声 MSE 仍在正确的归一化空间里，故仍可降低。  
- **修改**：  
  - **默认**：`normalize_pair(..., independent_future=False)`，历史与未来都用 **\(\mu_h,\sigma_h\)**标准化，`forecast()` 用同一组统计量反变换，与训练一致。  
  - **配置**：`config.independent_future_normalization`（默认 `False`）；为 `True` 时保留「未来独立统计」，且 **`validation_mse`** 改为用 `normalize_pair` 返回的 `f_mean`/`f_std`（有真值时）反归一化，不再错误地沿用仅基于历史的反变换。  
- **注意**：改过归一化策略后，必须用新策略 **重新训练**，旧 checkpoint 与新区间分布不一致。

### 3. NumPy 2.x 与 argparse（若同时使用上级目录 `SimDiff` 官方脚本）

- **`np.Inf` 被移除**：NumPy 2.0 起应使用 `np.inf`。若在 `SimDiff/utils/tools.py` 等仍写 `np.Inf`，训练会在 `EarlyStopping` 处报错。  
- **`run.py --help` 崩溃**：`help='... (%)'` 中 `%` 在 argparse 中需写成 `%%`。

本仓库 **`Simdiff_weather`** 未使用 `np.Inf`；若你在毕设里同时跑 `SimDiff/script/*.sh`，请确认上级 `SimDiff` 已按上述方式修补。

### 4. 误用「仅评估」导致指标灾难

- **现象**：从未训练或误用 `--skip_train` / `--eval_only`，测试 MSE 达万亿、MAE 极大。  
- **原因**：未经过梯度更新的权重等价于随机预测，不是「模型差」，而是**没有学习**。  
- **修改**：默认流程为 **先训练再测**；`--eval_only` 会打印醒目提示；旧参数 `--skip_train` 已删除，统一为 `--eval_only`。

### 5. 训练 Loss 数万、测试 MSE 达 \(10^{12}\) 量级

- **现象**：`train_loss`、`val_mse` 极大（如数万），`Test generative MSE` 天文数字。  
- **原因**：在 **滑动窗口内** 按通道算标准差时，Weather 等数据中多列在96 步内**几乎常数**（如长期为 0 的雨量、辐射），\(\sigma \to 0\)；即使 `clamp_min(1e-5)`，归一化后仍会出现 **\(10^4\)～\(10^6\)** 量级的 `fut_n`，扩散目标与网络输出尺度失控，MSE 爆炸。  
- **修改**：1. **默认启用训练集全局标准化**（`config.use_global_standardization=True`）：在 **整个训练段**上估计每维 `mean/std`，并对 `std` 设下限（`fit_global_standardizer`：`max(经验std, 1e-3, 1e-4×max(|mean|,1))`）。历史与未来统一用该 `(μ,σ)` 变换，`forecast` / `validation_mse` 与训练一致。  
  2. 读入数据后 `np.nan_to_num`，避免异常值。  
  3. 采样末尾对归一化空间输出按 **`z_clip+2`** 做 `clamp`（与训练时 Z 分数尺度一致），减轻未收敛时采样漂移。  
- **注意**：改过标准化后请 **删除旧 checkpoint并重新训练**。关闭全局标准化可设 `use_global_standardization=False`（不推荐）。

### 6. 采样发散：验证 Loss 正常但测试 MAE 飙升、预测曲线跌至不合理温度

- **现象**：`val_mse`（噪声预测）仍较低（如 ~0.038），但全测试集 MAE 从十几飙到几十甚至上百；示例曲线上预测在若干步后出现**剧烈锯齿**并**整体偏低**（如跌至 -10°C，而真值在 2°C～7°C）。
- **原因归纳**（本仓库已针对性修补）：
  1. **DDIM 子步数 `sampling_steps` 大于 `timesteps`**：离散噪声日程只有 `timesteps` 档。旧实现用 `linspace` 会产生大量 **重复时间索引** `(t, t)`，循环中再 `continue` 跳过，**有效去噪步数远少于预期**，轨迹与训练假设不一致，易数值失控。
  2. **DDIM 在 `α_t` 很小时，`pred_x0 = (x - √(1-α)ε)/√α` 无界**，少量误差会被放大，导致归一化空间爆炸，再经 `× global_std + global_mean` 反归一化后在物理尺度上极端偏移。
  3. **反归一化**：`forecast()` 使用模型 buffer `_g_mean` / `_g_std`（与 checkpoint 一并保存）。若数据划分或 CSV 变更后 **未重训** 却混用旧权重，或误用配置，可能出现尺度不一致；`--eval_only` 时程序会打印 **checkpoint 与当前数据估计的 global_mean 最大差** 供排查。
- **修改**：
  1. **`models/diffusion.py`**：`build_ddim_time_pairs()` 将 `sampling_steps` **截断到不超过 `timesteps`**，并去掉连续重复时间；DDIM 每步对 **`pred_x0` 与状态 `x`** 在归一化空间按 `clamp_abs`（与 `z_clip` 对齐）裁剪。
  2. **默认 `sampling_mode=ddpm`**：与训练时的反向随机一步一致，通常最稳；若用 DDIM，勿令 `sampling_steps > timesteps`。
  3. **调试**：`config.sample_debug=True` 或命令行 `--sample_debug`，在采样循环中打印归一化空间张量的 `min` / `max` / `mean`，定位从哪一步开始异常。
  4. **训练损失权重**略降（`training_noise_l1_weight` / `training_noise_temporal_diff_weight`），减轻辅助项对噪声主任务的干扰（需重新训练后更可靠）。
- **建议流程**：删掉错误实验的 checkpoint → 用当前默认配置 **重新训练** → 评估时用 **`python main.py --eval_only`**（或完整跑 `main.py`）；若仍异常，加 **`--sample_debug`** 看采样轨迹。

---

## 数据说明

- `weather.csv` 第一列为 `date`，其余 **21 列**为数值特征（程序自动识别 `input_dim=21`）。  
- 滑动窗口：长度为 `seq_len` 的历史 → 预测 `pred_len` 步未来。  
- 按时间顺序 **70% / 15% / 15%** 划分训练 / 验证 / 测试（不打乱时间）。

---

## 核心设计（与开题/模块说明对应）

1. **归一化**（`utils/data_loader.py` + `utils/normalizer.py` + `models/simdiff.py`）  
   - **默认（推荐）**：`use_global_standardization=True`，在**训练集全时段**上估计每通道 `global_mean` / `global_std`，历史与未来统一用其标准化，推理时用同一组参数反变换，避免窗口内近常数列导致爆炸。  
   - **`use_global_standardization=False`**：退回仅按**当前窗口**统计量（易在稀疏列上数值爆炸，见上文问题4）。  
   - **`independent_future_normalization=True`**：在全局（或窗口）前提下，未来仍可用自身 \(\mu_f,\sigma_f\)（与 `simdiff` 中分支一致）；无真值时 `forecast()` 仍有近似误差。

2. **扩散**（`models/diffusion.py`）  
   余弦 \(\beta\) 日程；训练目标为预测噪声 \(\epsilon\) 的 MSE。

3. **去噪网络**（`models/network.py`）  
   历史 token 与加噪未来 token 拼接，加入时间步嵌入，**TransformerEncoder** 输出与未来同形状的噪声预测。

4. **验证指标**  
   验证集使用与训练相同的 **噪声 MSE**（全扩散采样较慢，故早停不用采样）。  
   测试脚本中会额外对 **一个 batch** 做完整采样并报告原始尺度 MSE/MAE（**需训练充分**；MAE 为 21 维特征混合平均，不等价于「温度误差几度」）。

5. **训练稳定性（`utils/trainer.py` + `config`）**  
   学习率默认 **5e-5**，**梯度裁剪** `grad_clip_max_norm=0.5`，**ReduceLROnPlateau** 在验证不降时减半学习率；跳过非有限 `loss` 的 batch；checkpoint 内保存 `meta`（含 `global_mean/std` 等）便于追溯。

---

## 调参建议（毕设实验）

在 `config/config.py` 中可改：

| 字段 | 含义 |
|------|------|
| `seq_len` / `pred_len` | 历史/预测长度 |
| `d_model` / `n_heads` / `n_layers` | 网络容量 |
| `timesteps` | 扩散步数（越大越慢，一般100–500） |
| `sampling_mode` | `ddpm`（默认，稳）或 `ddim` |
| `sampling_steps` | 仅 DDIM：推理子步数，**勿大于 `timesteps`** |
| `sample_clip_pred_x0` | DDIM 每步是否裁剪预测的 x0，默认 True |
| `sample_debug` | 是否在采样循环打印张量 min/max/mean |
| `epochs` / `batch_size` / `learning_rate` | 训练设置 |
| `z_clip` | 标准化后裁剪到 `±z_clip`，默认 4 |
| `grad_clip_max_norm` | 梯度裁剪范数，默认 0.5 |

---

## 引用

若论文中引用 SimDiff，请使用官方 BibTeX（见 `SimDiff/README.md` 底部）。
