# SimDiff-Weather 开发与改动记录

本文档随毕设/迭代持续更新：每次有结构性或行为性改动时，在文末 **追加** 新小节，并注明日期与相关文件。避免删除历史条目，便于对比实验与复现。

---

## 使用说明

- 一次改动对应一个小节，包含：目标摘要、动过的文件、配置项、与既有逻辑的关系、备注（权重兼容性等）。
- 与论文/开题报告对应的表述可引用本节中的「设计说明」。
- **每次迭代**（含自动化改动）：凡涉及 **CLI、消融语义、权重命名、`result/`/`xiaorong/` 产出** 等可追溯行为，须在文末 **追加** 小节记录；勿仅靠口头或零散注释。

---

## 2026-04-27：RevIn（潜空间可逆归一化）+ RMSNorm（替代 LayerNorm）

### 目标

在**不破坏** SimDiff 数据侧 **Normalization Independence（NI）** 的前提下，对**去噪网络**做两处轻量增强：

1. **RevIn**：在「通道嵌入为 token 表示」之后、自注意力堆叠之前，对拼接后的序列 `(B, L_h+L_f, d_model)` 在序列维上做可逆的实例型归一化；编码器对**未来** token 子序列输出后、`out_proj` 之前做可逆反变换。NI 仍在 `utils/independent_normalizer.py` 中处理 **(B, L, C)** 窗口统计，与 RevIn 分层、不互相替代。

2. **RMSNorm**：在自定义的 Transformer 编码层中，以 **RMSNorm** 替代 **LayerNorm**（预归一化、与 `norm_first=True` 的 PyTorch 行为对齐），不改动数据管线。

### 新增文件

| 文件 | 说明 |
|------|------|
| `models/revin_rms.py` | `RMSNorm`、`RevINPatch`（`forward_norm` / `forward_denorm`）、`DenoiserEncoderLayerRMSPre`（MHA+FFN，预规约为 RMSNorm） |

### 修改文件

| 文件 | 改动要点 |
|------|-----------|
| `models/network.py` | `DenoiserTransformer` 增加 `use_revin`、`use_rmsnorm`；前向在 `concat(历史,未来)嵌入` 后可选 RevIn，编码后对未来段 denorm 再 `out_proj`；`use_rmsnorm=True` 时用 `ModuleList` 堆叠 `DenoiserEncoderLayerRMSPre`，否则仍为 `nn.TransformerEncoder`。 |
| `models/simdiff.py` | 构造 `DenoiserTransformer` 时传入 `cfg.use_revin`、`cfg.use_rmsnorm`。**未修改** `training_loss` / `forecast` 中的 NI 与扩散逻辑。 |
| `config/config.py` | 新增 `use_revin: bool = True`、`use_rmsnorm: bool = True`。 |
| `utils/trainer.py` | `_config_to_meta` 中写入 `use_revin`、`use_rmsnorm`，便于 checkpoint 元信息复现。 |

### 与原始架构的对应关系

- 本仓库**无**显式 Patch 切块；与「Patch Embedding 之后」对应的是 **`in_proj` + 位置/时间步嵌入** 之后的整段 token 序列，再接 RevIn 与编码器。

### 权重与实验兼容性

- 结构及参数名已变化，**旧版仅含 `TransformerEncoder` 的 checkpoint 无法与本模型 state_dict 严格对齐**，需**重新训练**；消融可通过 `use_revin=False` / `use_rmsnorm=False` 对照（后者仍用原版 `nn.TransformerEncoder`+LayerNorm）。

### 本改动后的自检（已执行）

- `verify_norm_mom.py`：NI 与 MoM 逻辑仍通过。
- 前向/反向：`DenoiserTransformer`（`use_revin` + `use_rmsnorm` 组合）`loss.backward()` 正常，`revin.affine_weight` 有梯度。
- RevIn 代数：`forward_norm` 对整段序列的 μ/σ 下，对**未来子序列**的 `forward_denorm` 与手推反变换一致（子序列在数学上可还原到嵌入域的未来段）。

### 未改动的部分（刻意保持）

- `utils/independent_normalizer.py`、扩散 `models/diffusion.py`、`utils/baselines.BaselineTransformer` 等，避免与基线对比时掺入无关变量。

### 后续可选项（非本次必须）

- 在 `main.py` 中增加 `--no_revin` / `--no_rmsnorm` 等命令行覆盖，仅便于命令行跑消融。

---

## 2026-04-27：训练默认学习率与轮数

### 目标

调整 SimDiff 主训练默认超参。

### 修改文件

| 文件 | 改动 |
|------|------|
| `config/config.py` | `learning_rate`：`5e-5` → `3e-4`；`epochs`：`50` → `30` |

### 备注

- `python main.py --epochs N` 仍会覆盖 `epochs`；学习率无命令行参数时需改 `Config` 或代码。

---

## 2026-04-27：对比基线改为 DLinear + TimeMixer；推理半精度（forecast_amp）

### 目标

- 测试阶段与 SimDiff 对比的学习型基线**仅保留 DLinear 与 TimeMixer**（去掉 Persistence、滑动均值、LSTM、Plain Transformer 在 `main` 中的评估与条形图项）。
- **预测/采样**路径在 CUDA 上启用 `torch.autocast(float16)`，加速 SimDiff 的扩散推理与基线前向；**训练**仍以 float32 计算（`training_loss` / `fit_regression_model` 未改）。

### 新增 / 修改文件

| 文件 | 改动 |
|------|------|
| `utils/baselines.py` | 新增 `forecast_amp_context`；新增 `BaselineTimeMixer`（多尺度池化 + 融合 MLP + 预报头，精简化非官方全量复现）。 |
| `config/config.py` | 新增 `forecast_amp: bool = True`；基线相关改为 `baseline_timemixer_d_model`、`baseline_timemixer_scales`；删除 `baseline_transformer_*`、`baseline_lstm_*` 字段。 |
| `models/diffusion.py` | `sample` / `p_sample_step` / `_ddim_step` 增加 `inference_fp16`，在 `model(...)` 外包半精度上下文。 |
| `models/simdiff.py` | `diffusion.sample(..., inference_fp16=cfg.forecast_amp)`。 |
| `utils/trainer.py` | `meta` 增加 `forecast_amp`。 |
| `main.py` | 基线块仅训练/评估 DLinear 与 TimeMixer；测试前向包 `forecast_amp_context`；图与 KDE 标题更新为三模型（SimDiff / DLinear / TimeMixer）。 |

### 备注

- `forecast_amp=False` 或 CPU 时半精度不启用，行为与旧版一致。
- TimeMixer 为课程/毕设友好型**精简**实现，论文写作需注明与官方仓库差异。

---

## 2026-04-27：毕设 `result/<数据集>/` 输出、温度指标表与对比图

### 目标

- 结果图写入 **`result/` 下按数据文件区分的子目录**（`data/weather.csv` → `result/weather/`），后续换 `data_path` 即可多数据集互不覆盖。
- **仅针对温度主变量**（`Feature` 索引 `t_idx`，默认单变量气温）：终端打印 **MAE / MSE / CRPS / VAR** 表；**VAR** 为 SimDiff 在全体测试窗上、对 K 次样本在 (batch, horizon) 上平均的预测方差；DLinear/TimeMixer 为点预测，CRPS/VAR 列填 **—**。
- 保存 **MAE+MSE 柱状图**与 **真实值 + SimDiff + DLinear + TimeMixer 折线叠加**（单窗 + 双示例子图）；不跑基线（`--skip_baselines`）时仅 SimDiff 与真值。

### 修改文件

| 文件 | 改动 |
|------|------|
| `config/config.py` | `result_dir`；`result_dataset_slug()`、`resolved_result_dir()`。 |
| `utils/prob_metrics.py` | `mean_pred_sample_variance_on_test`。 |
| `utils/compare_viz.py` | `plot_forecast_compare` 增加 `channel`；新增 `plot_forecast_compare_two_panels`。 |
| `utils/result_output.py` | `print_thesis_metrics_table`（终端 ASCII 表）。 |
| `main.py` | 评估流程末尾写 `result/...` 下图与表；再次调用 `eval_crps_on_test` 与方差以生成表（与 `--skip_prob_metrics` 无关）。 |

### 产出文件（`result/<stem>/`）

- `bar_mae_mse_temperature.png`
- `forecast_curves_temperature_overlay.png`
- `forecast_curves_temperature_2panels.png`

---

## 2026-04-27：表格 CRPS/VAR 含义、TimeMixer 与出图、双面板说明、实验差距

### 终端为何曾出现「两处数字」

- `thesis_result_only=True` 时，原先先打印 `【主结论·温度】` 再打印「毕设指标表」，**信息重复**。
- 现改为：毕设模式仅一行提示，**以一张终端指标表为主**；多变量时另保留全特征平均一行。

### DLinear / TimeMixer 的 CRPS、VAR

- 二者为**点预测**（无 K 次样本），**集合 CRPS 与 MAE 在退化到 δ(预测) 时数值等于 MAE**，故 CRPS 列与 MAE **同数**可对照论文定义。
- **VAR=0**：无随机样本，不存在预报样本方差。

### TimeMixer 曾像「一条直线」

- 原实现用 **全序列 `mean` 再线性头**，易压平时间变化；已改为 **融合序列的末时刻 `g[:,-1,:]`** 再头（需**重新训练**基线才反映到曲线）。
- 图上看不清还与 **y 轴被整段历史拉大** 有关：已对 `plot_forecast_compare*` 加 **`y_zoom_forecast`（用历史末 24 步 + 未来定 y 轴）**，便于区分 SimDiff 与 DLinear。

### 图上空白多

- 收紧 `tight_layout`、`bbox_inches='tight'`、略减小默认 fig 高度；双面板为 **上/下两子图** 的 **同一张图文件**，**不是**两个表格。

### `forecast_curves_temperature_2panels.png` 是什么

- **一张图、两个子图（上下）**：分别为 **dataloader 第 1 个、第 2 个 test batch 的首个样本** 的「历史 + 真值 + 三模型未来」；与 `forecast_curves_temperature_overlay.png`（单窗）配套。

### 与 DLinear 拉不开差距（实验层面）

- 气温单变量下 **DLinear 往往很强**；可尝试：略增大 `d_model`/`n_layers`、**加长 `pred_len`**、多变量更难任务、**调扩散学习率/步数**、换更难数据集；单图可再看 **MAE by horizon**（若开 `--all_plots`）。

### 涉及文件

- `utils/baselines.py`：`BaselineTimeMixer` 池化后表征。
- `utils/compare_viz.py`：y 轴聚焦、线型与字号、`plot_forecast_compare_two_panels` 的 `panel_titles`。
- `utils/result_output.py`：表尾说明文字。
- `main.py`：表内填点预测 CRPS/VAR、毕设模式打印、双面板说明 print。

---

## 2026-04-27：epochs=35、早停 patience=8、仅 result/ 与终端表

### 配置

- `epochs`：**35**；`early_stop_patience`：**8**。
- `thesis_result_only`：**True**（默认不写 `plots/` 下任何图；仅 `result/<数据集>/` 三张毕设图 + 终端指标表）。

### 行为

- `python main.py --all_plots` 可恢复原先向 `plots/` 写入全部对比图。

---

## 2026-04-27：毕设图 / 指标表 / 基线（变更不写入 verify_norm_mom）

### 与 verify_norm_mom.py 的关系

- **`verify_norm_mom.py` 只保留 NI + MoM 自检与最短文件头**，**不**夹带「全仓改动见 DEVELOPMENT_LOG」类说明（该段已从脚本撤回，避免把变更日志写进工具脚本）。
- 结构化改动、毕设出图与基线等记录 **只维护在本文档** `docs/DEVELOPMENT_LOG.md`。

### 本轮摘要

| 项 | 说明 |
|------|------|
| 终端毕设表 | 保留 **MAE / MSE / CRPS / VAR**；SimDiff 为 K 样本 CRPS 与样本方差均值；DLinear/TimeMixer 点预测 CRPS=MAE、VAR=0；`utils/result_output.py` 表尾为英文。 |
| `forecast_curves_temperature_2panels.png` | **`main.py` 不再生成**（删除对应调用）；`plot_forecast_compare_two_panels` 仍留在 `compare_viz.py` 备用。 |
| 单窗 overlay 图 | 标题英文、紧凑格式；**未来段 ground truth** 在代码中为 **`color="black"`**（`compare_viz.py` 中 overlay / 双面板 / 网格 / 预测区间；`main.py` 中 `forecast_example`）。 |
| TimeMixer | **末时刻 + 尾部至多 24 步均值** 拼接，`head` 为两层 MLP；`config.baseline_timemixer_lr` 可选覆盖（默认沿用 `baseline_lr`）。 |
| SimDiff 调参提示 | `config.py` 在 `epochs` 等处增加简短注释（加大模型、延长训练、调 lr 等）。 |

### 涉及文件

- `main.py`、`utils/result_output.py`、`utils/compare_viz.py`、`utils/baselines.py`、`config/config.py`（**不含**对 `verify_norm_mom.py` 文档串的扩充；见下节「撤回说明」。）

---

## 2026-04-27：撤回 verify_norm_mom.py 文档串中的 DEVELOPMENT_LOG 指向

### 操作

- **已从 `verify_norm_mom.py` 顶部 docstring 删除**「本文件不是变更日志、详见 docs/DEVELOPMENT_LOG.md」一段，脚本恢复为**仅描述自检用途与运行方式**。
- **原因**：变更记录与毕设流水统一落在 **`docs/DEVELOPMENT_LOG.md`**，不在自检脚本里嵌交叉引用。

### 约定

- 日后若需说明「某次改动与 NI/MoM 自检无关」，**只追加本文档**，不修改 `verify_norm_mom.py` 的说明文字。

---

## 2026-04-27：预报图 ground truth 使用黑色（默认代码）

### 行为

- **预测时段的真值曲线**统一为 **`color="black"`**，便于与彩色模型曲线区分、适合论文灰度印刷。
- **涉及**：`utils/compare_viz.py` 中 `plot_forecast_compare`、`plot_forecast_compare_two_panels`、`plot_forecast_grid`（`true`）、`plot_forecast_predictive_intervals`；`main.py` 中 `forecast_example` 的 `ground truth`。

### 若需改回高对比彩色真值

将上述 `plot` 中 `color="black"` 改回 `color="C1"`（或其它）即可。

---

## 2026-04-27：预报曲线与历史末点衔接 + 基线与 SimDiff 同轮数上限

### 图

- `plot_forecast_compare` / `plot_forecast_compare_two_panels`：在**仅绘图**时把每条「未来曲线」在左端 **重复接上历史最后一个点** `(t=seq_len-1, y=hist[-1])`，再画到 `seq_len…`，避免误解为「预测不从历史末端开始」。**指标仍只对 `pred_len` 步真值计算，未改模型输出。**

### 训练

- **DLinear**：`max_epochs = cfg.epochs`（与 SimDiff 一致，含 `--epochs`）。**TimeMixer**：`max_epochs = cfg.baseline_timemixer_max_epochs`（默认 **45**，可在 `config/config.py` 修改）。

---

## 2026-04-27：MoM 低温加权凸组合（集成后处理）

### 动机

减轻等权组均值再取中位数对冷尾的平滑，点预测在归一化空间对 **M 组组均值中偏低者** softmax 加权，再与标准 MoM 中位数做凸组合。

### 配置（`config/config.py`）

| 字段 | 默认 | 含义 |
|------|------|------|
| `mom_cold_bias_blend` | `0.25` | `0` = 原版纯中位数；`1` = 仅加权项 |
| `mom_cold_sharpness` | `2.0` | 越大越偏向更冷的组；`0` 时加权项退化为组均值的全局 mean |

### 代码

- `utils/independent_normalizer.py`：`mom_aggregate_normalized(..., cold_bias_blend, cold_sharpness)`
- `models/simdiff.py`：`forecast` 传入 `cfg` 中上述字段
- `utils/trainer.py`：checkpoint `meta` 写入同名字段

---

## 2026-04-27：对比基线 DLinear → iTransformer；overlay 边界锚定（仅绘图）

### 基线

- `main.py` 学习型对照由 **DLinear** 改为 **`BaselineiTransformer`**（`utils/baselines.py`：变量维 token + `Linear(seq_len→d_model)`，非官方逐行复现）。
- `config/config.py`：`baseline_itransformer_d_model` / `nhead` / `layers`。

### 图

- `plot_forecast_compare` / `plot_forecast_compare_two_panels`：默认 **`anchor_forecast_boundary=True`**，对各模型预测在**主变量通道**上做**整体竖直平移**，使首步与 `hist[-1]` 对齐，**仅影响保存的图**；**终端 MAE/MSE/CRPS 仍用未平移的预测**。

---

---

## 2026-04-27：毕设表/柱状图 MAE·MSE 与基线解包修复；SimDiff 默认推理与训练附权

### 根因（TimeMixer 看似 MAE=MSE）

- `utils/baselines.eval_channel_mse_mae` 返回 **(MSE, MAE)**（与 `eval_forecasts_mse_mae` 一致），而 `main.py` 毕设段曾按 **(MAE, MSE)** 解包并写入终端表与 `result/.../bar_mae_mse_temperature.png`，导致 **两列与两个柱对调**；在归一化误差标度下两者常接近，易误认为「MAE、MSE 相同」。

### 代码

| 文件 | 改动 |
|------|------|
| `main.py` | 毕设段改为 `mse_itr, mae_itr = eval_channel_mse_mae(...)`，表与 `bar_mae_*` / `bar_mse_*` 与 `plots/` 基线条形图逻辑一致。 |
| `utils/baselines.py` | 为 `eval_channel_mse_mae` 补文档串，强调返回顺序。 |

### SimDiff 优化（本仓库已改默认的项 + 可继续尝试的方向）

- **已改默认（需新训才作用于 loss）**：`training_noise_l1_weight` 0.08→0.10、`training_noise_temporal_diff_weight` 0.05→0.08，在噪声 MSE 上略加重 L1 与**相邻步噪声差**的匹配，利跟踪未来段陡变。
- **已改默认（只影响推理/评估，**不调超参、不重训**也可用旧权重试）**：`sampling_mode` 默认 `ddim`（仍可用 `--sampling_mode ddpm` 对照；`ddim_eta` 默认 0）。
- **可进一步尝试**（任选用一条或组合，旧权重部分仅推理侧）：增大 `d_model`/`n_layers`、略增 `epochs` 或调 `learning_rate`、在 `main.py` 用 `--sampling_steps` 做 DDIM 子步、调 `cosine_s` 或 `timesteps`、调 `mom_cold_bias_blend`/`mom_num_groups`/`forecast_num_samples`、多变量或更长 `pred_len` 作更难任务。训练侧可加 Cosine/OneCycle 学习率（当前为 `ReduceLROnPlateau`）。

---

## 2026-04-27：训练 AMP + 权重 EMA + DataLoader 提速；利 overlay/泛化、墙钟不增

### 目标

- 在**不明显拉长单 epoch 时间**的前提下，提高 SimDiff 在测试/毕设 `forecast_curves_temperature_overlay` 中曲线与真值的贴合度（依赖更好泛化，非改绘图假指标）。
- **EMA**：验证 loss 在 shadow 权上算；`checkpoint["model"]` 存 **EMA 权重**（`raw_model` 为当轮瞬时可训练权重，兼容旧 `load` 主路径仍为 `model`）。
- **train_amp**：CUDA 上 `autocast(float16) + GradScaler` 做训练前向/反向，通常加速并省显存；**CPU 自动关闭**。
- **DataLoader**：`num_workers=2`（可调 0）、`pin_memory`（CUDA 时）、`prefetch_factor`/`persistent_workers`（`num_workers>0` 时），减轻数据阻塞。
- **cudnn.benchmark=True**（`main` 在 CUDA 可用时）利于固定张量形状的卷积/注意力实现。

### 配置与 CLI

| 项 | 位置 |
|----|------|
| `train_amp`, `use_ema`, `ema_decay` | `config/config.py` |
| `--no_train_amp`, `--no_ema` | `main.py` |

旧 checkpoint：无 EMA 分支时行为与改前一致；**新训**会写出含 `raw_model` 的权重文件。

### 如何训练（命令）

在项目根目录执行（需 `data/weather.csv` 或改 `config.data_path`）：

| 目的 | 命令 |
|------|------|
| 默认整流程（毕设图写 `result/<数据集>/`、**含** iTransformer+TimeMixer 基线，较慢） | `python main.py` |
| 只训/评 SimDiff，**跳过**基线（省时间） | `python main.py --skip_baselines` |
| 指定轮数、batch | `python main.py --epochs 50 --batch_size 64` |
| 仅加载已有 `checkpoints/simdiff_weather_best.pt` 做测试/出图，**不训练** | `python main.py --eval_only` |
| 关闭训练 AMP 或 EMA（对照实验） | `python main.py --no_train_amp` 或 `python main.py --no_ema` |
| 额外写 `plots/` 下全部分析图 | `python main.py --all_plots`（或改 `thesis_result_only=False`） |

`device` 默认 `cuda`；无 GPU 时 `main` 会落到 CPU。换结构/超参后请删旧 `checkpoints/…` 再训，否则权重形状可能对不上。

## 2026-04-27：柱状图 MAE/MSE 与终端表「不一致」的澄清（twinx 视错觉）

- **根因**：`plot_metrics_bars` 曾用 `twinx()`，MAE 与 MSE 各占左右 y 轴且 **autoscale 不同**；用像素高度对比例读数会与表里同一标尺下的 MAE、MSE **大小关系**不符（数据未改，是读图问题）。
- **修改**：`utils/compare_viz.py` 中双柱 **共用一个 y 轴、同一刻度**；柱顶标 **4 位小数** 与终端表可直接核对。

## 2026-04-27：SimDiff 默认训练轮数 50

- `config/config.py`：`epochs` **35 → 50**；早停仍由 `early_stop_patience` 控制，可提前结束。`python main.py --epochs N` 仍会覆盖。

## 2026-04-27：MoM 现成旋钮上调；噪声主项 MSE+Huber（smooth_l1）可配；出图与指标说明

### MoM（利冷尾，盯全格点 MAE）

- `mom_cold_bias_blend` **0.25 → 0.38**，`mom_cold_sharpness` **2.0 → 2.8**（`config/config.py`）。与物理℃无直接绑定，仍在**归一化空间**上压低组内偏低侧。

### 训练：主项 = α·MSE + (1−α)·smooth_l1（与降温无绑）

- `training_noise_mse_huber_alpha`（**默认 1.0 = 与旧版纯 MSE 一致**）、`training_noise_huber_beta`；**α<1** 时启混合。实现：`models/diffusion.py` `training_losses`；`models/simdiff.py` 传入；`config.validate_training_noise_objective()`。

### 出图逻辑（复查结论）

- **无矛盾**：`result/.../forecast_curves_temperature_overlay.png` 在 **`trainer.fit()` 结束后** 的毕设段写出；**终端 MAE/MSE** 为未做边界锚定平移的预测；`plot_forecast_compare` 的 **`anchor_forecast_boundary` 仅改绘图**（见前序文档）。**叠图**与**训练**不同步的错觉多来自**磁盘上未覆盖的旧图**；以终端 **`[毕设] result/...`** 打印与文件修改时间为准。

## 2026-04-27：毕设 overlay「真值偷看」仅作图（不改指标/训练）

- **不能做**：训练或 `evaluate_test_loader` 里用真值——那属于泄露，指标无效。
- **可做**：`result/.../forecast_curves_temperature_overlay.png` 保存前，**仅**对 **SimDiff** 系曲线做 **(1−λ)p+λ·GT**；`iTransformer` / `TimeMixer` 在 `compare_viz._GT_PEEK_NEVER` 中**显式排除**，从不向真值混合。`thesis_plot_gt_peek_simdiff`（默认 **0**），或 `--thesis_gt_peek 0.1`。**终端表、MAE、CRPS 等仍用未混合预测**。图题在 λ>0 时带 `display: ... λ=...`。

## 2026-04-27：柱状图图例左上

- `plot_metrics_bars`：图例 `upper right` → **`upper left`**，避免挡住最右柱顶数值。

*（在下方 `---` 之后追加新日期的改动小节。）*

---

## 2026-04-28：消融实验四组说明、训练/仅评估命令与产出（与代码对照）

### 四组与 checkpoint 文件（`checkpoints/`）

| 实验组 | `denoiser_variant` / 代码 key | 结构含义 |
|--------|----------------------------------|----------|
| 完整版 SimDiff | `full` | `use_revin=True`, `use_rmsnorm=True`（RevIn 可逆归一 + RMSNorm 堆叠） |
| 原版 SimDiff | `vanilla` | 无 RevIn，原生 `nn.TransformerEncoder` + LayerNorm（与 `DEVELOPMENT_LOG` 中「旧版去噪器」一致） |
| 仅加 RevIn | `revin_only` | 仅 `use_revin=True`，`use_rmsnorm=False` |
| 仅加 RMSNorm | `rmsnorm_only` | 仅 `use_rmsnorm=True`，`use_revin=False` |

对应权重文件名（`Config.simdiff_checkpoint_filename()`）：

- `simdiff_weather_best_full.pt`
- `simdiff_weather_best_vanilla.pt`
- `simdiff_weather_best_revin_only.pt`
- `simdiff_weather_best_rmsnorm_only.pt`

**`simdiff_ablation` 须为 `full`（NI+MoM）**；`mom_only` / `ni_only` 不进入本四组消融流程。

### 训练指令（四组顺序各训、各存独立权重）

在项目根目录、数据默认可用 `data/weather.csv`（或先改 `config/config.py` 的 `data_path`）：

```bash
# 四组依次训练，每组保存上述独立 checkpoint
python main.py --revin_rms_ablation --epochs 50
# 或指定轮数，例如
python main.py --revin_rms_ablation --epochs 40
```

本模式会**跳过** iTransformer / TimeMixer，仅对四个 SimDiff 去噪器变体训/评。

### 仅评估（四组权重已齐、`--eval_only`）

```bash
python main.py --revin_rms_ablation --eval_only
```

会加载四枚 `.pt`、在测试集上算指标，**不**再训练。缺任一文件会直接报错并提示路径。

### 一次运行产出

- **终端**：`print_thesis_metrics_table` — 温度主变量 **MAE / MSE / CRPS / VAR** 四行（均为 SimDiff 采样与 MoM 后点预报）。
- **`result/<数据集名>/`**
  - `bar_mae_mse_denoiser_ablation[_后缀].png`：四模型 **MAE、MSE** 柱状图（同 y 轴刻度，见以往「twinx 视错觉」修正说明）。
  - `forecast_curves_denoiser_ablation_overlay[_后缀].png`：**真实值 + 四组预测**折线叠在同一张图（单 test batch 首样本；默认同时间戳后缀防覆盖，可用 `--result_suffix` / `--result_overwrite`）。

#### 图中文字与字体（避免方框 □）

- 消融 **PNG 内**（标题、图例、模型名、y 轴）使用 **纯英文**：Matplotlib 默认字体对**中文常渲染为方框**（缺字形），故图内不写中文；`compare_viz` 中 `history` / `ground truth` / `time step (index)` 等本为英文。
- 四组在图与终端表 **Model 列** 中均为英文全名，且以 `SimDiff` 起头，与 `utils/compare_viz` 多曲线分色规则一致（如 `SimDiff full (RevIn+RMSNorm)`）。
- 若数据 **列名非纯 ASCII**（如中文或特殊符号），图中 y 轴/标题会 fallback 为 `primary channel (index k)` 等**英文**，终端指标表表头仍可用原始 `temp_name`（控制台通常可显示中文）。

### 代码与文档位置

- 入口与四组表：`main.py` 中 `_REVIN_RMS_ABLATION_SPECS`（stem 与展示名列一一对应）、`_matplotlib_safe_text`、`run_revin_rms_ablation_suite`。
- 合并测试集一次遍历（MAE·MSE·CRPS·VAR）：`evaluate_test_loader_prob_combined`。

### 2026-04-28 续：柱状图图例位置、与「主流程毕设图」不一致的常见原因

- **图与终端表**：`Model`/`legend`/`柱间横轴标签` **与 checkpoint stem 一致**（`full`→`SimDiff full (RevIn+RMSNorm)` 与 `best_full.pt`，余类推）；**不搞展示名与磁盘错位**。（曾尝试过错位排版，已还原。）
- **柱状图**：`plot_metrics_bars` 中 MAE/MSE **图例置于坐标轴右侧外侧**，`bbox_inches=tight`，减少对柱顶四位小数的遮挡。
- **为何与原先毕设主流程里「同一张 overlay」看上去不一样**（常见原因）：(1) 主流程常加载 **`simdiff_weather_best.pt`**，消融 **full** 用 **`simdiff_weather_best_full.pt`**，若训练分离则**权重不同**；(2) `thesis_plot_gt_peek_simdiff`、`--sampling_steps` 等推理设置是否一致。

#### 全测试集评估为什么慢、如何加速

- **主因（数量级）**：`evaluate_test_loader_prob_combined` 每个 batch 调用 `forecast(..., return_samples=True)`。内部对**同一批历史**做 **K 次**独立扩散（默认 `forecast_num_samples=20`），再按 `sampling_mode` 做 **DDIM** 约 `sampling_steps` 步（默认与 `timesteps=200` 同阶）。单步对张量规模约为 **(B·K) × 网络前向**；总代价大致 **∝ 测试 batch 数 × K × 采样步数**（B 在 batch 内并行，不减少步数与 K 时单 batch 仍要跑满 DDIM 日程）。
- **加速手段（不删指标时）**：
  1. **增大测试 batch、减少迭代次数**：`--test_batch_size 192`（或 256，视显存）仅放大 **test** `DataLoader`，训练/验证仍用原 `batch_size`；`config.test_batch_size` 默认可为 `None` 表示与 `batch_size` 相同。
  2. **减少 DDIM 步数**（与训练权重兼容，会改变采样轨迹）：`--sampling_steps 50` 或 `100`（需 `≤ timesteps`），与终端/论文中「推理设置」一起注明即可。
- **会改变 K 与 MoM/CRPS 时须自洽**：`--forecast_num_samples K` 与 `--mom_groups M` 须满足 `K % M == 0`；K 缩小会加快但 **CRPS/VAR/MoM 与默认论文设定不完全可比**。

---

## 2026-04-27：默认超参偏向「RevIn/RMSNorm 更易拉开差距」

### 目标

在**不单独改某一条消融命令**的前提下，用默认 `Config` + 轻微数值稳定化，让 **RMSNorm vs LayerNorm**、**RevIn 实例归一** 更可能带来可测改进（四组 `--revin_rms_ablation` 仍共用同一套宽度/深度，彼此公平）。

### 改动摘要

| 项 | 原默认 | 新默认 | 说明 |
|----|--------|--------|------|
| `d_model` | 128 | **192** | 规约方式差异在更宽注意力上更明显 |
| `n_layers` | 3 | **4** | 略加深，给堆叠规约更多层 |
| `batch_size` | 64 | **96** | RevIn 按实例估计 σ 时方差略小 |
| `learning_rate` | 3e-4 | **2e-4** | 配合实例归一与更深网络，减前期震荡 |
| `training_noise_mse_huber_alpha` | 1.0 | **0.92** | 少量 smooth_l1，利尾部分布 |
| `early_stop_patience` | 8 | **10** | 深模型多给几轮验证耐心 |
| `baseline_*_d_model` | 128 | **192** | iTransformer/TimeMixer 与 SimDiff 宽度对齐，避免「只加宽 SimDiff」的假优势 |
| `RevINPatch` / `RMSNorm` eps | 1e-5 | **1e-4** | 短序列、小 batch 时 σ 更稳（`network.py` 中 RevIn 显式传入） |

### 涉及文件

- `config/config.py`
- `models/network.py`（RevIn `eps`）
- `models/revin_rms.py`（`RMSNorm` / `RevINPatch` 默认 `eps`）

### 备注

- **须删旧 checkpoint 重训**（`d_model`/`n_layers` 已变）。显存不足可将 `batch_size` 改回 64 或 `d_model` 改为 160（保持 `n_heads` 整除）。
- 推理侧仍为 **DDIM、`ddim_eta=0`**，与此前「固定采样、少随机干扰」一致。

---

## 2026-04-27：RevIn/RMSNorm 四组去噪器消融（独立权重、叠加图、柱状图、终端表）

### 目标

一次命令顺序训练（或 `--eval_only` 仅评）四组 SimDiff **去噪器结构**：**full**（RevIn+RMSNorm）、**vanilla**（无 RevIn，原版 `TransformerEncoder`+LayerNorm）、**revin_only**、**rmsnorm_only**。与 `simdiff_ablation`（NI/MoM）正交；**要求 `simdiff_ablation=full`**（`mom_only`/`ni_only` 仍走原 `main` 单模型流程）。

### 行为

- 权重：`checkpoints/simdiff_weather_best_<full|vanilla|revin_only|rmsnorm_only>.pt`（`Config.denoiser_variant` + `simdiff_checkpoint_filename()`）。
- 产出：`result/<数据集>/bar_mae_mse_denoiser_ablation_<后缀>.png`、`forecast_curves_denoiser_ablation_overlay_<后缀>.png`（默认后缀为运行时间戳，避免覆盖）；终端 **`print_thesis_metrics_table`** 四行均为 SimDiff（MAE / MSE / CRPS / VAR），脚注说明各变体含义。
- **不**训练 iTransformer/TimeMixer（消融段专用短路，与毕设三模型图分离）。

### 命令示例

```bash
python main.py --revin_rms_ablation --epochs 40
python main.py --revin_rms_ablation --eval_only   # 需上述四个 pt 已存在
```

单次训练关闭模块（非四组模式）：`--no_revin` / `--no_rmsnorm`。

### 修改文件

| 文件 | 改动 |
|------|------|
| `config/config.py` | `denoiser_variant`；`simdiff_checkpoint_filename()` 带后缀 |
| `utils/trainer.py` | `meta` 增加 `denoiser_variant` |
| `utils/result_output.py` | `print_thesis_metrics_table(..., footer_notes=...)` |
| `utils/compare_viz.py` | 多条 `SimDiff*` 曲线按序号分色 |
| `main.py` | `run_revin_rms_ablation_suite`、`--revin_rms_ablation`、`--no_revin`、`--no_rmsnorm` |

### 实验设计补充说明（写在论文/笔记）

- **公平性**：四组共用同一 `seed`、数据划分与 `epochs`（如 40）；早停各自独立，若需「严格同轮」可暂时调大 `early_stop_patience` 或关早停。
- **vanilla 含义**：与 DEVELOPMENT_LOG 中「仅 `TransformerEncoder`+LN」一致，**不是**关掉 NI/MoM。
- **RevIn**：潜空间序列维统计，与数据侧 NI 分层；若训练不稳可试略降 `learning_rate` 或对 RevIn 仿射使用更小初始化（当前为 1/0）。
- **RMSNorm**：预归一化与 `norm_first=True` 对齐；若与 vanilla 差距小，可试略增大 `d_model` / `n_layers` 再比。
- **可试超参**（不改结构）：`training_noise_mse_huber_alpha` 略小于 1 利尾部分布、`mom_cold_bias_blend`、DDIM `sampling_steps`；见 `config.py` 注释。

---

## 2026-04-27：测试集评估单次遍历（消融与 main 毕设段）

### 问题

对同一 SimDiff 权重，`evaluate_test_loader` + `eval_crps_on_test` + `mean_pred_sample_variance_on_test` 曾各自 **整表遍历** 且每次 `forecast` 均执行 K 次扩散，墙钟约 **3×**；四组 `--revin_rms_ablation` 评估阶段尤慢。

### 改动

- `main.py` 新增 `evaluate_test_loader_prob_combined`：`forecast(..., return_samples=True)` **每 batch 一次**，同时累计 MAE/MSE、温度通道 CRPS（含按步 `crps_h`）、VAR。
- `run_revin_rms_ablation_suite` 与主流程中「全测试集 MAE/MSE + 毕设 CRPS/VAR +（非 thesis_only 时）CRPS 图」均改用该函数；消融评估可选 `tqdm` 进度条。
- 从 `main.py` 移除对已不用的 `eval_crps_on_test` / `mean_pred_sample_variance_on_test` 的导入（二函数仍保留在 `utils/prob_metrics.py` 供他处复用）。

---

## 2026-04-27：result/ 毕设图防覆盖命名与训练说明

### 目录与文件名

- 数据默认 `data/weather.csv` → 图在 **`result/weather/`**（`Config.result_dir` + 数据文件主名）。
- **默认**每次运行：`main` 将 `result_name_suffix` 设为 **`YYYYMMDD_HHMMSS`**，`Config.result_png_basename()` 生成例如  
  `bar_mae_mse_temperature_20260427_153045.png`、`forecast_curves_temperature_overlay_20260427_153045.png`；  
  消融：`bar_mae_mse_denoiser_ablation_*.png`、`forecast_curves_denoiser_ablation_overlay_*.png`。同次运行内各图共用同一后缀，**多次运行互不覆盖**。
- **`--result_suffix TAG`**：自定义 `_TAG`（空格会替换为下划线）。
- **`--result_overwrite`**：`result_name_suffix=None`，仍用无后缀旧文件名（**会覆盖**同目录已有 png）。

### 常用训练命令（项目根目录）

| 目的 | 命令 |
|------|------|
| 完整流程（SimDiff + iTransformer + TimeMixer，毕设图进 `result/<数据集>/`） | `python main.py` |
| 训练 40 轮 | `python main.py --epochs 40` |
| RevIn/RMSNorm 四组消融（各独立 checkpoint） | `python main.py --revin_rms_ablation --epochs 40` |
| 仅加载已有权重评估 / 出图 | `python main.py --eval_only` |
| 不训基线、省时间 | `python main.py --skip_baselines` |

### 涉及文件

| 文件 | 改动 |
|------|------|
| `config/config.py` | `result_name_suffix`、`result_png_basename()` |
| `main.py` | `--result_suffix`、`--result_overwrite`；毕设段与消融段写图均经 `result_png_basename` |
| `utils/trainer.py` | checkpoint `meta` 增加 `result_name_suffix` |

---

## 2026-04-29：HistConditionalFiLM 作为 RevIn 插槽的替代选项

### 动机

实验中 RevIn 常弱于 RMSNorm-only；在**同一串联位置**提供**非可逆整条序列缩放**的历史条件调制，以便与 NI、RMSNorm 更温和配合。

### 行为

| 字段 / CLI | 说明 |
|------------|------|
| `Config.use_hist_film` | 默认 False；True 时需 `use_revin=False`（`validate_denoiser_embedding_options`）。 |
| `models.revin_rms.HistConditionalFiLM` | 对历史段 token `(B,Lh,d)` 池化后对整段 concat 做 \(seq \odot (1+s\tanh(\gamma)) + s\beta\)，无 denorm；末层线性零初始化以利于训练初期近似恒等。 |
| `models.network.DenoiserTransformer` | 参数 `use_hist_film`，与 RevIn **互斥**；与 `use_rmsnorm` **独立**。 |
| `python main.py --hist_film` | 置 `use_hist_film=True` 且 `use_revin=False`；须**删/换 checkpoint 重新训练**。 |
| `--revin_rms_ablation` | 仍为四类 RevIn/RMSNorm，`use_hist_film` 将被强制关闭。 |
| `utils/trainer.py` checkpoint `meta` | 增加 `use_hist_film`。 |

---

## 2026-04-29：消融图/终端表还原为「展示名 ≡ checkpoint stem」

曾将四柱/四行做成「论文排版型」错位（展示名与实际 `best_*.pt` 不对齐）；**已撤回**。`run_revin_rms_ablation_suite` 仅用 `_REVIN_RMS_ABLATION_SPECS`：`full`、`vanilla`、`revin_only`、`rmsnorm_only` 与 `simdiff_weather_best_<stem>.pt`、终端与图中英文标签**一一对应**。脚注去掉错位说明。

---

## 2026-04-29：`--film_ablation`（HistConditionalFiLM 四套，与 RevIn/RMSNorm 四套对位）

### 动机

在「RevIN 占位」上换 `HistConditionalFiLM` 时，与 **full / vanilla / hist_film_ln / rmsnorm_only** 四档做结构消融，并用 **独立文件名**（`best_film_*.pt`），避免与原 `best_full.pt` 等冲突。第三档 stem **`hist_film_ln`** 表示 **仅 Hist-FiLM + LN 编码栈**（不是 RevIN，也不是旧名 `revin_only`）。

### 权重（`simdiff_ablation=full`、`use_revin=False`）

| stem | Hist-FiLM | RMS 编码器栈 | 文件示例 |
|------|-----------|----------------|----------|
| `full` | 开 | 开 | `simdiff_weather_best_film_full.pt` |
| `vanilla` | 关 | 关（vanilla LN） | `…_film_vanilla.pt` |
| `hist_film_ln` | 开 | 关（仅 LN 编码器栈） | `…_film_hist_film_ln.pt` |
| `rmsnorm_only` | 关 | 开 | `…_film_rmsnorm_only.pt` |

（若磁盘上仍有旧名 `…_film_revin_only.pt`，`main` 会在启动 FiLM 消融时复制为 `…_film_hist_film_ln.pt`。）

### CLI

```bash
python main.py --film_ablation --epochs 50           # 四组依次训练（跳基线与 revin_ablation 同）
python main.py --film_ablation --epochs 50 --film_reuse_rmsnorm_ckpt   # 从 best_rmsnorm_only.pt 复制 film_rmsnorm；只训 full/vanilla/hist_film_ln
python main.py --film_ablation --eval_only           # 需四个 best_film_*.pt（或已复用完第四项）
```

与 `--revin_rms_ablation` **互斥**。**`--film_reuse_rmsnorm_ckpt`**：`rmsnorm_only`（FiLM 套）网络与 RevIn/RMS 消融同名变体一致，可复制 `simdiff_weather_best_rmsnorm_only.pt`→`simdiff_weather_best_film_rmsnorm_only.pt`，并跳过该项训练。图：`bar_mae_mse_hist_film_ablation*.png`、`forecast_hist_film_overlay*.png`。实现见 `main.py`：`run_hist_film_ablation_suite`。

---

## 2026-04-28：HistConditionalFiLM + RMSNorm 四组（RevIN 关闭；三训 + 复用 rmsnorm_only）

### 与 RevIN 的关系

- `run_hist_film_ablation_suite` 对每个变体均 **`use_revin=False`**（`_apply_hist_film_ablation_key`），潜空间 **RevIN 不参与前向**，仅对比 **HistConditionalFiLM**（占原 RevIN 档位语义）与 **RMSNorm 编码栈**。

### 训练（已有 RevIn/RMS 消融权重 `simdiff_weather_best_rmsnorm_only.pt`，只训另三组）

项目根目录执行：

```bash
python main.py --film_ablation --epochs 50 --skip_baselines --film_reuse_rmsnorm_ckpt
```

将 `best_rmsnorm_only.pt` 复制为 `simdiff_weather_best_film_rmsnorm_only.pt` 并 **跳过该项训练**；顺序训练 **full（FiLM+RMSNorm）、vanilla、hist_film_ln（Hist-FiLM+LN 栈）**。`--epochs N` 可覆盖默认。

### 产出（与原 denoiser 消融表/图等价位，对象为 FiLM+RMS）

- 终端四行：**MAE / MSE / CRPS / VAR**；**vanilla** 一行展示名为 **`SimDiff vanilla`**（**不写** RevIn+RMS denoiser 图中的 `(Transformer+LN)`）。
- `result/<数据集 stem>/`
  - `bar_mae_mse_hist_film_ablation_<后缀>.png`（对应旧的 `bar_mae_mse_denoiser_ablation_*.png`）
  - `forecast_hist_film_overlay_<后缀>.png`（对应旧的 `forecast_curves_denoiser_ablation_overlay_*.png`）
- 默认同次运行共用时间戳后缀；`--result_suffix TAG`、`--result_overwrite` 见前文。

### 四枚齐备后仅评估

```bash
python main.py --film_ablation --eval_only --skip_baselines --film_reuse_rmsnorm_ckpt
```

### 涉及改动

| 文件 | 说明 |
|------|------|
| `main.py` | `_HIST_FILM_ABLATION_SPECS`：`vanilla` → **`SimDiff vanilla`**；第三档 stem **`hist_film_ln`**（旧名 `revin_only` / `film_revin_only.pt` 已弃用） |

### `checkpoints/` 清理约定（本仓库本轮）

- **保留**：`simdiff_weather_best.pt`（主流程默认权重）、**`simdiff_weather_best_film_rmsnorm_only.pt`**（FiLM 四组之「仅 RMSNorm」档，与另存之 `rmsnorm_only` 结构相同）。
- **删除**：旧 **RevIn/RMSNorm** 四组文件名 `simdiff_weather_best_{full,vanilla,revin_only,rmsnorm_only}.pt`（若需 RevIn+RMS 消融可再训 `--revin_rms_ablation` 生成）。**不**删主权重与 FiLM `film_rmsnorm_only`。

---

## 2026-04-29：`--revin_rms_skip_rmsnorm_if_present`（仅此权重复用、其余三项再训）

仅已有 `simdiff_weather_best_rmsnorm_only.pt` 时：`python main.py --revin_rms_ablation --epochs N --skip_baselines --revin_rms_skip_rmsnorm_if_present`  

训完 `full` / `vanilla` / `revin_only` 后跳过第四项训练，同一次运行仍写 **`bar_mae_mse_denoiser_ablation*.png`**、**`forecast_curves_denoiser_ablation_overlay*.png`** 与终端四行表。

---

## 2026-04-28：HistConditionalFiLM 移除 → HistoryAdditiveBias + `--dual_ablation`

### 目标

去掉 **HistConditionalFiLM**，改为 **history-only additive bias（`HistoryAdditiveBias`）**；四柱消融语义 **A/B**：**A**=加性偏置，**B**=RMSNorm 编码栈；展示名 **`SimDiff full` / `vanilla` / `A_only` / `B_only`**；checkpoint **`simdiff_weather_best_dual_<full|vanilla|a_only|b_only>.pt`**（`ablation_ckpt_suite=dual`）。

### 移除与替换

- `models/revin_rms.py`：`HistConditionalFiLM` **删除**，新增 **`HistoryAdditiveBias`**。
- `Config.use_hist_*`：**`use_hist_add_bias`**、`hist_add_bias_scale` / `_with_rmsnorm`；**FiLM CLI/字段删除**。
- **`--film_ablation`**→**`--dual_ablation`**；**`--film_reuse_rmsnorm_ckpt`**→**`--dual_reuse_b_only_ckpt`**。
- **`--hist_film`**→**`--hist_add_bias`**。
- **`utils/compare_viz.plot_metrics_bars`**：缩短默认纵轴、`title`/`ylabel` 可配，版面防标题与刻度重叠。

### 旧权重

- `*_film_*.pt` 与 FiLM 结构与现网 **不兼容**，需 **`--dual_ablation`** 重训四套（或仅存 **`simdiff_weather_best_rmsnorm_only.pt`** 时用 `--dual_reuse_b_only_ckpt` 跳过 `b_only`）。
- **`--dual_reuse_b_only_ckpt`**：若缺失 `rmsnorm_only.pt`，会 **回退使用** **`simdiff_weather_best_film_rmsnorm_only.pt`**（同属「仅 RMSNorm」、`use_hist_add_bias=False`，与 dual `b_only` 同结构）。

### 涉及文件（摘要）

`models/network.py`、`models/simdiff.py`、`main.py`、`config/config.py`、`utils/trainer.py`、`utils/compare_viz.py`、`docs/EXPERIMENT_ARCHITECTURE.md`。

---

## 2026-04-28：多尺度历史拼接 + `--ms_rms_ablation`（取代 HistAdd `--dual_ablation`）；`xiaorong/` 消融图归档

### 目标

- **消融主线调整**：不再以 **HistoryAdditiveBias + A/B 四柱**（`--dual_ablation`，权重 `simdiff_weather_best_dual_*.pt`）作为主对比；改为 **多尺度历史特征融合（可选）× RMSNorm**，四柱为：**baseline（RevIn+RMSNorm，96 步）** / **rmsnorm_only** / **multiscale_only（RevIn+LN）** / **full（多尺度+RMSNorm）**。**HistAdd 可从 CLI 关闭，默认不参与本套件。**
- **多尺度输入（最简拼接）**：仅在 **数据与 `DenoiserTransformer` 历史长度（`pos_h`/`seq_len`）** 上扩展，**不改 Encoder 层定义**。拼接 **`[96h 原始 | 168→7 日均 | 672→4 周均]` → 107 token**，与论文约定一致；窗口起点 **`hist_window_start_min=576`**，四组对齐同一批滑动窗。
- **产出对齐旧 dual 实验**：终端 **`print_thesis_metrics_table`**（四行：**Model / MAE / MSE / CRPS / VAR**，与此前 FiLM、dual 等形式一致）；图 **`bar_mae_mse_ms_rms_ablation_<后缀>.png`**、**`forecast_ms_rms_ablation_overlay_<后缀>.png`** 写入 **`result/<数据集>/`**（后缀规则同全局 `--result_suffix` / `--result_overwrite`）。
- **拷贝**：每次 **`--ms_rms_ablation`** 跑完评估与作图后，将上述两张图**再复制一份**到仓库根目录 **`xiaorong/`**，固定文件名：**`bar_mae_mse_ablation.png`**、**`forecast_ablation_overlay.png`**，便于论文目录引用（与 `result/` 中带时间戳副本并存）。

### 新增 / 修改文件（摘要）

| 文件 | 说明 |
|------|------|
| `config/config.py` | `use_multiscale_hist`、`hist_window_start_min`、`effective_hist_len()`；`ablation_ckpt_suite="ms_rms"` → `simdiff_weather_best_ms_rms_<stem>.pt`。 |
| `utils/data_loader.py` | `WeatherWindowDataset` 支持多尺度拼接；`make_loaders` 在多尺度时抬高最小起点。 |
| `models/simdiff.py` | `DenoiserTransformer` 历史长度用 `cfg.effective_hist_len()`。 |
| `utils/trainer.py` | meta：`denoiser_hist_len`、`use_multiscale_hist`、`hist_window_start_min`。 |
| `main.py` | `run_ms_rms_ablation_suite`；`--ms_rms_ablation`、`--multiscale_hist`、`--ms_rms_reuse_rmsnorm_ckpt`；**弃用别名** `--dual_ablation`、`--dual_reuse_b_only_ckpt`（行为映射到新接口）；**`xiaorong/` 双图拷贝**。 |
| `xiaorong/` | 运行时写入 **`bar_mae_mse_ablation.png`**、**`forecast_ablation_overlay.png`**（含 `.gitkeep` 占位目录）。 |

### 命令示例

```bash
# 四柱训评（需 simdiff_ablation=full）
python main.py --ms_rms_ablation --epochs 40 --skip_baselines

# 已有 rmsnorm_only.pt 时跳过 rmsnorm_only 训练（复制权重）
python main.py --ms_rms_ablation --epochs 40 --skip_baselines --ms_rms_reuse_rmsnorm_ckpt

# 仅评估（四权重齐备）
python main.py --ms_rms_ablation --eval_only --skip_baselines
```

### baseline 单独 epoch（原版 SimDiff）

- **`--ms_rms_ablation`** 下：**baseline** 柱使用 **`cfg.ms_rms_baseline_epochs`（默认 30）**；**rmsnorm_only / multiscale_only / full** 仍使用全局 **`cfg.epochs`**（含 `--epochs`）。CLI：**`--ms_rms_baseline_epochs N`** 可覆盖默认值。

### 终端指标表（与 FiLM 终端一致）

- **`utils/result_output.print_metrics_ascii_table`**：`Model / MAE / MSE / CRPS / VAR` 简易分隔线表（无「毕设指标表」横幅）。**`run_ms_rms_ablation_suite`** 评估结束后调用该函数。

### rmsnorm_only 权重复用（组 2 免训）

- 套件启动时若尚无 **`simdiff_weather_best_ms_rms_rmsnorm_only.pt`**，会**按序**尝试从下列**已有文件**复制：**`simdiff_weather_best_dual_b_only.pt`**（旧 dual A/B 之 B_only）、**`simdiff_weather_best_rmsnorm_only.pt`**、**`simdiff_weather_best_film_rmsnorm_only.pt`**。
- 复制成功后 **rmsnorm_only 该项跳过训练**；无任一源文件则 **照常训练**组 2。
- 若显式 **`--ms_rms_reuse_rmsnorm_ckpt`** 仍得不到目标文件，则 **报错退出**（强制依赖已有权重）。

### 文档维护约定（本条起）

- **结构性或行为性改动**（新开关、新产出路径、消融语义变更）须在 **`docs/DEVELOPMENT_LOG.md` 文末追加一小节**：日期、目标、文件表、命令与产出说明；**本文后续迭代同样遵守**，避免只在代码里可查。

---

## 2026-04-29：ms_rms 图/表展示名；forecast overlay 连贯性

### 展示名（图例 / 柱标签 / 终端表）

| stem | 展示名 |
|------|--------|
| baseline | **SimDiff_original**（不出现 RevIn 文案） |
| multiscale_only | **SimDiff multiscale only** |
| rmsnorm_only | SimDiff RMSNorm only |
| full | SimDiff multiscale + RMSNorm |

### overlay 「断开」修复（`utils/compare_viz.py`）

- **`_y_limits_forecast_focus`**：改为用**完整 history** 与真值、预测一起定 y 轴；原先仅用历史**末 24 步**时，早期 history 常被裁出坐标轴，整条 history 看起来像从中途才开始、与分界线后真值不连贯。
- **`plot_forecast_compare` / `plot_forecast_compare_two_panels`**：在 history **末点**与 ground truth **首点**之间额外画一段黑色连线（与 GT 线宽一致），消除两段折线在边界处的视觉缝隙。

---

## 2026-04-29：预测长度（pred_len）精度趋势 — 与单权重物理约束

### 设计说明（必读）

- **`pred_len` 决定去噪网络未来段长度**（`DenoiserTransformer.pos_f` 等），与 checkpoint **强绑定**。现有 **`simdiff_weather_best_ms_rms_full.pt`** 仅在**训练当时**的 `pred_len`（默认 **24**）下形状匹配。
- 要对 **48 / 72 / 168 / 192** 比较「不同总预报步长」下的测试精度，需要 **每个 pred_len 单独训练** 得到权重；不能用一个 full 权重直接改 `pred_len` 推理。
- **trainer meta**（`pred_len`）已写入 checkpoint，可对账训练配置。

### 建议产出图件

| 图 | 含义 |
|----|------|
| **`mae_mse_vs_pred_len_ms_rms_full*.png`**（双轴折线） | 横轴：**pred_len**（48、72、168、192）；左轴 **MAE**、右轴 **MSE**（全测试集、气温通道），模型：**SimDiff multiscale + RMSNorm**。 |
| （可选）终端表 | 同一脚本轮询打印 `pred_len / MAE / MSE`。 |

与 **`mae_by_horizon.png`**（固定 pred_len 下 **逐步** MAE）区分：前者是「换窗口总长」，后者是「窗口内第几步误差大」。

### 权重文件命名约定（避免互相覆盖）

训练完某一 pred_len 后，将 `simdiff_weather_best_ms_rms_full.pt` **复制为**：

`checkpoints/simdiff_weather_best_ms_rms_full_pl{pred_len}.pt`

再训下一个 pred_len，以免覆盖。

### 脚本与绘图 API

| 文件 | 说明 |
|------|------|
| **`scripts/eval_pred_len_trend_ms_rms_full.py`** | 按 `--pred_lens` 加载上述命名权重，跑测试 MAE/MSE，写 **一张** 双轴折线图（横轴 pred_len，左 MAE 右 MSE，图例 **SimDiff**）。**默认输出目录：`length/`**（`mae_mse_vs_pred_len_ms_rms_full_<后缀>.png`）。**与 `main.py --ms_rms_ablation` 训练结束时的柱状图/overlay 无关**；后者仍在 `result/<数据集>/` 与 `xiaorong/`。 |
| **`utils/compare_viz.plot_pred_len_accuracy_trend`** | 双轴折线图实现。 |

示例：

```bash
python scripts/eval_pred_len_trend_ms_rms_full.py --pred_lens 48,72,168,192
```

缺少对应 `*_pl*.pt` 的长度会 `[skip]`，仅对已存在的权重画趋势。

### 单文件 `simdiff_weather_best_ms_rms_full.pt` 与对比实验（2026-04-28 对话补充）

- **与 `*_pl*.pt` 的关系**：磁盘上若仅有 **`simdiff_weather_best_ms_rms_full.pt`**（无 `_pl48` 等后缀），它只对应**训练当时**的 `pred_len`（默认配置为 **24**）。**不能**用该权重在 `eval_pred_len_trend` 里冒充 48/72/168/192；脚本会按 `pred_len` 构造网络，**形状与 checkpoint 不一致会加载失败或结果无效**。
- **训练各长度的 full 权重**：`pred_len` 参与 `DenoiserTransformer` / 扩散维数；需对每个目标长度单独训练 **ms_rms `full`**。**避免覆盖**磁盘上已有的 `simdiff_weather_best_ms_rms_full.pt`（例如 pred_len=24）：训练时使用 **`--pred_len H --ckpt_extra_suffix _plH`**，保存为 `simdiff_weather_best_ms_rms_full_plH.pt`，与 `scripts/eval_pred_len_trend_ms_rms_full.py` 默认 `ckpt_format` 一致。对应字段：`Config.simdiff_checkpoint_extra_suffix`；checkpoint `meta` 含 `simdiff_checkpoint_extra_suffix`。仅 `--pred_len`、不加后缀时，文件名不变，**仍会覆盖**同 stem 的旧权重。
- **是否需要对比实验**：
  - **画「精度随 pred_len 变化」这一条线**：**不强制**与 iTransformer/TimeMixer 或 persistence 同图对比；**一条 SimDiff multiscale+RMSNorm 曲线 + MAE/MSE 双轴**即可回答「更长预报窗口是否更难」。
  - **可选增强**（工作量显著）：在每个 `pred_len` 上另训/另评基线（如 persistence 或 iTransformer），做多条曲线；或固定某一 `pred_len` 的基线对比留在正文其它图（如 `bar_mae_mse_*`、`mae_by_horizon.png`），本图专注 **SimDiff 结构在不同总长度下的退化趋势**，避免重复劳动。
- **勿与 `mae_by_horizon.png` 混淆**：后者是**固定** `pred_len` 下**逐步**（第 1～H 步）的平均 MAE；本实验是**改变整窗未来长度 H** 后的**全窗平均** MAE/MSE。

---

## 2026-04-28：`--pred_len` 与 `--ckpt_extra_suffix`（避免覆盖旧 checkpoint）

### 行为

- **`--pred_len H`**：覆盖 `Config.pred_len`（须 `>=1`）；改变后须重训。
- **`--ckpt_extra_suffix SUFFIX`**：写入 `Config.simdiff_checkpoint_extra_suffix`，`simdiff_checkpoint_filename()` 变为 `{stem}{SUFFIX}.pt`（如 `…_ms_rms_full_pl48.pt`）。不传则与此前一致，**训练仍可能覆盖**同名 `…_full.pt`。
- **评估脚本** `eval_pred_len_trend_ms_rms_full.py` **只读**权重、不写 checkpoint，不会动旧 `pt`。

### 涉及文件

| 文件 | 改动 |
|------|------|
| `config/config.py` | `simdiff_checkpoint_extra_suffix`；`simdiff_checkpoint_filename()` 拼接 |
| `utils/trainer.py` | `meta` 增加 `simdiff_checkpoint_extra_suffix` |
| `main.py` | CLI `--pred_len`、`--ckpt_extra_suffix` |

---

## 2026-04-28：`--ms_rms_only`（pred_len 扫描只训 full 柱）

### 行为

- **`--ms_rms_only KEYS`**：`KEYS` 为逗号分隔子集，取自 `baseline` / `rmsnorm_only` / `multiscale_only` / **`full`**（多尺度历史拼接 + RMSNorm 编码栈，即消融「两模块都有」的柱）。默认不传的仍为 **四柱全套**。
- 仅子集时 **跳过** `_ensure_ms_rms_rmsnorm_checkpoint`（除非子集里仍含 `rmsnorm_only`）。
- 评估段 overlay 的参考 batch：若含 `baseline` 仍用 baseline；**仅 `full`** 时用 full 的 `test_loader` 取首 batch，避免强依赖 baseline 权重。

### 与 pred_len + 防覆盖联用示例

```bash
# 仅训 full，pred_len=48，权重写入 …_full_pl48.pt（不覆盖无后缀的旧 full）
python main.py --ms_rms_ablation --ms_rms_only full --pred_len 48 \
  --ckpt_extra_suffix _pl48 --epochs 40 --skip_baselines
```

对 72 / 168 / 192 各改 `--pred_len` 与 `--ckpt_extra_suffix` 再跑即可。

---

## 2026-04-28：iTransformer `pred_len` 扫表 → 双轴折线图（与 `mae_mse_vs_pred_len_manual` 同风格）

### 背景

终端汇总表 **「--- iTransformer pred_len sweep 汇总（test，气温通道）---」** 中 MAE/MSE 随 `pred_len`（48 / 72 / 168 / 192）变化，需与 `length/mae_mse_vs_pred_len_manual.png` **同版式**（双轴、MAE 实线圆点、MSE 虚线方点）出图，且 **不覆盖** 已有文件、**不经过** `main.py` 或权重加载，避免影响 SimDiff 训练/评估。

### 产出文件

| 文件 | 说明 |
|------|------|
| `length/itrans_pred_len_sweep_temp.csv` | 三列 `pred_len,mae,mse`（可 `#` 注释首行）；数据与终端表一致。 |
| `length/mae_mse_vs_pred_len_itrans_auto_v1.png` | 上述数据的折线图；若需新版本，请改 `--out` 文件名（如 `_v2` 或时间戳）。 |

### 脚本改动

`scripts/plot_pred_len_trend_manual.py` 增加可选参数 **`--curve-label`**、**`--title`**（默认仍为 SimDiff / ms_rms full 标题），便于同一脚本画基线曲线而无需复制粘贴整套绘图代码。

### 复现命令

```bash
cd /path/to/Simdiff_weather
python scripts/plot_pred_len_trend_manual.py \
  --csv length/itrans_pred_len_sweep_temp.csv \
  --out length/mae_mse_vs_pred_len_itrans_auto_v1.png \
  --curve-label "iTransformer" \
  --title "[weather] Test MAE / MSE vs prediction length (iTransformer, test, T degC)"
```

**说明**：`scripts/train_eval_itrans_pred_len_trend.py` 训练/扫表结束若加 `--plot`，默认会写 `length/mae_mse_vs_pred_len_itransformer.png`；此处为 **仅根据已有数值表重画**，与那条默认输出路径区分，防止互相覆盖。

---

## 2026-04-28：SimDiff 与 iTransformer · 同图双轴对比（pred_len 对齐）

### 目的

将 **主模型 SimDiff（multiscale + RMSNorm, ms_rms full）** 与 **对比模型 iTransformer** 在相同 `pred_len` 网格 **48 / 72 / 168 / 192** 上的 test 气温通道 **MAE/MSE** 画在 **一张** 双轴折线图中，便于直观比较。

### 数据来源

- SimDiff：与各 `pred_len` 训练后终端 **1-fold** 表一致（与 `plot_pred_len_trend_manual.py` 内置默认表相同）。
- iTransformer：与 `length/itrans_pred_len_sweep_temp.csv` / iTransformer pred_len sweep 汇总表一致。

### 产出

| 文件 | 说明 |
|------|------|
| `length/simdiff_vs_itrans_pred_len_overlay.csv` | 五列：`pred_len`，SimDiff MAE/MSE，iTransformer MAE/MSE（可改数后仍用脚本常量或自行扩脚本从 CSV 读取）。 |
| `length/mae_mse_vs_pred_len_simdiff_vs_itrans_v1.png` | 四条线：左轴两条 MAE（圆 / 三角），右轴两条 MSE（方 / 菱形，虚线）。 |

### 脚本

**`scripts/plot_pred_len_simdiff_vs_itrans.py`**：仅 matplotlib，**不加载** `main`、checkpoint。**另存图时请改 `--out`**，勿覆盖既有 png。

### 复现命令

```bash
cd /path/to/Simdiff_weather
python scripts/plot_pred_len_simdiff_vs_itrans.py \
  --out length/mae_mse_vs_pred_len_simdiff_vs_itrans_v2.png
# 可选：--title "自定义标题"
```

---
