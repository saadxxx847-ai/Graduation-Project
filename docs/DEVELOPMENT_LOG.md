# SimDiff-Weather 开发与改动记录

本文档随毕设/迭代持续更新：每次有结构性或行为性改动时，在文末 **追加** 新小节，并注明日期与相关文件。避免删除历史条目，便于对比实验与复现。

---

## 使用说明

- 一次改动对应一个小节，包含：目标摘要、动过的文件、配置项、与既有逻辑的关系、备注（权重兼容性等）。
- 与论文/开题报告对应的表述可引用本节中的「设计说明」。
- **每次迭代**（含自动化改动、助手/Cursor 批量改动）：凡涉及 **CLI、消融语义、权重命名、`result/`/`xiaorong/` 产出、`compare_viz`/毕设图语义** 等可追溯行为，须在文末 **追加** 小节记录；勿仅靠口头或零散注释。

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

## 2026-04-28：ETTh1 单变量 OT 预测、毕设图写入 `ETTh1/`、与 weather 权重的关系

### 数据

- 文件：`data/ETTh1.csv`；默认 **`temperature_only=True` 时只取 `OT` 列**（与 weather 只取 `T (degC)` 对称）。实现：`utils/data_loader.resolve_temperature_column_name` 在未匹配气温别名后回落到列名 **`OT`**；`main.resolve_temperature_feature_index` 在多变量表中对 `OT` 解析索引。
- **不要**在未改代码时用 `--all_features` 却仍想「只评 OT」——除非全开列并手动约定目标通道。

### CLI 扩展（`main.py`）

| 参数 | 作用 |
|------|------|
| **`--data_path`** | 指定相对项目根的 CSV（例 `data/ETTh1.csv`），换数据集等价于改 `Config.data_path`。 |
| **`--figures_dir`** | 毕设柱状图（`bar_mae_mse_*`）与 **forecast overlay**（`forecast_curves_*`）写入 **`<项目根>/<figures_dir>/`**；不传则仍为 **`result/<数据文件 stem>/`**（如 `result/ETTh1/`）。两者二选一即可；若希望和旧 `weather` 目录完全并排、又避免与 `result/ETTh1` 混名，可把图统一 export 到顶层的 **`ETTh1/`** 文件夹（见下文命令）。 |

### 与 iTransformer / TimeMixer 同图对比的训练命令

学习型基线在 **`use_multiscale_hist=False`** 时才会跑（history 长度为 `seq_len=96`，与基线架构一致）。**不要**同时使用 **`--multiscale_hist`** 或 **`--ms_rms_ablation`** 来期待同一条 main 里出现 iTransformer/TimeMixer——多尺度历史下 main 会跳过基线（见终端 `[note] use_multiscale_hist...`）；`ms_rms` 套件则是四柱 SimDiff 消融，不带 iTransformer/TimeMixer。

在一台机上从 **weather** 换到 **ETTh1** 时，**务必**指定 **checkpoint 后缀**，避免覆盖 **`simdiff_weather_best.pt`** 等文件名：

```bash
python main.py --data_path data/ETTh1.csv --figures_dir ETTh1 \
  --ckpt_extra_suffix _etth1_ot --epochs 50
```

完成后会得到：

- **`ETTh1/`**（或你选择的名字）下的 **`bar_mae_mse_temperature_<时间戳>.png`** 与 `forecast_curves_temperature_overlay_<时间戳>.png`；
- **终端**：`utils/result_output.print_thesis_metrics_table`，含 **SimDiff** 与 **iTransformer / TimeMixer** 的 **MAE、MSE、CRPS、VAR**（点预测两行：CRPS=MAE；VAR=0，与既有 weather 约定一致）。

仅评估（已训好 **`…_etth1_ot.pt`** 且基线在内存中会重训——见下）：

```bash
python main.py --data_path data/ETTh1.csv --figures_dir ETTh1 \
  --ckpt_extra_suffix _etth1_ot --eval_only
```

当前实现下 **`--skip_baselines` 未传**时，**iTransformer / TimeMixer** 在每次运行（含 `eval_only`）会从数据 **重新训练**到早停——没有单独保存基线权重；若需「只评 SimDiff、跳过基线」，加 **`--skip_baselines`**。

### 能否直接加载 **weather** 上训好的 SimDiff 权重？

**不建议作为最终报告结果，仅可作 smoke test。**

- 单通道时 **张量形状**与 ETTh1(OT) 一致，**可能能 load 进网络**；但 **RevIn 缓冲、未来边际统计、数据分布**均针对 weather，**指标与曲线通常不可信**。
- 正式对比：应在 **ETTh1 上重新训练**（或至少在该集上 **微调**），并使用 **`--ckpt_extra_suffix`** 保留 **weather** 与 **ETTh1** 两套 `pt`。

### 产出文件示例（文件名带默认时间戳后缀）

- `ETTh1/bar_mae_mse_temperature_<suffix>.png` — SimDiff vs 基线 MAE/MSE 柱图。
- `ETTh1/forecast_curves_temperature_overlay_<suffix>.png` — 真值 + 多模型预测折线叠加（批量 0）。

---

## 2026-04-29：`Config` 默认 SimDiff：**RevIn 关、RMSNorm 开、多尺度历史开**

- **`config/config.py`**：`use_revin=False`，`use_rmsnorm=True`，`use_multiscale_hist=True`。
- **`main.py`**：新增 **`--revin`**（临时开 RevIn）；**`--single_scale_hist`** 关闭多尺度（恢复 96 步历史，可与 iTransformer/TimeMixer 同跑学习型基线）。**`--multiscale_hist`** 仍可显式开多尺度（默认已与 Config 一致）。**`--no_revin`** 显式关闭 RevIn（若与 **`--revin`** 同时传，以 **`--no_revin`** 为准）。
- **`_clear_ms_rms_key` / RevIn·RMS 消融段结束时的 cfg** 与新版默认对齐（RevIn 关、RMSNorm 开、恢复多尺度开）。
- **注意（已部分替代，见下一条）**：若需 **SimDiff 与基线同一历史长度（均 96 步）** 的严格对照，仍可用 **`--single_scale_hist`**；多尺度全开时请看「多尺度 SimDiff + 基线」一条。

---

## 2026-04-29：`plot_forecast_compare` 横轴与多尺度 history（future 起点 bug）

- **现象**：多尺度时 `hist` 长度为 `seq_len+11`，但若 `t_fut` 仍从 `seq_len` 起算，真值会与 history 在 **x=seq_len … ehl-1** 上重叠，且 `compare_viz` 中 **history 末点 ↔ 真值首点** 连接线会在 x 轴 **折返**（看起来像 ground truth「扭到虚线左边」）。
- **修复**：`main.forecast_overlay_time_axes(cfg)` 统一为 `t_hist = 0..ehl-1`、`t_fut = ehl .. ehl+Lf-1`，`ehl = effective_hist_len()`；毕设 overlay、`run_revin_rms_ablation_suite` 叠图、以及 `forecast_example.png` 分界线 **（`axvline(ehl-0.5)`）** 一并改正。

## 2026-04-29：多尺度 SimDiff 与同 run 训练 iTransformer / TimeMixer

- **做法**：数据层多尺度历史中，前 `seq_len` 步为原始分辨率段（与旧时单尺度一致）；基线仅消费 **`hist[:, :seq_len, :]`**，SimDiff 仍消费全长 `seq_len+11`。**`utils/baselines.BaselineHistTrim`** + **`main.py`** 中学习型基线在 `use_multiscale_hist=True` 时自动外包一层。
- **含义**：两条基线在信息上 **弱于** 全量多尺度 SimDiff（公平性说明写进 `[note]` 终端提示）；若希望基线也得到「整段上下文」量级，需在架构上重做基线或使用 **`--single_scale_hist`** 做「人人 96 步」的另一套对照。

---

## 2026-04-29：多尺度历史 NI — `hist_stats_span`（仅用前 `seq_len` 步估 μ_h, σ_h）

### 现象与根因

- 多尺度下 `hist` 为 `[seq_len 细粒度 | 7 日均价 | 4 周均价]`（Lh = seq_len+11）。此前 `normalize_history` 在 **整段 Lh** 上算 mean/std，池化段的尺度与分布与小时段不同，**污染** μ_h、σ_h，使条件化输入与训练目标空间错位；表现为多条模型曲线相对真值 **整体平移**、趋势拟合差等（与仅关多尺度后误差减小的现象一致）。
- **不是** `inverse_transform_future` 公式错误：实现为 \(Y = z\sigma_f + \mu_f\)，与 `normalize_future` 可逆。

### 改动

| 文件 | 说明 |
|------|------|
| `utils/independent_normalizer.py` | `normalize_history(..., hist_stats_span=None)`：默认仍用整段 Lh（兼容旧调用）；若传 `hist_stats_span=seq_len`，则 μ_h、σ_h 仅来自 `hist[:, :seq_len]`，再对 **全长** hist 做仿射。 |
| `models/simdiff.py` | `training_loss` / `forecast` / `get_denoise_trajectory_physical` 均传入 `hist_stats_span=int(cfg.seq_len)`。 |
| `verify_norm_mom.py` | 与主流程一致，对 batch 使用 `hist_stats_span=cfg.seq_len`。 |
| `utils/normalizer.py` | `normalize_pair` 增加可选 `hist_stats_span` 并下传。 |

### 权重兼容

- **多尺度 SimDiff 既有 checkpoint** 在旧统计量分布上训练，与本 fix 后的输入分布不一致，需 **重新训练** 后再报告指标与曲线。

---

## 2026-04-29：`Config.learning_rate` 默认 `2e-4` → `3e-4`

- **`config/config.py`**：`learning_rate` 默认值改为 **`3e-4`**（与此前文档中「主训练默认 lr」表述对齐；无 CLI 覆盖时生效）。

---

## 2026-04-29：Ground truth / overlay 核查与绘图修正（非 NI 训练语义错误）

### 关于「未来用自身 μ/σ、推理用历史 μ/σ」类说法

- **与本仓库不符**：SimDiff 在 `forecast(..., future=fut)` 下用 **`normalize_future(future)`** 的 μ_f、σ_f 做反变换（见 `models/simdiff.py` `_future_mu_sig_for_inverse`）；训练目标也在 **未来窗口独立** z-score 空间，与有真值评估时一致。iTransformer / TimeMixer 在 **原始尺度** 上拟合，不存在同一类 denorm 混用。
- **数据加载** `WeatherWindowDataset.__getitem__` 返回的 `fut` 与 `hist` 在时间上连续：`fut = data[i+seq_len : i+seq_len+pred_len]`，**ground truth 未画错**。

### 实际 bug（多尺度 + 纯可视化 / 辅助图）

1. **Overlay 边界锚定与 GT 连线**：`hist[-1]` 在多尺度下是 **周池化尾点**，不是「预报起点前一小时」；`anchor_forecast_boundary` 与 history→GT 连线若用 `-1`，会与不同时刻的真值首步错位。**修复**：`plot_forecast_compare*` 增加 **`hist_anchor_index`**；`main.thesis_overlay_hist_anchor_index(cfg)` 在多尺度时取 **`seq_len-1`**。
2. **`plot_forecast_predictive_intervals`**：`t_fut` 曾误用 **`seq_len` 起点**，与 `t_hist=0..ehl-1` 在 x 轴上 **重叠 11 步**。**修复**：`t_fut = arange(ehl, ehl+pred_len)`。
3. **`plot_forecast_grid`**：曾固定 `seq_len` 为横轴长度，与 `Lh>seq_len` 的 multiscale **不匹配**；主流程误传 **`_ehl`（未定义）**。**修复**：按每条样本 `h.shape[0]` 生成 `t_hist/t_fut`，签名改为 **`(examples, pred_len, ...)`**。

### 涉及文件

| 文件 | 说明 |
|------|------|
| `utils/compare_viz.py` | `hist_anchor_index`；`plot_forecast_grid` 按 Lh 定轴 |
| `main.py` | `thesis_overlay_hist_anchor_index`；毕设 overlay / 预测区间 / grid 调用 |

---

## 2026-04-29：彻底杜绝 overlay 上 ground truth 画进 history x 区间

### 根因

- 多尺度时 ``Lh = effective_hist_len() > seq_len``。若调用方仍传入 ``t_fut = seq_len .. seq_len+Lf-1``，而 ``t_hist = 0 .. Lh-1``，则 **GT 与 history 在 x∈[seq_len, Lh-1] 上重叠**，表现为「真值折线跑回 history 时段」。此错误曾由 **main 与 ``plot_forecast_compare`` 的拆参组合** 反复引入。
- **治本**：在 **`plot_forecast_compare`**、**`plot_forecast_predictive_intervals`**、**`plot_forecast_compare_two_panels`** 内部，**仅根据** ``hist.shape[0]`` 与 ``true_fut.shape[0]`` 生成 ``t_hist`` / ``t_fut``，**不再**接受外部传入的时间轴；并对 ``preds`` 与 ``Lf`` 做一致性检查。
- **`main.py`**：移除仅服务于上表除「手写 forecast_example」外对 ``forecast_overlay_time_axes`` 的依赖；**`forecast_example.png`** 亦改为按 ``hist0/true0`` 长度本地生成横轴。
- **`forecast_overlay_time_axes(cfg)`** 仍保留，供文档/外部脚本与手工对齐语义（与现绘图推导一致：``t_fut[0]=Lh``）。

---

## 2026-04-29：overlay「真值画到虚线前」实为边界 **连接线** 误导

### 现象

- 多尺度下 `hist_anchor_index=seq_len-1` 时，曾在 **细粒度末步** 与未来首点之间画 **粗黑实线**，与 **ground truth** 线同色同宽；该线段横跨 x=seq_len..Lh-1（池化段），**几乎全部落在竖虚线（Lh-0.5）左侧**，易被误认为「GT 进 history」。
- **真实** GT 折线仅绘制在 `t_fut = Lh ..`；无需因本项重训权重。

### 修复

- `plot_forecast_compare` / `plot_forecast_compare_two_panels`：边界连接改为 **仅** `(Lh-1, hist[-1]) → (Lh, true_fut[0])`；样式改为 **灰虚线、较细**，与黑色实线 GT 区分。

---

## 2026-04-29：预报起点装饰（底色/圆点）已撤回

- **曾**在 overlay 上加 ``[Lh-1,Lh]`` 浅底色与末/首点 scatter；用户反馈后 **已恢复**为仅 **``ax.axvline(..., gray : )``** 的简单分界（与此前习惯一致）。

---

## 2026-04-29：ETTm1（15min）单变量 OT 预测、毕设图写入 `ETTm1/`

### 数据与 CLI

- 文件：`data/ETTm1.csv`；默认 **`temperature_only=True`** 时只取 **`OT`** 列（与 ETTh1 / weather 单变量流程一致）。
- **训练 + 对比基线 + 毕设图 + 终端指标表**（无代码改动，沿用 `main.py`）：

```bash
python main.py --data_path data/ETTm1.csv --figures_dir ETTm1 \
  --ckpt_extra_suffix _ettm1_ot
```

- 权重：`checkpoints/simdiff_weather_best_ettm1_ot.pt`（勿与其他数据集后缀混用以免覆盖）。
- 仅评估（已有权重后）：同上命令加 **`--eval_only`**（基线仍会当场重训至早停）。

### 本轮产出示例（文件名含时间戳后缀）

| 路径 | 说明 |
|------|------|
| `ETTm1/bar_mae_mse_temperature_<suffix>.png` | SimDiff vs iTransformer vs TimeMixer 的 MAE/MSE 柱状图 |
| `ETTm1/forecast_curves_temperature_overlay_<suffix>.png` | 真值 + 三模型预测折线叠加（test batch 0） |
| 终端 `print_thesis_metrics_table` | OT 通道：**MAE / MSE / CRPS / VAR**（点预测两行：CRPS=MAE，VAR=0） |

### 备注

- ETTm1 为 **15 分钟**采样；多尺度日历语义已与 ETTh 对齐方式见文末 **「ETTm 多尺度池化日历对齐」**（`multiscale_steps_per_hour=4`，勿再按旧版 168/672 解释为「与 ETTh 相同日历窗」）。
- 本次完整跑（早停约 epoch 43）：示例数值（仅作存档）：SimDiff MAE≈0.393、MSE≈0.389、CRPS≈0.276、VAR≈0.314；基线 MAE 高于 SimDiff。

---

## 2026-04-29：毕设 overlay 竖直锚点改为 history **末格**（`-1`）

### 现象

- 多尺度下 history 长度为 `Lh=seq_len+11`，绘图锚点若取 **`seq_len-1`**（细粒度末步），`_anchor_preds_to_hist_end` 会把三条预测曲线的第一步对齐到 **`hist[seq_len-1]`**。
- 折线 **`history`** 仍画满 `0..Lh-1`，终点为 **`hist[Lh-1]`**（池化尾），与锚点取值不同 → 分界处三条模型曲线与 history **看似不在同一衔接高度**。

### 改动

- **`main.py`**：`thesis_overlay_hist_anchor_index` 改为恒返回 **`-1`**（末 conditioning token），使展示用平移后的预测第一步与 **图中 history 终点**同 y。
- **不影响**：终端 MAE/MSE/CRPS（仍为未平移预测）；ground truth 仍为真实 `true_fut`。竖直对齐与「折线在虚线处是否一笔连成」属不同层面，后者见下文 **overlay 边界衔接绘制**。

---

## 2026-04-29：ETTm 多尺度池化日历对齐（fixed 168/672 按小时硬编码）

### 问题

- `_concat_multiscale_history` 中日窗 **168**、周窗 **672** 对应 **每小时一步**（ETTh）：168 步 = 7 天、672 步 = 28 天。
- **ETTm（15min）** 若仍用 168/672，则仅约 **42 小时 / 7 天**，与「日/周尺度」设计不符。

### 改动

| 文件 | 说明 |
|------|------|
| `config/config.py` | `multiscale_steps_per_hour: Optional[int]`（`None`=自动）。 |
| `utils/data_loader.py` | `resolve_multiscale_steps_per_hour`（文件名含 `ettm`→4，否则→1）；`multiscale_window_start_min(seq_len,sph)`；`_concat_multiscale_history(..., steps_per_hour)` 中日窗 `7×24×sph`、周窗 `28×24×sph`。 |
| `utils/trainer.py` | checkpoint `meta` 增加 `multiscale_steps_per_hour`。 |
| `main.py` | `--multiscale_steps_per_hour`；打印 `multiscale_steps_per_hour` 与 `hist_window_start_min`。 |

### 兼容性

- **ETTh / weather**：`steps_per_hour=1`，行为与旧版一致（`wmin=576`）。
- **ETTm**：`steps_per_hour=4`，`wmin=2592`；**须在 ETTm 上删除旧 checkpoint 并重训**，旧权重输入分布已变。

### 关于「数据泄漏 / 索引错误」（评审常见质疑）

- `WeatherWindowDataset.__getitem__`：`fut = data[i+seq_len : ...]`，`hist`（含池化段）仅用 **索引 `< i+seq_len`** 的历史片段；未见未来标签泄入 conditioning。
- 「三模型均值回归」多为任务/容量/训练现象；池化对齐后可再评估，非单靠改索引可证伪。

---

## 2026-04-29：`compare_viz` overlay 折线在竖虚线处「断开」— 边界衔接绘制

### 现象

- `history` 与「GT / 各模型未来」原为 **两次独立** ``ax.plot``（``x`` 分别为 ``0..Lh-1`` 与 ``Lh..``），Matplotlib **不会**自动绘制 ``(Lh-1)→(Lh)`` 的连线，竖虚线两侧看起来像「断了」。
- 原先仅用 **灰色虚线** 连接 ``hist[Lh-1]`` → ``true_fut[0]``，**未**与各彩色预测线衔接。

### 改动

| 文件 | 说明 |
|------|------|
| `utils/compare_viz.py` | 新增 ``_t_fut_with_bridge``；``plot_forecast_compare`` / ``plot_forecast_compare_two_panels`` 对 GT 与每条预测在 ``x`` 轴插入 ``Lh-1``，``y`` 首点为 ``hist[-1,c]``，与未来段连成 **一条折线**；去掉单独的灰虚线段（黑色 GT 粗线已覆盖 ``Lh-1→Lh`` 的首段）。 |

### 语义

- **仅绘图**：不改变磁盘评估指标；锚平移逻辑仍为 `_anchor_preds_to_hist_end`。
- **竖虚线**仍位于 ``Lh-0.5``，用于分隔 conditioning 与未来时段。

---

## 2026-04-29：核查 `bug/3.txt`「NI 泄漏」论断、单尺度 overlay 观感、SimDiff 验证指标可选

### 1. 对外部「用 hist μ/σ 反变换未来」建议的结论

- **`bug/3.txt`** 主张将 `_future_mu_sig_for_inverse` 改为仅用 history 估计 μ、σ，并声称当前「训练推理归一化不一致」。
- **本仓库结论**：该改法 **不适用**。SimDiff 采用 **Normalization Independence**（见 `utils/independent_normalizer.py`）：训练目标在 **`normalize_future(future)`** 空间；评估 **有真值** 时用 **同一 batch 的 μ_f、σ_f** 反变换，与 `training_loss` **一致**；**无真值** 时用 **`make_loaders` 写入的训练集未来边际** μ、σ。
- **勿**在未同步改写 `training_loss` / `normalize_future` 的前提下改用 hist 统计量反变换未来，否则会破坏 NI 语义。
- **`models/simdiff.py`** 模块注释已简短重申上述边界，避免后续误改。

### 2. 为何 `--single_scale_hist` 下图「更像从同一点出发」

- 单尺度时 **`Lh=seq_len`**，灰色 history 与 `BaselineHistTrim` 消费的序列长度一致；多尺度时 **`Lh=seq_len+11`**，末端为池化 token，曾与可视化锚点产生观感差异（参见前文 overlay 锚点与边界衔接条目）。
- 叠加 **`compare_viz`** 边界衔接绘制后，两类设定均应从图上连贯延伸；若仍觉得多尺度难解释，可优先用单尺度做对照实验。

### 3. SimDiff 在单尺度 ETTm 上弱于 iTransformer / TimeMixer — 现象说明

- **现象**：直接回归类基线在 **原始尺度** 上优化；SimDiff **早停 / LR 调度**依据 **`Trainer.validate()` 的扩散噪声训练损失**（`training_loss`），与终端报告的 **原始尺度 MAE** 不完全同一目标，可能出现「噪声损失好但曲线一般」的 checkpoint。
- **曾尝试（已撤回）**：验证阶段改用 **MoM 预报 MAE**（`forecast_mae`）做早停，与 overlay 更对齐，但 **每 epoch 需完整扩散采样**，训练 **过慢**，已 **从代码中移除**；当前 **`validate()` 仅保留噪声损失**（与最初行为一致）。
- 若需原始尺度意义上的「好权重」，可调 `epochs` / `learning_rate` / 模型宽度，或以 **`--eval_only` 在若干 checkpoint 上扫测试 MAE**（不塞进每个 epoch 的 validate）。

---

## 2026-04-30：`data/wind.csv` 上 OT 预测、15min 多尺度、`result/` 文件名与英文图表

### 数据

- **`data/wind.csv`**：含 **`OT`** 列；时间步为 **15 分钟**。多尺度池化与 ETTm 相同，须要 **`steps_per_hour=4`**。

### 代码

| 文件 | 说明 |
|------|------|
| `utils/data_loader.py` | `resolve_multiscale_steps_per_hour`：`wind` stem 返回 **4**；`resolve_temperature_column_name` 已由 **OT** 分支选中目标列。 |
| `main.py` | **`wind.csv`** 且未指定 **`--ckpt_extra_suffix`** 时自动 **`_wind`** → **`simdiff_weather_best_wind.pt`**，避免与 **`weather`** 共用默认 checkpoint。CLI 显式后缀优先。 |
| `main.py` | 毕设图逻辑名：**`bar_mae_mse_comparison`**、**`forecast_curves_overlay`**（接替原 `*_temperature*`）；图题与柱状图 **`ylabel` 英文**；`print_thesis_metrics_table(..., english=True)`。 |

### 训练与评估（摘录）

```bash
cd Simdiff_weather
python main.py --data_path data/wind.csv
python main.py --data_path data/wind.csv --eval_only
```

- 不加 **`--all_features`**（保持仅 **OT** 单变量）。
- 产出在项目根 **`wind/`**（`--data_path data/wind.csv` 时 **`Config.resolved_result_dir()`** → `Simdiff_weather/wind/`，与 `result/<其它数据集>/` 区分）；终端指标表：**MAE / MSE / CRPS / VAR**；仍可用 **`--figures_dir <子路径>`** 覆盖。
- **展示名称**：对内仍为多尺度历史 + RMSNorm 等栈；图上与表中统一称 **SimDiff**。

### 兼容性

- 历史文档里的 **`bar_mae_mse_temperature*.png`** 等与当前 **`bar_mae_mse_comparison*.png`**、**`forecast_curves_overlay*.png`** 为同一流水线不同文件名。
- **目录约定（更新）**：`data_path` stem 为 **`wind`** 时，毕设图默认目录为 **项目根下 `wind/`**（不再是 `result/wind/`）；其它数据集仍为 **`result/<stem>/`**。需要自定义位置时用 **`--figures_dir`**。

## 2026-04-30：`bug/3.txt`「hist 归一 future」vs 本文实现；**train z-score 文献指标**（不拖慢训练）

### 结论（与 `models/simdiff.py` / NI 文档一致）

- **未采纳** `bug/3.txt` 将 **`normalize_future`** 改为完全用 **hist 的 μ/σ** 的做法：那会 **改变 SimDiff NI 的训练目标与论文式表述**，与本仓库 **窗口级未来独立归一化** 不一致；与 `DEVELOPMENT_LOG` 中 NI 条目结论相同。
- **训练速度**：不重写 `training_loss`，不增加每 epoch 全流程采样早停；**零额外前向**——z 指标在已有 `evaluate_test_loader_prob_combined` 循环内用 **`/σ_train`** 与张量变换完成。

### 实现

| 文件 | 说明 |
|------|------|
| `utils/data_loader.py` | 仅用 **训练划分** 整条时间序列估计每通道 **μ、σ**（带 floor），写入 **`cfg.train_metric_z_mu/sigma`**（**不含 val/test**）。 |
| `main.py` | `evaluate_test_loader_prob_combined` 返回 **`mae_ch_z`、…、`crps_mean_z`**；**毕设横幅表** **`print_thesis_metrics_table`** 与 **`bar_mae_mse_comparison_*`** 以 **MAE_z/MSE_z** 为主；物理尺度副表 **`print_metrics_ascii_table`**、副柱图 **`bar_mae_mse_physical_*`**。**（本条历史）**早期实现曾以物理为主表、`bar_train_z_*` 为副图，见文末 **2026-04-30（续）**。 |
| `utils/baselines.py` | **`eval_channel_mse_mae_train_zscore`**（点基线同 σ）。 |
| `config/config.py`、`utils/trainer.py` | 配置字段；checkpoint **meta** 可选记录 z 统计量。 |

### 指标定义

- **`MSE_z`、`MAE_z`**：**\((\hat y - y)^2 / \sigma_{\text{train}}^2\)**、**\(\|\hat y - y\| / \sigma_{\text{train}}\)** 在测试集平均；与「两列先减再除 σ」在差分上等价于 **μ_train 抵消**。  
- **SimDiff 的 CRPS/VAR（z）**：在 **primary channel** 对样本与观测做 **\((\cdot - \mu_{\text{train}}) / \sigma_{\text{train}}\)** 后重算（与物理尺度 CRPS 并行报告）。

---

## 2026-04-30：Wind 陡变 / NI 与训练目标错位——x0 辅助损失、可选 μ/σ 混合反变换、稀疏 forecast MAE 选优

### 背景

- `debug/1.txt`：`wind` 单变量上 overlay 易出现近似均值回归、难以跟随真值陡降；根因含「归一化空间太平滑」与「早停指标与物理 MAE 不一致」等。
- 此前日志曾写明「全量每 epoch 用 forecast MAE 早停过慢故移除」；本轮改为**可选项**：仅每 **N** 个 epoch 且在验证集**前 B 个 batch** 上算 forecast MAE 存 best，控制 wall time。

### 代码

| 文件 | 说明 |
|------|------|
| `config/config.py` | `training_noise_x0_aux_weight`（默认 0）；`ni_inverse_hist_frac`（默认 0）。校验：`validate_ni_inverse_options`。 |
| `models/diffusion.py` | `training_losses(..., x0_aux_weight)`：在已有 ε 预测上追加 **λ·MSE(x̂0, x0)**，无额外前向。 |
| `models/simdiff.py` | `training_loss` 传入 x0 权重；`_future_mu_sig_for_inverse(hist, future, ...)` 支持 **(1-w)·未来统计 + w·历史统计**（`mom_only` 仍仅用 μ_h,σ_h）。 |
| `utils/trainer.py` | **已移除** `val_forecast_mae_every` 稀疏 forecast 验证与 **`ReduceLROnPlateau`**；早停仍按 **验证噪声 loss**，**学习率全程为 `Config.learning_rate`**。 |
| `main.py` | CLI：`--x0_aux_weight`、`--ni_inverse_hist_frac`（**无** `--val_forecast_mae_every` / `--val_forecast_max_batches`）。 |

### 论文 / 指标一致性

- **正式对比 SimDiff（NI）**：评测须 **`--ni_inverse_hist_frac 0`**（默认即 0）。`w>0` 仅作工程试探或开环观感，**不是**严格 NI。
- **`x0_aux_weight`**：仍在 **normalize_future** 空间拟合 x0，与 NI 训练语义一致。

### Wind 推荐训练命令（在可接受耗时下尽量对齐曲线）

```bash
cd /path/to/Simdiff_weather
# 删除旧 best 后再训，避免加载过时权重
rm -f checkpoints/simdiff_weather_best_wind.pt

python main.py --data_path data/wind.csv \
  --x0_aux_weight 0.12 \
  --epochs 50
```

- 若曲线仍偏平滑：可试略增 **`--training_noise_temporal_diff_weight`**（需在代码里暂改 Config 或后续加 CLI）、或 **`--ddim_eta 0.15`**（推理随机性，不重训也可试）。
- 仅刷新 overlay、**不重训**且愿牺牲 NI 可解释性：可用 **`--ni_inverse_hist_frac 0.2`** 与 `--eval_only` 看形态（**勿**与论文表 MAE 混报）。
- 产出目录：默认 **`wind/`**（或 `--figures_dir result/wind` 自定）。

### 评估出图

```bash
python main.py --data_path data/wind.csv --eval_only --ni_inverse_hist_frac 0
```

---

### （更新）已撤回的慢速验证

- **`validate_forecast_mae` / `val_forecast_mae_every`**：因训练过慢已从 **`utils/trainer.py`** 与 **Config/CLI** 删除；checkpoint 仍按 **验证集噪声损失** 选优。
- **学习率**：SimDiff 训练 **不再使用 `ReduceLROnPlateau`**，全程 **`learning_rate`（默认 3e-4）** 静态，除非你在代码里改 `AdamW` 初始 lr。

---

## 2026-04-30：毕设 `forecast_curves_overlay` 可选测试 batch（`--thesis_overlay_test_batch`）

- **动机**：overlay 原先固定 **test_loader 第 0 个 batch**；全集 MAE 变好时该窗可能仍「看起来像旧图」。
- **用法**：`--thesis_overlay_test_batch J`（`J` 从 0 起）。**只改** overlay 画的窗口，**不改**表内全测试集指标；`J` 超范围时告警并退回 0。
- **实现**：`main.thesis_overlay_fetch_batch`；`Config.thesis_overlay_test_batch`。

---

## 2026-04-30：多尺度 vs 单尺度——`train_loss` 起始数值为何从几百 vs 约 0.4（问与答）

### 现象

- **多尺度历史**（默认 `use_multiscale_hist=True`）训练初期：终端 `train_loss`（及 `val_mse` / `val_noise`）常出现 **几十～数百或更高**，随后若干 epoch 内快速下降。
- **单尺度历史**（`--single_scale_hist`，如 `checkpoints/simdiff_weather_best_wind_noms.pt` 一类实验）：第一轮 `train_loss` 即可在 **约 0.3～1.x** 量级。

### 解答（非两套损失、非 bug）

1. **`train_loss` 定义相同**：均为 `models/simdiff.py` 的 `training_loss` → 在 **`normalize_future(future)`** 得到的 **`fut_n`（NI 未来 z-score 空间）** 上做扩散噪声目标（`GaussianDiffusion.training_losses`：MSE/Huber、可选 L1、时间差分、可选 x0 辅助）；**不是** CSV 物理风速尺度。
2. **单尺度**：`hist` 仅 **seq_len** 步、分辨率一致；`normalize_history(..., hist_stats_span=seq_len)` 与整段历史语义一致，**条件 `hist_n` 更齐**，冷启动时 **ε̂** 误差往往就是 **O(1)** 量级 → **loss 一上来就小** 很正常。
3. **多尺度**：`hist` 为 **细粒度段 + 日/周池化尾段**；μ_h、σ_h 仍只由 **前 seq_len 步** 估计，再仿射到 **含池化 token 的整段**。尾段与参照段尺度混杂，**条件分布更难**，随机初始化下 **ε 预测可差很多** → **初始 `train_loss` 可大若干数量级**，随后优化变顺，与日志一致。
4. **与旧日志中 `val_forecast_mae`、学习率调度**：仅影响选 checkpoint / lr，**不改变 `train_loss` 定义**；当前仓库已改为 **静态 lr** 且已移除稀疏 forecast 验证（见上节）。

### 使用建议

- **不要**用首轮 `train_loss` 绝对值横向对比「多尺度 run vs 单尺度 run」判断谁更对；应看 **收敛趋势、验证/测试噪声 loss 与最终 MAE/MSE**。

---

## 2026-04-30：关多尺度后 overlay 上 **ground truth 也变了**？（问与答）

### 问题

两张国别为「多尺度 / 单尺度」的 `forecast_curves_overlay`，均称 **test batch 0**，但 **黑色真值折线不同**——真值不应该是固定的吗？

### 解答

- **「batch 0」固定的是 DataLoader 里第 0 个 batch**，不是「固定在同一日历时刻」。
- **`utils/data_loader.make_loaders`**：仅当 **`use_multiscale_hist=True`** 时会把 **`hist_window_start_min`** 抬到 `multiscale_window_start_min(seq_len, steps_per_hour)`（Wind 15min 下为 **2592**，见 `multiscale_steps_per_hour`）。
- **`WeatherWindowDataset.__getitem__`**：`i = idx + window_start_min`。  
  - **多尺度**：`idx=0` → `i=2592`（在 **test 段**矩阵内），未来窗 = `data[2592+seq_len : …]`。  
  - **单尺度**：`window_start_min` 常为 **0**（未进入多尺度分支则不改写），`idx=0` → `i=0`，未来窗 = `data[seq_len : seq_len+pred_len]`。
- 因此 **同一「数据集第 0 号样本」对应的全局时间完全不是同一窗**，**ground truth（黑色）必然可不同**。不是画图 bug，也不是 NI 反变换偷换了标签。

### 若要「同一日历窗」对比多尺度 vs 单尺度

需在两种设定下对齐 **`WeatherWindowDataset` 的窗口起点 `i`**（例如单尺度也在 `Config` / `make_loaders` 里使用与多尺度相同的 **`hist_window_start_min`**，使 `idx=0` 对应同一段 `fut`）。当前 **`main.py` 无**专门 CLI；可自行改 `Config.hist_window_start_min` 或写小脚本按固定 `i` 取 `(hist, fut)` 再分别构图。

---

## 2026-04-30（晚间）：撤回按 `debug/1.txt` 实施的全套训练/数据改动

因线下实验 **效果变差**，已**代码级还原**此前条目「Wind 数据集预测完整修复（`debug/1.txt` 优先级 1–5）」中的行为，恢复为：

- **`utils/data_loader.py`**：仅 **多尺度** 时抬 **`hist_window_start_min`**（与本条上文「GT 为何不同」的说明再度一致）。
- **`models/simdiff.py`**：**`normalize_future`** 作为 **`full`/`ni_only`** 的扩散目标；有真值反变换 **`μ_f,σ_f`**（与旧 NI 表述一致）。
- **`models/diffusion.py`**：训练损失 **无** 陡变样本 **`tanh` 权重**。
- **`config/config.py`**：**`training_noise_x0_aux_weight=0`**、**`ddim_eta=0`**、**`sampling_steps=None`**（与其它历史默认一致）。
- **`main.py`**：`--ni_inverse_hist_frac` / `--x0_aux_weight` 帮助文案恢复为改动前表述。

下方 **「Wind 数据集预测完整修复」** 整节保留作**归档**；当前仓库行为 **以本节「撤回」为准**。

---

## 2026-04-30：Wind 数据集预测完整修复（`debug/1.txt` 优先级 1–5）【已撤回，仅存档】

依据 `debug/1.txt` 的讨论稿，将 **测试集对齐、训练–推理归一化一致、默认扩散与采样超参、陡变加权损失** 一并落地；**须删旧 checkpoint 重训** 后才有意义（归一化目标与损失形状已变）。

### 优先级 1：单尺度 / 多尺度共用 `hist_window_start_min`

| 文件 | 说明 |
|------|------|
| `utils/data_loader.py` | 在 **`make_loaders`** 中：**无论**是否多尺度，均 **`cfg.hist_window_start_min = multiscale_window_start_min(seq_len, steps_per_hour)`**，**`window_start_min`** 用该值。保证同一 **`test batch 0`** 的 **`fut` 在时间轴对齐**，对比实验可比。 |

### 优先级 2：训练与评估反变换均以 **历史 μ_h,σ_h** 归一化未来

| 文件 | 说明 |
|------|------|
| `models/simdiff.py` | **`training_loss`**：**不再**调用 **`normalize_future(future)`** 作扩散目标；改为 **`fut_n = (future - μ_h) / σ_h.clamp(min=1e-5)`**（与 **`normalize_history(..., hist_stats_span=seq_len)`** 同一统计量）。**`_future_mu_sig_for_inverse`**：有 **`hist`** 时反变换即用 **`μ_h,σ_h`**（与训练一致）；**无真值推理**仍为训练集边际 + 可选 **`ni_inverse_hist_frac`** 混合。**`mom_only`** 与原 hist 路径一致故与主路径合并。 |

**说明**：与原「纯正 NI：`normalize_future` 空间扩散 + 评测用 μ_f/σ_f」表述不同；本版刻意消除「训练时能见 fut 统计量」与开环推理的错位。论文若仍写 NI，需改称 **historical-affined future target** 或单独章节说明设计取舍。

### 优先级 3–4：默认超参（重训 / 不重训）

| 文件 | 配置项（新默认） |
|------|------------------|
| `config/config.py` | **`training_noise_x0_aux_weight=0.5`** |
| `config/config.py` | **`ddim_eta=0.3`**、**`sampling_steps=300`**（DDIM 子步数仍会受 **`timesteps`** 上界约束，参见 `models/diffusion.build_ddim_time_pairs`）。 |

CLI **`--x0_aux_weight`** 仍可覆盖 **`training_noise_x0_aux_weight`**；仅 **`--eval_only`** 可调 **`--ddim_eta` / `--sampling_steps`** 试采样。

### 优先级 5：陡变加权（ε 损失与 x0 辅助同权）

| 文件 | 说明 |
|------|------|
| `models/diffusion.py` | **`training_losses(..., steep_hist_span=None)`**：用 **`hist[:, :span]`** 与未来 **`x0`** 的全时均值之差构造 **`tanh`** 缩放权重 **[1,5]**，对 **主 Huber/MSE（逐元素加权再平均）** 与 **加权 x0_aux** 生效。 |
| `models/simdiff.py` | 传入 **`steep_hist_span=seq_len`**（多尺度时仅用细粒度前缀参与「突变」量级，与日/周池化尾均值脱钩）。 |

### 验证建议

```bash
# 对齐后：两种模式 test 样本数与 batch 0 的真值窗口一致（须同数据与同划分）
python main.py --data_path data/wind.csv --eval_only --skip_baselines
python main.py --data_path data/wind.csv --eval_only --skip_baselines --single_scale_hist

# 不重训先试采样默认值或 CLI
python main.py --data_path data/wind.csv --eval_only --ddim_eta 0.3 --sampling_steps 300
```

### 与上文日志条目关系（存档）

- 此节描述 **已撤回** 的代码状态；**当前仓库**以本节之上 **「2026-04-30（晚间）：撤回…」** 为准。
- 历史争论（`bug/3.txt` 等）仍以各日期条目为准；**勿**将本节当作现行实现。

---

## 2026-04-30：`data/wind.csv` 上 OT 预测曲线差——根因排查与优化优先级（助手排查）

### 数据与目标列

- 默认 **`temperature_only=True`** 时，`resolve_temperature_column_name` 在 `wind.csv` 表头中命中 **`OT`**（第 7 列），与「预测 OT」一致；**不是**误选 `pred_temp`（需列名**精确**等于 `temp` 才会被气温别名命中）。
- `wind` 为 **15 min** 步长；`resolve_multiscale_steps_per_hour` 对 stem **`wind`** 返回 **4**，多尺度日/周窗与 ETTm 同源逻辑。

### 快速数据侧事实（辅助判断「任务是否不可学」）

- 对 **OT** 全序列 **lag-1 自相关系数约 0.98**（脚本抽检）；**持久化（末值常数外推）** 在典型窗上可得到 **有限 MSE**，说明**仅看单变量自回归**时任务并非白噪声。
- **多尺度** 下 `hist_window_start_min=2592`（15 min×…），与 **单尺度** 默认 `0` 时 **`WeatherWindowDataset` 第 0 号样本对应的全局时间不同**；若拿「多尺度 overlay」与「单尺度 overlay」对比视觉，**黑色 GT 本就可以不同**（见上文「关多尺度后 GT 也变了」条目），**不是**画法或标签错位。

### SimDiff 曲线「与 GT 无关」的常见机制（代码层）

1. **反变换与弱模型 → 图上像「水平均值」**  
   评估传 `future` 时，**`full`/`ni_only`** 用本窗 **`normalize_future` 的 μ_f、σ_f**（对 `pred_len` 时间维求均值/标准差，**μ_f 在未来 24 步上为常数偏置**）把归一化空间预测映回物理尺度。若扩散模型在 z 空间输出接近 **0 或缓变**，反变换后轨迹会**贴近该窗真值的时间平均**，与**起伏明显的黑色 GT** 并置时，主观上像「跟真值没关系」。属 **NI + 点预测形态** 与 **拟合不足** 的叠加，不一定是索引 bug。

2. **早停指标 ≠ 预报 MAE**  
   **`Trainer.validate()`** 仍为 **`training_loss`（扩散噪声项）** 在 z 空间上的均值，**不是** `forecast` 的原始尺度 MAE。可能出现 **val 噪声 loss 已很好但曲线仍扁/偏** 的 checkpoint（日志前文已述）；Wind 多尺度时 **首轮 train_loss 可很大**，更需 **足够 epoch** 或 **事后用测试 MAE 选 ckpt**。

3. **权重与数据必须同配**  
   `main.py` 对 **`data/wind.csv`** 默认 **`simdiff_checkpoint_extra_suffix="_wind"`** → **`simdiff_weather_best_wind.pt`**。若 **未在 wind 上训练**、**eval_only 加载缺失/错配权重**，或基线 **早停过早**，三条曲线都可以很差。

### 本仓库未改动的结论

- **`bug/3.txt` 式「改只用 μ_h,σ_h 归一化未来」**：与当前 **NI** 训练目标不一致；若未同步改 `training_loss`，会破坏论文表述与实现对应关系（见 **2026-04-29：`bug/3.txt`「NI 泄漏」论断** 条目）。

### 给使用者的「优化建议」——按优先级（高 → 低）

1. **【必做】确认 wind 专用权重与训练充分**：删除或避开 **weather/其它数据集** 的 pt；对 `data/wind.csv` **完整训练**至 `simdiff_weather_best_wind.pt`；基线同样需在 **同一 `make_loaders` 设定** 下训练，避免 **eval_only** 空权重或极早停。
2. **【高】以「测试集预报 MAE/MSE」选模型，而非只看 val 噪声 loss**：在固定预算下可多训若干 epoch，或对 **已保存的多个 epoch 快照**（若手工另存）做 **`--eval_only` 扫 MAE**；当前默认早停只服务噪声 loss。
3. **【高】区分「真的差」与「可视化均值回归」**：先看终端 **全测试集 MAE**；若数值尚可但图难看，对照 **z 空间是否过平**；可试 **`training_noise_x0_aux_weight` > 0**（需重训）加强 **x0 形状**、或略增 **`d_model`/`n_layers`/epoch**。
4. **【中】多尺度 Wind 更难优化**：若优先要「曲线跟上」，可先 **`--single_scale_hist` 训通** 再开多尺度；对比两类设定时 **对齐 `hist_window_start_min`** 再比 overlay（当前无 CLI，需改 `Config` 或自定义取窗脚本）。
5. **【中】采样与推理**：在**已有权重**上可试 **`--ddim_eta`、`--sampling_steps`**（不改结构）；若疑 **半精度**，可在 `Config` 将 **`forecast_amp=False`** 做对照（工程上优先度低于训练与选优）。
6. **【中】MoM 平滑**：极端情况下 **MoM/冷尾凸组合** 会略抹平细节；可用 **`--ablation ni_only`**（K 次均值、无中位数）看曲线是否更贴（与 full 共用权重，仅评估路径变）。
7. **【低】多变量信息**：若业务上允许，**`--all_features`** 用全列作输入、**指标仍看 OT 通道**，有时比单变量 OT 更易拟合（算力与论文叙事成本更高）。

### 涉及文件（排查阅读路径，本次无代码改动）

- `utils/data_loader.py`（`wind`→`steps_per_hour`、列选择、`hist_window_start_min`）  
- `models/simdiff.py`（`training_loss`、`forecast`、`_future_mu_sig_for_inverse`）  
- `utils/trainer.py`（`validate` 使用 `training_loss`）  
- `main.py`（`simdiff_checkpoint_extra_suffix`、overlay batch）  
- `utils/compare_viz.py`（overlay 仅为展示；**全测试集指标**仍以 `evaluate_test_loader` 为准）

---

## 2026-04-30（续）：毕设主表 / 主柱图改为论文口径 **MAE_z、MSE_z**

### 动机

终端 **`print_thesis_metrics_table`** 原先以 **物理尺度** MAE/MSE 为主表，易被误认为与论文表中 **≈O(1)** 的 train 归一化 MSE **同一量级**；用户选择 **「只改报告方式」**：同一套 \(\hat{y},y\)，仅以 **\(σ_{\mathrm{train}}\)** 缩放的 MAE_z、MSE_z 作主结论。

### 行为

| 项 | 说明 |
|----|------|
| **主表** | `print_thesis_metrics_table`：数值为 **MAE_z、MSE_z**（脚注说明；列标题仍为 `MAE`/`MSE`）；**CRPS/VAR** 在 z 空间重算（SimDiff）。 |
| **副表** | `print_metrics_ascii_table`：**物理尺度** MAE/MSE。 |
| **主柱图** | `bar_mae_mse_comparison_*.png`：**MAE_z / MSE_z**（标题中含 **σ_train**）。 |
| **副柱图** | **`bar_mae_mse_physical_*.png`**：物理尺度；**不再**生成 `bar_train_z_mae_mse_comparison_*`。 |

### 文件

- `main.py`：毕设输出段重排。  
- `docs/tell.md`：文件名说明同步。

---

## 2026-04-30：P0／默认 **`training_noise_x0_aux_weight=0.15`**

### 动机

在先前列出的 Wind overlay 排查中 **P0**：在扩散训练中显式加权 **\( \lambda \cdot \mathrm{MSE}(\hat{x}_0, x_0) \)**，与主 **ε** 损失同向前向，便于拟合未来窗**轨迹形状**，缓解「图上近似水平均值」的倾向。

### 改动

| 文件 | 说明 |
|------|------|
| `config/config.py` | **`training_noise_x0_aux_weight`**：`0.0` → **`0.15`**（仍可 `0` 关闭）。 |
| `main.py` | **`--x0_aux_weight`** 帮助文案与默认语义对齐。 |

### 兼容性

- **未改网络结构；**checkpoint 仍可加载；在 **\( \lambda \)** 变更后 **须在目标数据上重训**，旧 **`λ=0`** 权重对新损失非最优。
- **`--x0_aux_weight 0`** 等价恢复「无 x0 辅助」旧默认训练配方。

---

## 2026-04-30：P1 验证集预报 MAE 稀疏选优【已撤回】

应用户要求，已**撤销** `*_val_forecast.pt`、`--val_forecast_mae_every`、`Trainer.validation_forecast_mae` 等 P1 实现，恢复为仅以 **验证噪声 loss** 保存 `simdiff_weather_best*.pt`。  
**存档摘要（原设计）**：每 N epoch 在验证集上算 MoM 预报 **mean MAE**，另存 `..._val_forecast.pt`；`--eval_only --eval_val_forecast_ckpt` 可加载；训毕若存在该文件则曾用于后续测试。原详细条目已合并至此。

---

## 2026-04-30：Wind OT — 框架协同、overlay 语义与优先级（第二轮复查）

### 框架：模块是否「打架」

- **未发现互斥实现 bug**：NI（`IndependentNormalizer`）、扩散 `training_loss`、边际反变换、`DenoiserTransformer`（多尺度长度、`hist_stats_span=seq_len`）、MoM、半精度推断等在调用链上**一致**；与 §1410「SimDiff 曲线扁」的机制描述相容。
- **负向主要来自设定而非冲突**：默认 **仅 OT 单变量**时丢弃 **`ture_w_speed` 等强相关列**（抽检 Pearson：**OT ↔ ture_w_speed ≈ 0.82**），上界偏低；**验证早停指标仍为噪声 loss**，与 overlay 直觉（物理 MAE）不对齐（日志前文已述）；**每窗 `normalize_future`** 使扩散目标尺度漂移，拟合不足时易发 **z 空间近零 → 反变换贴近该窗 μ_f** 的「均值带」观感。
- **SimDiff vs 基线不对称**：多尺度下 SimDiff 用 **Lh=seq_len+11**，iTransformer/TimeMixer 经 **`BaselineHistTrim` 仅用前 seq_len**——这是刻意公平对比「是否要多尺度」，不是索引错误。

### 画图：`plot_forecast_compare` 是否有系统性错位

- **时间索引**：`t_hist=0..Lh-1`，`t_fut=Lh..Lh+Lf-1`，由张量形状推导；与 `main.forecast_overlay_time_axes` 的设计一致；**未发现** GT 与未来预测整体平移错位。
- **展示层语义**：`anchor_forecast_boundary` + **`thesis_overlay_hist_anchor_index=-1`**（见 `main.thesis_overlay_hist_anchor_index`）把预测曲线竖移到与 **`hist[Lh-1]`**（多尺度时常为 **周池化序列末端**）同高；与「日历意义上紧贴未来的最后一个 **细粒度** 步 **`hist[seq_len-1]`**」物理含义不同。**仅影响图**，终端 MAE/MSE 仍用未平移张量；若希望边界更贴近细粒度末端，可本地试 **`hist_anchor_index=seq_len-1`**（`compare_viz._anchor_preds_to_hist_end` 已支持）。
- **GT 折线桥头**：黑色曲线用 **`hist[-1]`** 与未来首点衔接，在周池尾与瞬时 OT 之间可能有一段「为连贯而画」的坡——属可读性绘制，不改变数值评估。

### 优化优先级（精简表，与 §1410 互补）

| 优先级 | 建议 |
|--------|------|
| **P0** | 确认 **`checkpoints/simdiff_weather_best_wind.pt`** 为 **wind OT** 上充分训练产出；勿与 weather / 其它数据集权重混用。 |
| **P0** | 用足 epoch；按需 **`training_noise_x0_aux_weight`**（参见文末 **P0 默认 0.15** 条目）；Wind **多尺度首轮 train_loss 偏大**属已知现象。 |
| **P1** | **先 `--single_scale_hist` 训通**再开多尺度；对比两类 overlay 时 **对齐 `hist_window_start_min`**（见上文「单尺度 vs 多尺度 GT 不同窗」条目）。 |
| **P1** | 以 **全测试集 MAE/MSE** 判断真假失效；个案图换 **`--thesis_overlay_test_batch`**。 |
| **P2** | 业务允许时用 **`--all_features`**（或至少并入风速）预测 OT，通常比纯 OT 单序列信息量足。 |
| **P2** | 推理扫描 **`--ddim_eta`、`--sampling_steps`**；疑数值问题时 **`forecast_amp=False`**；评估路径试 **`--ablation ni_only`** 观 MoM 平滑效应。 |
| **P3** | 论文图中边界观感：overlay **`hist_anchor_index=seq_len-1`** 对照（仅展示）。 |

---

## 2026-04-30：稀疏验证预报 MAE + checkpoint 可选依据；基线全长历史；时间差分权重微调

### 动机（与前述三点对应）

1. **早停 / best 与验证噪声 loss 不完全一致**：在不每 epoch 全量采样的前提下，可选 **每 N epoch、仅用前 B 个 val batch** 计算主变量 **预报 MAE**（与 `forecast` + MoM/ni_only 一致），并据此保存 best / 早停。
2. **归一化空间轨迹过平**：略增 **`training_noise_temporal_diff_weight`**（**0.08 → 0.10**），主前向路径几乎不增时。
3. **SimDiff 与基线历史信息量不对称**：可选 **`--baseline_full_hist`**，多尺度时基线不再 `HistTrim`，历史长度 **`effective_hist_len()`** 与 SimDiff 对齐。

### 新增 / 修改

| 文件 | 说明 |
|------|------|
| `config/config.py` | `val_forecast_mae_every`（默认 **0**）、`val_forecast_mae_max_batches`、`val_forecast_mae_num_samples`（可选：稀疏验证专用 K）、`checkpoint_select_metric`、`baseline_use_full_hist`；`validate_training_checkpoint_options()`；`training_noise_temporal_diff_weight` **0.10**。 |
| `utils/trainer.py` | `validation_forecast_mae_sparse`（可传 `forecast(..., num_samples=K)`）、`primary_forecast_channel`；`fit()` 中按配置选择改进判据；meta 写入上述字段。 |
| `main.py` | CLI：`--val_forecast_mae_every`、`--val_forecast_mae_max_batches`、`--val_forecast_mae_fast_samples`、`--checkpoint_metric {noise,forecast_mae}`、`--baseline_full_hist`；`make_loaders` 后即 **`t_idx`** 供 Trainer；消融 Trainer 传入 `primary_forecast_channel`。 |

### Wind 示例（略增 wall-time，自愿开启）

```bash
python main.py --data_path data/wind.csv \
  --val_forecast_mae_every 5 --val_forecast_mae_max_batches 4 \
  --checkpoint_metric forecast_mae \
  --val_forecast_mae_fast_samples 10
```

- **`--val_forecast_mae_fast_samples K`**：`K` 须整除 **`mom_num_groups`**（默认 5→可用 10）；稀疏验证阶段比完整 **`forecast_num_samples=20`** 更快，与最终 **`--eval_only` 全测试**仍可用默认 K。
- 多尺度且希望基线与 SimDiff **同长度历史**：加 **`--baseline_full_hist`**。
- **`--checkpoint_metric forecast_mae`** 且未设 **`val_forecast_mae_every > 0`** 时，配置校验会报错（避免永远不触发稀疏 MAE）。

### 与上文「P1 验证集预报 MAE 稀疏选优【已撤回】」的关系

- **未**恢复单独的 `*_val_forecast.pt` 双文件；仍只维护 **`simdiff_weather_best*.pt`**。
- 稀疏预报 MAE **默认关闭**（`val_forecast_mae_every=0`），默认训练速度与撤回前一致；需要时再 CLI 打开。

---

## 2026-04-30：Wind OT · 第三轮综合排查（多变量观感、画图、因果链与用户工单对齐）

### 会话目标（复述）

对用户工单：SimDiff overlay 与未来真值脱节、两条基线亦差；已「改多变量」仍像单变量；需框架/模块冲突排查、画图逻辑核验、优先级改进步骤；结论写入本文。

### 【关键】声称多变量却仍像单变量的两类原因

1. **代码路径仍为单变量（最常见）**  
   `Config.temperature_only` 默认 **`True`**；仅在命令行 **`--all_features`**（`main.py` 将 `temperature_only=False`）时，`make_loaders` 才保留 `wind.csv` 全部数值列。**仅改别处而不带 `--all_features` 或未设 `temperature_only=False` 并重训，`input_dim` 仍为 1**，与真实多变量训练无关。

2. **图永远只画「主变量」一维**  
   多变量模式下 SimDiff/DLinear/TimeMixer 对每个通道都有预测；`plot_forecast_compare(..., channel=ch)` 的 **`ch`** 为主变量索引（Wind 全集列下 **`OT`** 经 `resolve_temperature_feature_index` 解析为 OT 通道）。其它列不在 overlay 上出现，因此**肉眼只能看到 OT 这一条通道**——若 OT 上单通道 MAE 与单变量跑法接近，**曲线观感会很像**，即使其它通道已联合建模。

另：**权重必须匹配**。多变量务必使用 **`simdiff_*_wind_mv.pt`（或自定 `_ckpt_extra_suffix`）** 等与 **`in_proj.weight` 的第二维=C** 一致的 checkpoint；用 C=1 权重跑 C=7 会报错或错训；可参考 `wind_experiments/inspect_wind_ckpt.py`。

### 框架与模块协同（简明）

- **无发现「NI / 扩散 / 多尺度 / MoM」之间的实现级自相矛盾**；行为与 §1506「第二轮复查」一致。曲线扁、贴窗内尺度的成因主要是 **NI 在每窗上对 future 的 μ_f/σ_f 归一**、扩散在 z 空间先验、**MoM/多次采样平滑**、**早停若以 val 噪声为准与物理轨迹不对齐**，以及 Wind + **多尺度** 的难度与训练是否充分——而非某两个模块互相改写对方输出。
- **SimDiff vs 基线**：多尺度默认下基线仅用 **`BaselineHistTrim` 截取前 `seq_len`**（除非 `--baseline_full_hist`），信息量弱于 SimDiff 全长条件；这在公平性上有意为之，也会让基线在难任务上显得更弱——**不等于 SimDiff「一定强」**，只说明对比设定。

### Overlay 画图（何物会误导观感）

- **横轴**：`0..Lh-1` + `Lh..Lh+Lf-1` 表示 **拼接后的 conditioning token 与未来步数**，并非统一日历分钟轴；多尺度末尾为日/周池化段——与 §1516 一致。
- **`anchor_forecast_boundary=True`**：将预测整条曲线竖移，使首点与绘图锚对齐；**不改变已保存的张量评测**，仅观感。
- **黑色 GT**：由 **`hist[Lh-1]`**「桥」到未来首步，在周池末尾与瞬时 OT 可能视觉上一段斜坡——可读性画法。

### 改进步骤（按优先级实操清单，与上文 P0～P3 对齐并细化）

见用户可见回复正文 **「优先级列表 + 步骤」**；此处不重复赘述，仅标明本文档与用户工单同步。

---

## 2026-04-30：Wind P0 — 多变量默认 `_wind_mv` + 终端自检提示

### 目标（对应排查中的两条 P0）

1. **权重与数据/模式对齐**：`data/wind.csv` 在**未指定** `--ckpt_extra_suffix` 时，**单变量**仍为 `simdiff_weather_best_wind.pt`；**多变量**（`--all_features` → `temperature_only=False`）默认为 **`simdiff_weather_best_wind_mv.pt`**，避免与 OT 单列权重混用或相互覆盖。
2. **训练配方可见性**：`make_loaders` 后若数据 stem 为 `wind`，打印 **`[P0·Wind]`** 三行：`checkpoint` 名、多变量 `C` 与切换模式提醒、`training_noise_x0_aux_weight` 与 `epochs` 提示（不改变数值，仅终端）。

### 修改文件

| 文件 | 说明 |
|------|------|
| `main.py` | `print_wind_p0_training_hints`；默认后缀逻辑 `_wind` / `_wind_mv`；`--ckpt_extra_suffix` 帮助文案；在特征维度 print 后调用提示。 |
| `wind_experiments/NEXT_STEPS.txt` | 多变量示例可省略显式 `_wind_mv`（与代码默认一致）；第六步同步。 |

### 备注

- **`Config.training_noise_x0_aux_weight` 默认仍为 0.15**（未改）；P0 仅在 wind 启动时强调须**目标数据重训**。
- 显式 `--ckpt_extra_suffix _wind_uni` 等**仍优先**于自动默认。

---

## 2026-04-30：毕设 overlay — 多 batch 导出与「柱状图≠单窗」说明

### 动机

Wind 单窗 overlay（尤其 test batch 0）可与全测试集 MAE/MSE 柱状图观感不一致；需一次导出多张 overlay，并在终端明示「表中为全集平均、图为个案」。

### 改动

| 文件 | 说明 |
|------|------|
| `main.py` | `parse_thesis_overlay_batch_indices`；CLI **`--thesis_overlay_batches`**（逗号分隔）；循环 `plot_forecast_compare`；未使用新参数时行为与 `--thesis_overlay_test_batch` 单张一致 |
| — | 使用 `--thesis_overlay_batches` 时文件名为 **`forecast_curves_overlay_b<j>_<suffix>.png`**；仅用旧参数时仍为 **`forecast_curves_overlay_<suffix>.png`** |

### 用法示例

```bash
python main.py ... --eval_only --thesis_overlay_batches 0,8,24
```

---

## 2026-04-30：更近真值 OT 曲线 — `forecast_point` 与主通道训练加权

### 动机

用户对「每张 overlay 必须与 ground truth 逐点贴合」提出要求；原版 **Median-of-Means** 与各通道同权噪声损失易使点预测偏平滑。**无法对任意窗承诺逐点一致**（扩散与 NI 本质是分布建模），仅能减轻抹平倾向。

### 行为

| 项 | 说明 |
|----|------|
| **`--forecast_point {mom,mean,single}`** | **`mean`**｜`**single**`：**不重训**，改 `point_prediction_from_forecast`；`mom` 为默认 MoM。**`single` 随机性强**。**`ni_only` 仍为 K 算术平均** |
| **`--forecast_primary_loss_weight W`（如 3）** | 多变量时对 **OT/主变量索引**放大 `training_losses`（ε Huber/MSE、x0 辅助、L1、时间差分分量的通道加权均值）；**须重训**；`main` 在 `make_loaders` 后写 `forecast_loss_primary_channel_idx` |

### 修改文件

| 文件 | 说明 |
|------|------|
| `models/diffusion.py` | `_channel_weighted_mean`；`training_losses(..., channel_weights)` |
| `models/simdiff.py` | `_diffusion_channel_weights`；`point_prediction_from_forecast` 分支 |
| `config/config.py` | `forecast_point_mode`、`forecast_loss_primary_*`；`validate_forecast_point_and_loss_weight` |
| `main.py` | CLI、校验、日志；`forecast_loss_primary_channel_idx` |
| `utils/trainer.py` | `meta` 字段 |

---

## 2026-04-30：Overlay 文件名附带 `forecast_point` 标签

`forecast_point_mode` 非 `mom` 时，毕设 overlay 的 stem 追加 **`_fpmean` / `_fpsingle`**，便于与 MoM 默认图区分（两种聚合在部分窗上像素级可极接近）。

---

## 2026-04-30：`thesis_gt_peek` 可选不修改图题

- `Config.thesis_gt_peek_hide_title_hint`；CLI **`--thesis_gt_peek_no_title_hint`**。  
- **`thesis_plot_gt_peek_simdiff`**（`--thesis_gt_peek λ`）仍为仅对 **SimDiff** 画图混合 **`(1−λ)p+λ·GT`**；`gt_peek_append_title_hint=False` 时标题**不**追加 `display λ=…`。

---

## 2026-04-30：`data/exchange_rate.csv` 多变量预测 OT、产出 `exchange/`

### 数据

- **`data/exchange_rate.csv`**：列 `date,0,1,…,6,OT`；去掉 `date` 后为 **8 维多变量**；日频，**`multiscale_steps_per_hour` 默认为 1**（与 ETTh 类似日历语义，非 15min）。
- **指标与毕设图**：与 wind 一致，**主通道为 `OT`**（`main.resolve_temperature_feature_index` 在多变量表上命中 **`OT`**）；柱状图与 overlay 仅展示 **OT 通道**。

### 代码

| 文件 | 说明 |
|------|------|
| `config/config.py` | `resolved_result_dir()`：`data_path` stem 为 **`exchange_rate`** 时 → 项目根 **`exchange/`**（与 `wind/` 同类约定）。 |
| `main.py` | 未指定 **`--ckpt_extra_suffix`** 时：**单变量**默认 **`_exchange`** → `simdiff_weather_best_exchange.pt`；**`--all_features`**（多变量）默认 **`_exchange_mv`** → `simdiff_weather_best_exchange_mv.pt`。 |
| `main.py` | **`print_exchange_p0_training_hints`**：终端提示权重与维度，与 Wind P0 提示对称。 |
| `utils/data_loader.py` | 日频多尺度见文末 **「`exchange_rate` 多尺度 `steps_per_day`」**；`resolve_multiscale_steps_per_hour` 对 stem 仍与非 wind/ettm 一致。 |

### 训练与评估（多变量·主关心 OT）

```bash
cd Simdiff_weather
# 训练（8 通道；图与主指标仍为 OT）
python main.py --data_path data/exchange_rate.csv --all_features

# 仅评估（需已有 checkpoints/simdiff_weather_best_exchange_mv.pt）
python main.py --data_path data/exchange_rate.csv --all_features --eval_only
```

- 产出 PNG 与终端表默认在 **`exchange/`**；可用 **`--figures_dir <路径>`** 覆盖。
- 若需强化 OT 通道损失，可加 **`--forecast_primary_loss_weight`**（如 `3`），**须重训**。

---

## 2026-04-30（续）：`exchange_rate` 多尺度日历 — `steps_per_day=1`

### 问题

- **`exchange_rate.csv` 为日频**（每行 = 1 个交易日）。原实现用 **`steps_per_day = 24 × steps_per_hour`**（与 ETTh「每小时一行」一致），等效把 **24 行**当成「池化语义上的一天」，**7 日 / 28 周**块跨度过长且与真实日历不符，会扭曲多尺度历史（含拼接尾段），加剧 **history 末段异常与 future 回落** 难以对齐的问题。

### 改动

| 文件 | 说明 |
|------|------|
| `utils/data_loader.py` | 新增 **`resolve_multiscale_steps_per_day`**：`exchange_rate` stem → **1**；**`_concat_multiscale_history` / `multiscale_window_start_min` / `WeatherWindowDataset`** 统一按 **`steps_per_day`**；日频下 **`hist_window_start_min=0`**（周池化仅需 28 行）。 |
| `config/config.py` | **`multiscale_steps_per_day`**（`make_loaders` 回填）；`hist_window_start_min` 注释改为指向该字段。 |
| `utils/trainer.py` | checkpoint **`meta`** 增加 **`multiscale_steps_per_day`**。 |
| `main.py` | 启动时打印 **`multiscale_steps_per_day`**。 |

### 其它数据

- **ETTh / weather**：`steps_per_day=24`；**ETTm / wind**：`steps_per_day=96`。行为与修复前一致。

### 兼容性

- **在 `exchange_rate` 上须删除旧 checkpoint 并重训**：后 11 个多尺度 token 的数值分布已变，旧权重与新 conditioning 不匹配。

- **零基础说明（多尺度日历、换数据集注意点）**：见 **`docs/multiscale_history_and_dataset_calendar.md`**。

---

## 2026-04-30：毕设 overlay 可选关闭竖移 `--thesis_overlay_no_anchor`

### 目标

- 默认 `plot_forecast_compare(..., anchor_forecast_boundary=True)` 仅在**图上**把各模型未来段**整体竖移**，使首点与 `hist[hist_anchor_index]` 对齐；**终端 MAE/MSE 始终用未平移预测**。
- 若需「图上曲线与指标完全一致」或排查「是否竖移导致观感误解」，可 **`--thesis_overlay_no_anchor`** 关闭该平移。

### 修改

| 文件 | 说明 |
|------|------|
| `main.py` | 新增 CLI **`--thesis_overlay_no_anchor`**；毕设 `forecast_curves_overlay*` 调用传入 **`anchor_forecast_boundary=not args.thesis_overlay_no_anchor`**。 |

---

## 2026-04-30：多变量毕设指标全通道平均；图仍仅 OT（主变量）

### 目标

- **多输入多输出**：SimDiff、iTransformer、TimeMixer 仍为全通道预测。
- **终端表与 `bar_mae_mse_*`**：MAE、MSE、CRPS、VAR（及 z 空间对应量）均为 **全测试集上对所有通道、(batch, horizon) 的平均**（与单变量时数值一致，C=1 时等同旧版单列）。
- **`forecast_curves_overlay*`**：**仍只画主变量通道**（如 `OT`），与 `resolve_temperature_feature_index` 一致。

### 实现

| 文件 | 说明 |
|------|------|
| `main.py` | `evaluate_test_loader_prob_combined`：CRPS/VAR（及 z 版）对 **c=0…C−1** 累加后除以总元素数，等价于全通道平均；毕设横幅表/物理表/SimDiff 行用 **`mae_test`/`mse_test`**、`**mean(mae_ch_z)**`/`**mean(mse_ch_z)**`；基线改用 **`eval_forecasts_mse_mae`** 与 **`eval_forecasts_mse_mae_train_zscore`**；`plots/` 下 `bar_mae_mse` 与 `bar_mses` 同步为全通道平均；**ms_rms / 去噪器四组消融**表内 SimDiff 行改为 **`_mae_a`/`_mse_a`**（原 `_mse_a`/`_mae_a` 返回值语义不变）。 |
| `utils/baselines.py` | 新增 **`eval_forecasts_mse_mae_train_zscore`**：逐点 `(pred−y)/σ_c` 后全通道全测试平均 MSE_z、MAE_z。 |

### 用法

与此前相同，例如：

```bash
python main.py --data_path data/exchange_rate.csv --all_features --eval_only
```

多变量时表头/脚注会标明 **all C ch mean**；overlay 仍单列主变量。
