# SimDiff-Weather 开发与改动记录

本文档随毕设/迭代持续更新：每次有结构性或行为性改动时，在文末 **追加** 新小节，并注明日期与相关文件。避免删除历史条目，便于对比实验与复现。

---

## 使用说明

- 一次改动对应一个小节，包含：目标摘要、动过的文件、配置项、与既有逻辑的关系、备注（权重兼容性等）。
- 与论文/开题报告对应的表述可引用本节中的「设计说明」。

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

*（在下方 `---` 之后追加新日期的改动小节。）*
