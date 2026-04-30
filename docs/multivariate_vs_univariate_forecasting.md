# Wind 数据集：单变量预测 vs 多变量预测（与本项目实现的对应关系）

本文说明在 **SimDiff-Weather** 中，默认 **仅 OT 单变量** 与 **`--all_features` 多变量** 各自代表什么、_tensor 形状如何变化、评估指标落在哪一维，以及如何与你此前的 Wind 实验对照。

---

## 1. 你「之前的做法」：单变量（默认）

### 1.1 配置含义

- **`temperature_only=True`**（默认），且不加 **`--all_features`**。
- `utils/data_loader.make_loaders` 会调用 **`resolve_temperature_column_name`**：在 `data/wind.csv` 表头里依次匹配气温别名；Wind 一般为 **`OT`**，最终 **只保留这一列**。
- 此时 **`cfg.input_dim = n_features = 1`**。

### 1.2 数据张量含义

| 符号 | 含义（单变量） |
|------|----------------|
| `hist` | `(B, Lh, 1)`：历史窗口内 **仅有 OT** 这一条序列（多尺度开启时 `Lh = seq_len + 11`，否则 `Lh = seq_len`）。 |
| `fut` | `(B, Lf, 1)`：未来 `pred_len` 步内 **仅有 OT** 真值。 |

模型学习与采样均在 **`C = 1`** 上进行：**历史上看见的也只有 OT**，无法在同一窗口内直接使用风速、气压等通道。

### 1.3 评估与作图

- **主变量索引** `t_idx = 0`（唯一通道）。
- 终端 MAE/MSE、毕设柱状图、`forecast_curves_overlay` 中的曲线：**全部是 OT（原始物理尺度）**。

### 1.4 典型命令

```bash
python main.py --data_path data/wind.csv
# 等价于 temperature_only=True；Wind 会自动使用 checkpoints/simdiff_weather_best_wind.pt（若未手写 --ckpt_extra_suffix）
```

---

## 2. 「多变量」做法：`--all_features`

### 2.1 配置含义

- 命令行 **`--all_features`** → 代码里 **`cfg.temperature_only = False`**。
- `make_loaders` **不再**裁剪到单列，保留 CSV 中 **除 `date` 外全部数值列**。
- 对 **`data/wind.csv`**，通常为 **7 列**（顺序以 CSV 表头为准），例如：

  `pred_w_speed, pred_w_dir, pred_temp, pred_pressure, pred_humidity, ture_w_speed, OT`

- 此时 **`cfg.input_dim = n_features = 7`**。

### 2.2 重要概念：不是「只开一个输出的回归」，而是联合预测

在本项目中，SimDiff（及当前的 iTransformer / TimeMixer 基线）对应：

- **输入**：`(B, Lh, C)` —— **多通道历史**。
- **输出（监督目标）**：`(B, Lf, C)` —— **未来每一步上 C 个变量都要有真值**，扩散在对 **整段未来、所有通道** 的归一化空间建模后再反变换。

因此从网络结构上讲，这是 **多输入、多输出（MIMO）联合预报**：同一套参数同时刻画 **未来窗口内全部变量的联合演化**，而不是单独只生成 OT 一条序列、其它列仅当特征不进损失。

### 2.3 那为什么还说「主要看 OT」？

训练好后，`main.py` 会通过 **`resolve_temperature_feature_index(feat_names)`** 在多变量列表里解析 **主评估通道**：Wind 全列时命中列名 **`OT`**（一般为 **最后一列，索引 `t_idx = 6`**）。

- **终端表、柱状图、overlay**：仍以 **OT 通道**为主结论（与你论文「预测 OT」叙事一致）。
- 其它通道：**同样被预测**，并在训练中参与损失；它们常作为 **辅助任务 / 额外上下文表征**，有助于 OT（尤其在 OT 与风速等高度相关时）。

### 2.4 典型命令（建议单独 checkpoint 后缀）

单变量权重 **`input_dim=1`** 与多变量 **`input_dim=7`** **结构不同，不能直接混用同一 `.pt`**。

```bash
# 训练（示例后缀，避免覆盖单变量 best）
python main.py --data_path data/wind.csv --all_features \
  --ckpt_extra_suffix _wind_mv

# 仅评估 / 出图
python main.py --data_path data/wind.csv --all_features \
  --ckpt_extra_suffix _wind_mv \
  --eval_only
```

若未指定 `--ckpt_extra_suffix`，Wind 默认会使用 **`_wind`**；指定 **`_wind_mv`** 后，以该后缀为准，权重文件名形如 **`simdiff_weather_best_wind_mv.pt`**（具体规则见 `Config.simdiff_checkpoint_filename()`）。

---

## 3. 并排对比小结

| 项目 | 之前：单变量（默认） | 多变量：`--all_features` |
|------|----------------------|---------------------------|
| **CSV 使用的列** | 仅 **OT** | **全部数值列**（Wind 约 7 列） |
| **`input_dim` / `C`** | **1** | **7**（Wind；其它数据集依列数而定） |
| **历史 `hist`** | `(B, Lh, 1)` | `(B, Lh, C)` |
| **未来 `fut`** | `(B, Lf, 1)` | `(B, Lf, C)` |
| **模型输出语义** | 只建模 OT 的未来轨迹 | **联合**建模 **所有通道**的未来 `(Lf, C)` |
| **主评估通道** | `t_idx = 0`（唯一通道） | **`OT` 对应列**，Wind 一般为 **`t_idx = 6`** |
| **权重兼容性** | 原 **`…_wind.pt`**（单变量） | **须单独训练**；不可用单变量 ckpt 加载到 `C=7` |
| **直观利弊** | 实现简单、叙事「纯 OT」；丢掉风速等强相关信息时 **更难拟合陡变** | 参数与优化更难；往往 **更易利用风速等与 OT 相关的信息**，OT 通道指标有机会更好 |

---

## 4. 与「多输入单输出」表述的差别（避免术语混淆）

- **常见口头说法**：「多变量输入，只关心 OT 这一个输出。」
- **本仓库实际实现**：更准确地说是 **「多通道输入 + 多通道未来监督（联合预测）」，报表与 overlay 聚焦 OT 那一维**。  
  若需要 **严格的「其它列仅作输入、网络只输出 OT」**（输出形状 `(B, Lf, 1)`），需要在数据管线与模型头上单独扩展，**当前默认入口不包含这一种结构**。

---

## 5. 实验对照建议（与你此前 Wind 单变量实验可比）

1. **固定**：`data/wind.csv`、`seq_len`、`pred_len`、划分比例、随机种子、是否多尺度等与单变量那次一致。
2. **唯一变量**：是否加 **`--all_features`**。
3. **对比指标**：均以 **`OT` 通道**（多变量下的 `t_idx`）的测试 MAE/MSE（以及你关心的 z 口径表）为准。
4. **图表**：overlay 仍为 OT；若要多变量其它通道的诊断图，需另行脚本或扩展（默认毕设图为 OT 主变量）。

---

## 6. 相关代码入口（便于溯源）

| 说明 | 路径 |
|------|------|
| 是否只用单列 | `utils/data_loader.make_loaders`、`resolve_temperature_column_name` |
| `--all_features` | `main.py` 参数解析 |
| 主变量通道索引 | `main.resolve_temperature_feature_index` |
| NI / 形状 `(B,L,C)` | `models/simdiff.py`、`utils/independent_normalizer.py` |

---

*文档用途：Wind OT 场景下单变量默认流程与 `--all_features` 多变量流程的对照说明；与具体日期实验可在 `DEVELOPMENT_LOG.md` 中交叉引用。*
