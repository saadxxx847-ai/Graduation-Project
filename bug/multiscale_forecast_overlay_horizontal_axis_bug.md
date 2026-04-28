# 多尺度历史下预报叠图横轴错位 Bug 记录

> 用途：答辩 / 汇报时可说明「实验过程中遇到的问题与处理」。  
> 关联代码：`main.forecast_overlay_time_axes`、`utils/compare_viz.plot_forecast_compare`。  
> 记录在仓库中的变更说明：`docs/DEVELOPMENT_LOG.md`（2026-04-29 条目）。

---

## 1. 现象（实验中发现了什么）

在 **`forecast_curves_temperature_overlay`** 一类的图里：

- **`history`**（历史观测）是一段灰色折线；
- **`ground truth`**（未来段真实值）应为黑色粗线；
- **各模型预测**为彩色虚线/点划线。

启用 **多尺度历史拼接**（`use_multiscale_hist=True`，历史张量长度为 `seq_len + 11`，例如在 `seq_len=96` 时长度为 **107**）后，图上出现：

- **`ground truth` 与未来预测在横轴上出现不合理的「扭折」**：黑色真值线看起来像 **折回到虚线左侧**，或与灰色历史段 **在时间上重叠**；
- 观感上像是「真值画错了」或模型输出异常，但核对 **数值评估（MAE/MSE 等）**：预测与标签的计算并未因此出错。

结论：**问题出在可视化坐标，而非扩散模型或损失定义本身。**

---

## 2. Bug 成因（为什么会出现）

### 2.1 数据结构

多尺度数据中，滑动窗口给的 **`hist`** 在时间维上是 **整块 conditioning**，长度为：

\[
L_h = \texttt{effective\_hist\_len} = seq\_len + 11
\]

（前 `seq_len` 步为与高分辨率对齐的原始序列，后面是多尺度聚合段；实现见 `utils/data_loader._concat_multiscale_history`。）

未来 **`true_fut`**、`pred` 的形状均为 `(pred_len, channels)`，只对应「紧接着 conditioning 之后的 `pred_len` 步」，**在时间语义上应紧贴在整块 `hist` 之后**。

### 2.2 绘图时的错误假设

绘图函数 `plot_forecast_compare` 接收：

- `t_hist`：长度为 `len(hist)`，表示 history 每个点对应的 **横轴索引**，一般为 `0, 1, ..., L_h-1`；
- `t_fut`：**未来段**对应的横轴索引，长度为 `pred_len`。

旧代码里 **`t_fut` 被错误地仍按「单尺度」写法生成**：

```text
t_fut = seq_len, seq_len+1, ..., seq_len + pred_len - 1
```

即从 **`seq_len`（例如 96）** 开始就画「未来」，而 **`t_hist` 在多尺度时已延伸到 `L_h-1`（例如 106）**。

于是：

1. **横轴重叠**：区间 **`x ∈ [seq_len, L_h - 1]`（例如 96～106）** 内，既画了 **`hist`**（整块历史中后部），又把 **`true_fut`、`pred` 的第一个点画在同一个区间**。同一时间区间内重叠两条语义不同的序列，视觉上产生 **穿插、扭折**。
2. **连接线折返**：`compare_viz.plot_forecast_compare` 为消除「历史末端与真值起点」的缝隙，在 **`(t_hist[-1], hist[-1])`** 与 **`(t_fut[0], true_fut[0])`** 之间画了辅助连线。当 **`t_fut[0] = seq_len` 且 `t_hist[-1] = L_h - 1 > seq_len`** 时，**连线在 x 方向上从右向左**，出现明显的 **横向折返**，黑色真值看起来像「扭到虚线左边」。

因此：**不是 ground truth 数值错位，而是横轴把「未来」起点提前到了 `seq_len`，与多尺度历史的真实长度 `L_h` 不一致。**

---

## 3. 解决方案（如何实现修复）

### 3.1 正确的横轴约定

定义 **`ehl = effective_hist_len()`**，则：

- **`t_hist = 0, 1, ..., ehl - 1`**（与 `hist` 行数一致）；
- **`t_fut = ehl, ehl+1, ..., ehl + pred_len - 1`**（未来段紧接整块 conditioning，**不与 history 在 x 上重叠**）。

项目在 `main.py` 中新增 **`forecast_overlay_time_axes(cfg)`**，统一生成上述 `t_hist` 与 `t_fut`，供：

- 毕设 **`forecast_curves_temperature_overlay`**；
- **`run_revin_rms_ablation_suite`** 中的叠图（此前同样存在全长 `hist` 与错误 `t_fut` 的组合）；
- 以及 **`forecast_example.png`** 中分割历史/未来的 **竖线位置**（改为 **`ehl - 0.5`**，与 `t_hist` 末端对齐）。

### 3.2 未改动的部分

- **`run_ms_rms_ablation_suite`** 中若 **故意只取 `hist[:, :seq_len]`** 做展示，横轴仍为 `0..seq_len-1`，与截取后的张量长度一致，**保持原逻辑**，无需套用 `effective_hist_len` 全长。

### 3.3 指标与训练

- 修复 **仅影响保存的 PNG**，**不改变** `forecast` 张量、**不改变** MAE/MSE/CRPS 等指标计算；
- **`plot_forecast_compare` 中的 `anchor_forecast_boundary`**（为美观对预测做竖直平移）仍为仅影响绘图的选项，与本次横轴 bug 无关。

---

## 4. 小结（向老师口头可压缩版）

| 项目 | 说明 |
|------|------|
| **问题** | 多尺度历史长度大于 `seq_len` 时，未来段横轴仍从 `seq_len` 起算，导致真值/预测与历史在 x 轴上重叠，辅助连线在 x 方向折返，图上像「ground truth 扭到左边」。 |
| **性质** | **可视化坐标错误**，非模型或标签错误。 |
| **修复** | 未来段横轴起点改为 **`effective_hist_len`**，使 `t_fut` 与 `t_hist` 无缝衔接、无重叠。 |

---

## 5. 复现修复效果

使用当前仓库中的 `main.py` 与 `compare_viz.py`，重新生成 overlay 图（例如仅评估模式重画，无需重训）：

```bash
cd /path/to/Simdiff_weather
python main.py --data_path data/ETTh1.csv --figures_dir ETTh1 \
  --ckpt_extra_suffix <与你权重一致的后缀> --eval_only
```

（若需基线重训可去掉 `--eval_only`；以实际参数为准。）

新生成的 **`forecast_curves_temperature_overlay_*.png`** 中，**ground truth** 应从 **history 右端** 连续向右延伸，不再出现上述扭折。

---

*文档版本：与仓库修复提交对应；日期以 `DEVELOPMENT_LOG.md` 为准。*
