# 不同预测长度下各模型预测精度对比（MSE/MAE）


| 数据集 | 预测长度 | DLinear | iTransformer | mr-Diff | SimDiff | 改进版 SimDiff |
|--------|----------|---------|----------------|---------|---------|----------------|
| ETTh1(OT) | 48 | 5.566730 / 1.792192 | 5.750392 / 1.875543 |  |  | 2.391751 / 1.169282 |
| ETTh1(OT) | 72 |  |  |  |  |  |
| ETTh1(OT) | 168 |  |  |  |  |  |
| ETTh1(OT) | 192 |  |  |  |  |  |
| ETTm1(OT) | 48 |  |  |  |  |  |
| ETTm1(OT) | 72 |  |  |  |  |  |
| ETTm1(OT) | 168 |  |  |  |  |  |
| ETTm1(OT) | 192 |  |  |  |  |  |
| Exchange(USD) | 48 |  |  |  |  |  |
| Exchange(USD) | 72 |  |  |  |  |  |
| Exchange(USD) | 168 |  |  |  |  |  |
| Exchange(USD) | 192 |  |  |  |  |  |
| Weather(Temp) | 48 | 3.063846 / 1.214358 | 3.293931 / 1.302845 |  | 0.742116 / 0.572760 |  |
| Weather(Temp) | 72 |  |  |  |  |  |
| Weather(Temp) | 168 |  |  |  |  |  |
| Weather(Temp) | 192 |  |  |  |  |  |
| Wind(Speed) | 48 |  |  |  |  |  |
| Wind(Speed) | 72 |  |  |  |  |  |
| Wind(Speed) | 168 |  |  |  |  |  |
| Wind(Speed) | 192 |  |  |  |  |  |

<!-- 未齐文件: 缺少 etth1_p168_mrdiff.json, 缺少 etth1_p168_ours.json, 缺少 etth1_p168_simdiff.json, 缺少 etth1_p192_mrdiff.json, 缺少 etth1_p192_ours.json, 缺少 etth1_p192_simdiff.json, 缺少 etth1_p48_mrdiff.json, 缺少 etth1_p48_simdiff.json, 缺少 etth1_p72_mrdiff.json, 缺少 etth1_p72_ours.json, 缺少 etth1_p72_simdiff.json, 缺少 ettm1_p168_mrdiff.json, 缺少 ettm1_p168_ours.json, 缺少 ettm1_p168_simdiff.json, 缺少 ettm1_p192_mrdiff.json, 缺少 ettm1_p192_ours.json, 缺少 ettm1_p192_simdiff.json, 缺少 ettm1_p48_mrdiff.json, 缺少 ettm1_p48_ours.json, 缺少 ettm1_p48_simdiff.json ... -->
