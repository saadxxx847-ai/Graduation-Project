论文输出目录说明（本文件由 paper_output 自动生成/更新）

01_fitting_curves/pred_len_<预测步长>/
  各数据集：真实值 vs 改进版预测（每数据集 1 张 PNG）

02_horizon_by_predlen/
  多预测步长（48/72/168/192）下改进版 MAE、MSE 柱状图

03_ablation_module/
  ETTh1 上模块消融柱图（文件名含 pred_len，多组互不影响）

04_tables/
  table1 综合对比表；table2 消融表（含 _p<步长> 后缀防覆盖）
