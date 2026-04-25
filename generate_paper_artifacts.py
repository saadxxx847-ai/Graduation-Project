#!/usr/bin/env python3
"""
只生成论文要求的图与表（不跑 main 训练）：
  第一类：5 张拟合曲线（fig1_fitting_curves/）
  第二类：多步长 MAE/MSE 柱状图（fig2_*）
  第三类：消融柱图（fig3_*）
  表1：综合对比  表2：ETTh1 消融

依赖：outputs/metrics/*.json 已齐全；checkpoints 有各 simdiff_*_p{curve_pred_len}_ours.pt

输出分目录见 outputs/paper/ 下 01～04 与 00_README_目录说明.txt。

可选：--clean_diagnostics 清空 plots/diagnostics/ 下旧调试 PNG。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="生成论文图+表到 outputs/paper")
    ap.add_argument(
        "--metrics_dir",
        type=str,
        default="outputs/metrics",
        help="save_run_metrics_dir 写出的 JSON 目录",
    )
    ap.add_argument("--out_dir", type=str, default="outputs/paper")
    ap.add_argument("--ablation_pred_len", type=int, default=168)
    ap.add_argument("--curve_pred_len", type=int, default=168)
    ap.add_argument(
        "--clean_diagnostics",
        action="store_true",
        help="删除 plots/diagnostics/ 下旧 PNG；默认不删，避免误伤分类存放的图",
    )
    args = ap.parse_args()

    if args.clean_diagnostics:
        for sub in ("plots/diagnostics", "plots/legacy"):
            pd = ROOT / sub
            if pd.is_dir():
                n = 0
                for f in pd.glob("*.png"):
                    f.unlink()
                    n += 1
                if n:
                    print(f"已删除 {sub}/ 下 {n} 个 PNG")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from utils import paper_output as po

    ns = argparse.Namespace(
        metrics_dir=args.metrics_dir,
        out_dir=str(out),
        ablation_pred_len=int(args.ablation_pred_len),
    )
    rc = po._cmd_merge(ns)
    if rc != 0:
        return rc
    ns2 = argparse.Namespace(
        metrics_dir=args.metrics_dir,
        out_dir=str(out),
        curve_pred_len=int(args.curve_pred_len),
        ablation_pred_len=int(args.ablation_pred_len),
    )
    return po._cmd_plots(ns2)


if __name__ == "__main__":
    raise SystemExit(main())
