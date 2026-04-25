#!/usr/bin/env bash
# 主对比子集：按需取消注释；结果写入 outputs/metrics，供 generate_paper_artifacts 合并。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MET="${METRICS_DIR:-outputs/metrics}"
mkdir -p "$MET"

PRED_HORIZONS="${PRED_HORIZONS:-48 72 168 192}"
PRESETS="${PRESETS:-weather etth1 ettm1 exchange wind}"

for p in $PRESETS; do
  for h in $PRED_HORIZONS; do
    echo "=== $p pred_len=$h | SimDiff(基线) ==="
    python main.py --data_preset "$p" --pred_len "$h" --skip_lstm --save_run_metrics_dir "$MET"
  done
done

# 改进版 Patch+RoPE
# for p in $PRESETS; do
#   for h in $PRED_HORIZONS; do
#     python main.py --data_preset "$p" --pred_len "$h" --use_patch --use_rope --skip_lstm --save_run_metrics_dir "$MET"
#   done
# done

# mr-Diff
# for p in $PRESETS; do
#   for h in $PRED_HORIZONS; do
#     python main.py --data_preset "$p" --pred_len "$h" --mrdiff --skip_lstm --save_run_metrics_dir "$MET"
#   done
# done

echo "已写入 $MET ；合并表/图: python generate_paper_artifacts.py --metrics_dir $MET"
