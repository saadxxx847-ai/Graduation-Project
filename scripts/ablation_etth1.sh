#!/usr/bin/env bash
# 消融仅 ETTh1(OT)、pred_len=168；四组分别训练，ckpt 名 simdiff / patch / rope / ours
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
H=168
MET="${METRICS_DIR:-outputs/metrics}"
mkdir -p "$MET"
OPTS=(--data_preset etth1 --pred_len "$H" --skip_baselines --skip_lstm --save_run_metrics_dir "$MET")

echo "1) Baseline: NI+MoM SimDiff"
python main.py "${OPTS[@]}"

echo "2) +Patch embedding"
python main.py --use_patch "${OPTS[@]}"

echo "3) +RoPE"
python main.py --use_rope "${OPTS[@]}"

echo "4) Ours: Patch+RoPE"
python main.py --use_patch --use_rope "${OPTS[@]}"

echo "指标目录: $MET ；出图/表: python generate_paper_artifacts.py --metrics_dir $MET"
