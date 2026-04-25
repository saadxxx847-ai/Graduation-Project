#!/usr/bin/env bash
# 一次性按论文需求跑完「总表」所需的 5×4×3 次 main（每格需 simdiff / ours / mrdiff 各 1 次）。
# 非常耗时；仅评估时若已有 checkpoints 可把下面 python 改为加 --eval_only。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MET="outputs/metrics"
mkdir -p "$MET"
PRESETS="weather etth1 ettm1 exchange wind"
HORIZONS="48 72 168 192"

for p in $PRESETS; do
  for h in $HORIZONS; do
    echo "=== $p pred_len=$h  (1/3 SimDiff) ==="
    python main.py --data_preset "$p" --pred_len "$h" --skip_lstm \
      --save_run_metrics_dir "$MET"
    echo "=== $p pred_len=$h  (2/3 改进：Patch+RoPE) ==="
    python main.py --data_preset "$p" --pred_len "$h" --use_patch --use_rope --skip_lstm \
      --save_run_metrics_dir "$MET"
    echo "=== $p pred_len=$h  (3/3 mr-Diff) ==="
    python main.py --data_preset "$p" --pred_len "$h" --mrdiff --skip_lstm \
      --save_run_metrics_dir "$MET"
  done
done

echo "=== 消融 ETTh1 pred_len=168 (4 组，跳过统计基线以省时间) ==="
AB=(--data_preset etth1 --pred_len 168 --skip_baselines --skip_lstm --save_run_metrics_dir "$MET")
python main.py "${AB[@]}"
python main.py --use_patch "${AB[@]}"
python main.py --use_rope "${AB[@]}"
python main.py --use_patch --use_rope "${AB[@]}"

echo "完成。下一步: python generate_paper_artifacts.py --metrics_dir $MET --out_dir outputs/paper"
echo "  或: python -m utils.paper_output all --metrics_dir $MET --out_dir outputs/paper"
