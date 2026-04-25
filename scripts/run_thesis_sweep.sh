#!/usr/bin/env bash
# 毕设主实验：5 数据集 × 4 预测步长 ×（SimDiff / mr-Diff / 改进版 SimDiff），
# 并写出 outputs/metrics/*.json 供合并表与论文图。
# 再加 ETTh1 消融 4 组（pred_len=168）。建议 GPU 上串行执行；全程约需数小时～数天。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MET="${METRICS_DIR:-outputs/metrics}"
mkdir -p "$MET" outputs/paper

HORIZONS=(48 72 168 192)
PRESETS=(weather etth1 ettm1 exchange wind)

train_one() {
  local preset="$1" pl="$2" mode="$3"
  case "$mode" in
    simdiff)
      python main.py --data_preset "$preset" --pred_len "$pl" --skip_lstm \
        --save_run_metrics_dir "$MET" ;;
    mrdiff)
      python main.py --data_preset "$preset" --pred_len "$pl" --mrdiff --skip_lstm \
        --save_run_metrics_dir "$MET" ;;
    ours)
      python main.py --data_preset "$preset" --pred_len "$pl" --use_patch --use_rope --skip_lstm \
        --save_run_metrics_dir "$MET" ;;
    *)
      echo "unknown mode $mode" >&2
      exit 1 ;;
  esac
}

echo "=== 主对比：三种扩散变体 × ${#PRESETS[@]} 数据集 × ${#HORIZONS[@]} 步长 ==="
for pl in "${HORIZONS[@]}"; do
  for p in "${PRESETS[@]}"; do
    echo "--- $p pred_len=$pl simdiff ---"
    train_one "$p" "$pl" simdiff
    echo "--- $p pred_len=$pl mr-Diff ---"
    train_one "$p" "$pl" mrdiff
    echo "--- $p pred_len=$pl 改进版(Patch+RoPE) ---"
    train_one "$p" "$pl" ours
  done
done

echo "=== 消融补训：主循环已生成 etth1 p168 的 simdiff / ours 指标，此处只训 +Patch、+RoPE 两组 =="
H=168
python main.py --data_preset etth1 --pred_len "$H" --use_patch --skip_baselines --skip_lstm \
  --save_run_metrics_dir "$MET"
python main.py --data_preset etth1 --pred_len "$H" --use_rope --skip_baselines --skip_lstm \
  --save_run_metrics_dir "$MET"
echo "（若需单独跑完整四组消融，用: ./scripts/ablation_etth1.sh）"

echo "=== 合并表与图（outputs/paper 下 01～04）==="
python generate_paper_artifacts.py --metrics_dir "$MET" --out_dir outputs/paper

echo "完成。如仅补图不训，可： python generate_paper_artifacts.py --metrics_dir $MET"
echo "五模型拟合曲线（需已训好权重）: python scripts/render_thesis_curves.py --pred_len 168 --out_subdir pred_len_168"
