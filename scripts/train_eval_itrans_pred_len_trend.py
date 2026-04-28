#!/usr/bin/env python3
"""
对多个 pred_len 分别训练 BaselineiTransformer（seq_len=96、单变量气温），
每个长度在测试集上评完后打印与 main 消融段类似的简易终端表（便于截图），默认不保存 png。

说明：
- 不训练 SimDiff；不修改 main.py。
- 点预测：CRPS 列与 MAE 同值（退化）、VAR=0，与 print_thesis_metrics_table 脚注约定一致。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import Config
from main import resolve_temperature_feature_index
from utils.baselines import BaselineiTransformer, eval_channel_mse_mae, fit_regression_model
from utils.data_loader import make_loaders
from utils.result_output import print_metrics_ascii_table


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_pred_lens(s: str) -> list[int]:
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return [max(1, int(x)) for x in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="iTransformer：按 pred_len 逐个训练+测试，每长度打印终端指标表（默认不画图）"
    )
    parser.add_argument("--pred_lens", type=str, default="48,72,168,192")
    parser.add_argument("--epochs", type=int, default=20, help="每长度最大训练轮数（验证早停）")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="额外保存双轴趋势图（需 matplotlib）",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default=None,
        help="与 --plot 合用；默认 length/mae_mse_vs_pred_len_itransformer.png",
    )
    parser.add_argument(
        "--save_ckpt",
        action="store_true",
        help="将每长度最优权重存 checkpoints/itransformer_weather_best_pl{H}.pt",
    )
    args = parser.parse_args()

    pred_lens = _parse_pred_lens(args.pred_lens)
    cfg = Config()
    cfg.use_multiscale_hist = False
    cfg.hist_window_start_min = 0
    cfg.validate_mom_config()
    cfg.validate_training_noise_objective()
    cfg.validate_simdiff_ablation()
    cfg.validate_denoiser_embedding_options()

    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")

    summary: list[tuple[int, float, float]] = []
    ckpt_dir = cfg.resolved_checkpoint_dir()

    for pl in pred_lens:
        cfg.pred_len = pl
        cfg.validate_mom_config()
        train_loader, val_loader, test_loader, n_features, feat_names = make_loaders(cfg)
        t_idx = resolve_temperature_feature_index(feat_names)
        _tn_raw = feat_names[t_idx] if t_idx < len(feat_names) else str(t_idx)
        temp_name = " ".join(str(_tn_raw).split())
        slug = cfg.result_dataset_slug()

        sl, C = cfg.seq_len, n_features
        model = BaselineiTransformer(
            sl,
            pl,
            C,
            d_model=cfg.baseline_itransformer_d_model,
            nhead=cfg.baseline_itransformer_nhead,
            num_layers=cfg.baseline_itransformer_layers,
            dropout=cfg.dropout,
        )
        model = fit_regression_model(
            model,
            train_loader,
            val_loader,
            device,
            max_epochs=max(1, int(args.epochs)),
            lr=cfg.baseline_lr,
            patience=cfg.baseline_early_stop_patience,
            grad_clip_max_norm=cfg.baseline_grad_clip_max_norm,
            name="iTransformer",
        )

        def predict(h: torch.Tensor) -> torch.Tensor:
            return model(h)

        mse_ch, mae_ch = eval_channel_mse_mae(predict, test_loader, device, t_idx)
        mae_f = float(mae_ch)
        mse_f = float(mse_ch)
        summary.append((pl, mae_f, mse_f))

        print_metrics_ascii_table(
            [
                (
                    "iTransformer",
                    mae_f,
                    mse_f,
                    f"{mae_f:.6f}",
                    f"{0.0:.6f}",
                )
            ],
            headline=f"{slug} · {temp_name} · iTransformer (pred_len={pl}, 1-fold)",
            footer_notes=(
                "点预测：CRPS 与 MAE 同列（退化）；VAR = 0。",
            ),
        )

        if args.save_ckpt:
            ckpt_path = ckpt_dir / f"itransformer_weather_best_pl{pl}.pt"
            torch.save({"model": model.state_dict(), "meta": {"pred_len": pl, "seq_len": sl}}, ckpt_path)
            print(f"[saved ckpt] {ckpt_path}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("--- iTransformer pred_len sweep 汇总（test，气温通道）---")
    print(f"{'pred_len':>8} {'MAE':>14} {'MSE':>14}")
    print("-" * 40)
    for pl, ma, ms in summary:
        print(f"{pl:>8} {ma:>14.6f} {ms:>14.6f}")
    print()

    if args.plot:
        from utils.compare_viz import plot_pred_len_accuracy_trend

        xs = [r[0] for r in summary]
        maes = [r[1] for r in summary]
        mses = [r[2] for r in summary]
        slug = cfg.result_dataset_slug()
        if args.out_png:
            out = Path(args.out_png)
            if not out.is_absolute():
                out = ROOT / out
        else:
            out = ROOT / "length" / "mae_mse_vs_pred_len_itransformer.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_pred_len_accuracy_trend(
            out,
            xs,
            maes,
            mses,
            curve_label="iTransformer",
            title=f"[{slug}] Test MAE / MSE vs prediction length (iTransformer, seq_len={cfg.seq_len})",
        )
        print(f"[saved plot] {out}")


if __name__ == "__main__":
    main()
