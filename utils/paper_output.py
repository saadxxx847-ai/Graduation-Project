"""
从 save_run_metrics_dir 产出的多份 JSON 合并论文用总表、曲线图与柱状图。

前置：对每种 (data_preset, pred_len, 模型) 已训练/评估并写 JSON 到 --save_run_metrics_dir。

用法:
  python -m utils.paper_output merge  --metrics_dir outputs/metrics
  python -m utils.paper_output plots  --metrics_dir outputs/metrics --out_dir outputs/paper
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 与毕业论文表一致（不含年份）
COL_ORDER = [
    "DLinear",
    "iTransformer",
    "mr-Diff",
    "SimDiff",
    "改进版 SimDiff",
]
PRESET_ORDER = ["etth1", "ettm1", "exchange", "weather", "wind"]
PRESET_TABLE_LABEL = {
    "etth1": "ETTh1(OT)",
    "ettm1": "ETTm1(OT)",
    "exchange": "Exchange(USD)",
    "weather": "Weather(Temp)",
    "wind": "Wind(Speed)",
}
HORIZONS = [48, 72, 168, 192]

# 在 out_dir 下分类落盘，避免全堆一个目录、不同 pred_len 互相覆盖
DIR_FIT = "01_fitting_curves"
DIR_HORIZON = "02_horizon_by_predlen"
DIR_ABLATION = "03_ablation_module"
DIR_TABLES = "04_tables"


def paper_subdirs(out_root: Path) -> dict[str, Path]:
    d = {
        "fit": out_root / DIR_FIT,
        "horizon": out_root / DIR_HORIZON,
        "ablation": out_root / DIR_ABLATION,
        "tables": out_root / DIR_TABLES,
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    layout = (
        "论文输出目录说明（本文件由 paper_output 自动生成/更新）\n\n"
        f"{DIR_FIT}/pred_len_<预测步长>/\n"
        "  各数据集：真实值 vs 改进版预测（每数据集 1 张 PNG）\n\n"
        f"{DIR_HORIZON}/\n"
        "  多预测步长（48/72/168/192）下改进版 MAE、MSE 柱状图\n\n"
        f"{DIR_ABLATION}/\n"
        "  ETTh1 上模块消融：MAE / MSE 柱图（文件名含 pred_len，多组互不影响）\n\n"
        f"{DIR_TABLES}/\n"
        "  table1 综合对比表；table2 消融表（含 _p<步长> 后缀防覆盖）\n"
    )
    (out_root / "00_README_目录说明.txt").write_text(layout, encoding="utf-8")
    return d


def _load_jsons(metrics_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(metrics_dir.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return rows


def _cell(mse: float | None, mae: float | None) -> str:
    if mse is None or mae is None or not np.isfinite(mse) or not np.isfinite(mae):
        return ""
    return f"{float(mse):.6f} / {float(mae):.6f}"


def _index_by_preset_horizon(
    jrows: list[dict],
) -> dict[tuple[str, int], dict[str, dict]]:
    """
    键: (data_preset, pred_len)
    值: {'simdiff': record, 'ours': ..., 'mrdiff': ...} 以及每文件自带的 DLINEAR
    """
    by_ph: dict[tuple[str, int], dict[str, dict]] = {}
    for r in jrows:
        p = str(r.get("data_preset", "")).lower()
        h = int(r.get("pred_len", 0))
        tag = str(r.get("checkpoint_tag", ""))
        key = (p, h)
        if key not in by_ph:
            by_ph[key] = {}
        by_ph[key][tag] = r
    return by_ph


def _dl_it_from_any(
    d: dict[str, dict] | None,
) -> tuple[tuple[float | None, float | None], tuple[float | None, float | None]]:
    if not d:
        return (None, None), (None, None)
    for tag in ("simdiff", "ours", "patch", "rope", "mrdiff", "mom_only"):
        t = d.get(tag)
        if t is not None and t.get("DLinear_mse") is not None:
            dlm = (float(t["DLinear_mse"]), float(t["DLinear_mae"]))
            itm: tuple[float | None, float | None] = (None, None)
            if t.get("iTransformer_mse") is not None and t.get("iTransformer_mae") is not None:
                itm = (float(t["iTransformer_mse"]), float(t["iTransformer_mae"]))
            return dlm, itm
    return (None, None), (None, None)


def merge_table(metrics_dir: Path) -> tuple[str, str, str]:
    """返回 (md, csv, 诊断说明)。"""
    jrows = _load_jsons(metrics_dir)
    if not jrows:
        return "", "", f"未在 {metrics_dir} 下找到 .json 指标文件。请先多 run main.py 并加 --save_run_metrics_dir 该目录。"

    by = _index_by_preset_horizon(jrows)
    lines_csv = [
        "数据集,预测长度,DLinear,iTransformer,mr-Diff,SimDiff,改进版 SimDiff"
    ]
    lines_md: list[str] = [
        "# 不同预测长度下各模型预测精度对比（MSE/MAE）\n",
        "\n| 数据集 | 预测长度 | DLinear | iTransformer | mr-Diff | SimDiff | 改进版 SimDiff |",
        "|--------|----------|---------|--------------|---------|---------|----------------|",
    ]
    missing: list[str] = []

    for p in PRESET_ORDER:
        for h in HORIZONS:
            key = (p, h)
            cell = {name: _cell(None, None) for name in COL_ORDER}
            b = by.get(key, {})
            dlm, itm = _dl_it_from_any(b)

            # SimDiff
            srec = b.get("simdiff")
            if srec is not None:
                cell["SimDiff"] = _cell(
                    float(srec.get("diffusion_mse", np.nan)),
                    float(srec.get("diffusion_mae", np.nan)),
                )
            else:
                missing.append(f"缺少 {p}_p{h}_simdiff.json")

            # 改进
            orec = b.get("ours")
            if orec is not None:
                cell["改进版 SimDiff"] = _cell(
                    float(orec.get("diffusion_mse", np.nan)),
                    float(orec.get("diffusion_mae", np.nan)),
                )
            else:
                missing.append(f"缺少 {p}_p{h}_ours.json")

            # mr
            mrec = b.get("mrdiff")
            if mrec is not None:
                cell["mr-Diff"] = _cell(
                    float(mrec.get("diffusion_mse", np.nan)),
                    float(mrec.get("diffusion_mae", np.nan)),
                )
            else:
                missing.append(f"缺少 {p}_p{h}_mrdiff.json")

            if dlm[0] is not None:
                cell["DLinear"] = _cell(dlm[0], dlm[1])
            if itm[0] is not None:
                cell["iTransformer"] = _cell(itm[0], itm[1])

            rowname = PRESET_TABLE_LABEL.get(p, p)
            row_csv = ",".join(
                [rowname, str(h), cell["DLinear"], cell["iTransformer"], cell["mr-Diff"], cell["SimDiff"], cell["改进版 SimDiff"]]
            )
            lines_csv.append(row_csv)
            line_md = (
                f"| {rowname} | {h} | {cell['DLinear']} | {cell['iTransformer']} | {cell['mr-Diff']} | {cell['SimDiff']} | {cell['改进版 SimDiff']} |"
            )
            lines_md.append(line_md)

    md = "\n".join(lines_md) + "\n"
    if missing:
        md += f"\n<!-- 未齐文件: {', '.join(sorted(set(missing))[:20])} ... -->\n"
    return md, "\n".join(lines_csv) + "\n", "\n".join(sorted(set(missing)))


def merge_ablation_table(
    metrics_dir: Path, pred_len: int = 168
) -> tuple[str, str]:
    """表格2：ETTh1(OT) 消融，4 行 MSE/MAE。"""
    tags = [
        ("simdiff", "Baseline：原始 SimDiff（NI + MoM）"),
        ("patch", "+ Patch embedding（分块嵌入）"),
        ("rope", "+ RoPE（旋转位置编码）"),
        ("ours", "Your model：SimDiff + Patch + RoPE（最终模型）"),
    ]
    lines_md = [
        "# 消融实验：ETTh1(OT) 预测精度（MSE/MAE）\n",
        f"\n| 设置 | pred_len={pred_len} (MSE/MAE) |",
        "|------|----------------|",
    ]
    lines_csv = ["设置,MSE/MAE"]
    for tag, label in tags:
        r = _find_record(metrics_dir, "etth1", pred_len, tag)
        cell = _cell(
            float(r["diffusion_mse"]) if r and r.get("diffusion_mse") is not None else None,
            float(r["diffusion_mae"]) if r and r.get("diffusion_mae") is not None else None,
        )
        lines_md.append(f"| {label} | {cell} |")
        lines_csv.append(f"{label},{cell}")
    return "\n".join(lines_md) + "\n", "\n".join(lines_csv) + "\n"


def _find_record(metrics_dir: Path, preset: str, h: int, tag: str) -> dict | None:
    p = metrics_dir / f"{preset}_p{h}_{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def plot_horizon_bars(
    metrics_dir: Path, out_path: Path, out_path_mse: Path | None = None
) -> None:
    """
    第二类等：多预测步长 **柱状图**（48/72/168/192）。
    每个步长 5 根柱 = 5 个数据集在「改进版」下的 MAE；另存一张 MSE 柱图（可选同文件第二子图）。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ds = len(PRESET_ORDER)
    n_h = len(HORIZONS)
    w = 0.12
    x0 = np.arange(n_h) * 1.35

    def _one_metric(key: str, ylabel: str, title: str, save_p: Path) -> None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for i, p in enumerate(PRESET_ORDER):
            vals: list[float] = []
            for h in HORIZONS:
                r = _find_record(metrics_dir, p, h, "ours")
                if r is None or r.get(key) is None:
                    vals.append(float("nan"))
                else:
                    vals.append(float(r[key]))
            x = x0 + (i - (n_ds - 1) / 2) * w
            ax.bar(
                x,
                vals,
                width=w * 0.95,
                label=PRESET_TABLE_LABEL.get(p, p),
            )
        ax.set_xticks(x0)
        ax.set_xticklabels([str(h) for h in HORIZONS])
        ax.set_xlabel("预测步长 (pred_len)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_p, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _one_metric(
        "diffusion_mae",
        "MAE (测试集，目标列)",
        "多预测步长下 Improved SimDiff MAE 对比",
        out_path,
    )
    p_mse = out_path_mse or out_path.parent / (out_path.stem + "_mse" + out_path.suffix)
    _one_metric(
        "diffusion_mse",
        "MSE (测试集，目标列)",
        "多预测步长下 Improved SimDiff MSE 对比",
        p_mse,
    )


def _ablation_tags() -> list[tuple[str, str]]:
    return [
        ("simdiff", "Baseline\n(NI+MoM)"),
        ("patch", "+Patch\nembedding"),
        ("rope", "+RoPE"),
        ("ours", "Patch+RoPE\n(Ours)"),
    ]


def plot_ablation_bars(
    metrics_dir: Path, out_path: Path, pred_len: int = 168
) -> str:
    """
    消融：etth1、pred_len 固定，比较 simdiff / patch / rope / ours 的 MAE 柱状图。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tags = _ablation_tags()
    names: list[str] = []
    maes: list[float] = []
    for tag, name in tags:
        r = _find_record(metrics_dir, "etth1", pred_len, tag)
        names.append(name)
        if r is not None and r.get("diffusion_mae") is not None:
            maes.append(float(r["diffusion_mae"]))
        else:
            maes.append(float("nan"))
    w = 0.6
    x = np.arange(len(names))
    plt.figure(figsize=(9, 4.2))
    plt.bar(x, maes, width=w, color="steelblue", alpha=0.9)
    plt.xticks(x, names, rotation=0, ha="center", fontsize=9)
    plt.ylabel("MAE")
    plt.title(f"消融实验 (ETTh1(OT), pred_len={pred_len}) — MAE")
    plt.tight_layout()
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(out_path, dpi=150)
    plt.close()
    miss = [t for t, m in zip(["simdiff", "patch", "rope", "ours"], maes) if not np.isfinite(m)]
    if miss:
        return f"未找到: " + ", ".join(f"etth1_p{pred_len}_{m}.json" for m in miss)
    return ""


def plot_ablation_bars_mse(
    metrics_dir: Path, out_path: Path, pred_len: int = 168
) -> str:
    """同消融 4 组，指标换为 MSE。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tags = _ablation_tags()
    names: list[str] = []
    mses: list[float] = []
    for tag, name in tags:
        r = _find_record(metrics_dir, "etth1", pred_len, tag)
        names.append(name)
        if r is not None and r.get("diffusion_mse") is not None:
            mses.append(float(r["diffusion_mse"]))
        else:
            mses.append(float("nan"))
    w = 0.6
    x = np.arange(len(names))
    plt.figure(figsize=(9, 4.2))
    plt.bar(x, mses, width=w, color="coral", alpha=0.9)
    plt.xticks(x, names, rotation=0, ha="center", fontsize=9)
    plt.ylabel("MSE")
    plt.title(f"消融实验 (ETTh1(OT), pred_len={pred_len}) — MSE")
    plt.tight_layout()
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(out_path, dpi=150)
    plt.close()
    miss = [t for t, m in zip(["simdiff", "patch", "rope", "ours"], mses) if not np.isfinite(m)]
    if miss:
        return f"未找到: " + ", ".join(f"etth1_p{pred_len}_{m}.json" for m in miss)
    return ""


@torch.no_grad()
def plot_five_ours_curves(
    out_dir: Path, pred_len: int = 168, device: str | None = None
) -> str:
    """
    5 个数据集上：真实 vs 改进版预测（单条测试窗）。需 checkpoints/simdiff_*_p{pred_len}_ours.pt 存在。
    """
    from config.config import Config
    from models.simdiff import SimDiffWeather, point_prediction_from_forecast
    from utils.data_loader import make_loaders

    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    hints: list[str] = []
    t_hist = None

    for p in PRESET_ORDER:
        cfg = Config()
        cfg.data_preset = p
        cfg.pred_len = pred_len
        cfg.univariate = True
        cfg.use_patch = True
        cfg.use_rope = True
        cfg.mrdiff_denoiser = False
        tr, _va, te, nf, _fn = make_loaders(cfg)
        ck = cfg.resolved_checkpoint_dir() / cfg.simdiff_checkpoint_filename()
        if not ck.is_file():
            hints.append(f"缺权重: {ck}（需先训 --use_patch --use_rope）")
            continue
        st = torch.load(ck, map_location=dev, weights_only=False)
        meta = st.get("meta") or {}
        if "timesteps" in meta:
            cfg.timesteps = int(meta["timesteps"])
        if meta.get("sampling_steps") is not None:
            cfg.sampling_steps = int(meta["sampling_steps"])
        m = SimDiffWeather(cfg).to(dev)
        m.load_state_dict(st["model"], strict=True)
        m.eval()
        hist, fut = next(iter(te))
        hist = hist[:1].to(dev)
        fut = fut[:1].to(dev)
        with torch.no_grad():
            o = m.forecast(hist, future=fut)
        pred = point_prediction_from_forecast(o, m.cfg).cpu().numpy()[0, :, 0]
        tru = fut.cpu().numpy()[0, :, 0]
        ho = np.arange(cfg.seq_len)
        fidx = np.arange(cfg.seq_len, cfg.seq_len + pred_len)
        t_hist = ho
        plt.figure(figsize=(9, 3.2))
        plt.plot(ho, hist.cpu().numpy()[0, :, 0], "C0", label="历史", linewidth=1.2)
        plt.plot(fidx, tru, "C1", label="真实", linewidth=2)
        plt.plot(fidx, pred, "C2--", label="Improved SimDiff", linewidth=2)
        plt.axvline(cfg.seq_len - 0.5, color="gray", linestyle=":", linewidth=1)
        plt.xlabel("时间步 (窗口内下标)")
        plt.ylabel("目标量")
        plt.title(
            f"{PRESET_TABLE_LABEL.get(p, p)}  —  真实 vs Improved SimDiff (pred_len={pred_len})"
        )
        plt.legend()
        plt.tight_layout()
        # out_dir 建议为 01_fitting_curves/pred_len_{pred}，文件名不再重复 p{len}
        fp = out_dir / f"{p}_true_vs_ours.png"
        plt.savefig(fp, dpi=150)
        plt.close()
    return "\n".join(hints) if hints else ""


def _cmd_merge(args: argparse.Namespace) -> int:
    md, csv, miss = merge_table(Path(args.metrics_dir))
    if not md:
        print(miss, file=sys.stderr)
        return 1
    outd = Path(args.out_dir) if args.out_dir else Path(args.metrics_dir)
    tdir = paper_subdirs(outd)["tables"]
    (tdir / "table1_master_comparison.md").write_text(md, encoding="utf-8")
    (tdir / "table1_master_comparison.csv").write_text(csv, encoding="utf-8")
    pl = int(getattr(args, "ablation_pred_len", 168))
    md2, cs2 = merge_ablation_table(Path(args.metrics_dir), pred_len=pl)
    t2b = f"table2_ablation_ETTh1_p{pl}"
    (tdir / f"{t2b}.md").write_text(md2, encoding="utf-8")
    (tdir / f"{t2b}.csv").write_text(cs2, encoding="utf-8")
    print("已写:", tdir)
    if miss:
        print("缺文件提示 (合并处留空):", file=sys.stderr)
        print(miss, file=sys.stderr)
    return 0


def _cmd_plots(args: argparse.Namespace) -> int:
    md = Path(args.metrics_dir)
    od = Path(args.out_dir)
    pred_len = int(args.curve_pred_len)
    sub = paper_subdirs(od)
    apl = int(args.ablation_pred_len)
    # 文件名带 pred 范围与曲线 pred，避免多跑相互覆盖
    htag = f"H{'-'.join(str(h) for h in HORIZONS)}_curveP{pred_len}"
    plot_horizon_bars(
        md,
        sub["horizon"] / f"improved_mae_bars__{htag}.png",
        out_path_mse=sub["horizon"] / f"improved_mse_bars__{htag}.png",
    )
    print("已写:", sub["horizon"])
    ab = plot_ablation_bars(
        md,
        sub["ablation"] / f"etth1_mae_ablation_p{apl}.png",
        pred_len=apl,
    )
    if ab:
        print(ab, file=sys.stderr)
    else:
        print("已写:", sub["ablation"] / f"etth1_mae_ablation_p{apl}.png")
    ab2 = plot_ablation_bars_mse(
        md,
        sub["ablation"] / f"etth1_mse_ablation_p{apl}.png",
        pred_len=apl,
    )
    if ab2:
        print(ab2, file=sys.stderr)
    else:
        print("已写:", sub["ablation"] / f"etth1_mse_ablation_p{apl}.png")
    fit_dir = sub["fit"] / f"pred_len_{pred_len}"
    fit_dir.mkdir(parents=True, exist_ok=True)
    cmsg = plot_five_ours_curves(fit_dir, pred_len=pred_len)
    if cmsg:
        print(cmsg, file=sys.stderr)
    else:
        print("已写: 拟合曲线 ->", fit_dir)
    print(
        "目录结构: 01_fitting_curves/pred_len_*/  02_horizon_by_predlen/  "
        "03_ablation_module/  04_tables/"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="论文：合并表与出图")
    sp = ap.add_subparsers(dest="cmd", required=True)

    m = sp.add_parser("merge", help="合并为 table1/2 的 .md 与 .csv")
    m.add_argument("--metrics_dir", type=str, required=True)
    m.add_argument("--out_dir", type=str, default=None, help="默认与 metrics_dir 相同")
    m.add_argument(
        "--ablation_pred_len",
        type=int,
        default=168,
        help="表格2/消融图使用的 etth1 预测步长",
    )
    m.set_defaults(func=_cmd_merge)

    p = sp.add_parser("plots", help="第1类5张拟合+第2类柱图+第3类消融；需 checkpt 与 JSON")
    p.add_argument("--metrics_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="outputs/paper")
    p.add_argument(
        "--curve_pred_len", type=int, default=168, help="5 条拟合曲线与 ours 权重的 p 下标"
    )
    p.add_argument(
        "--ablation_pred_len",
        type=int,
        default=168,
        help="消融柱图/表2 的 pred_len",
    )
    p.set_defaults(func=_cmd_plots)

    a = sp.add_parser("all", help="merge + plots 一步完成")
    a.add_argument("--metrics_dir", type=str, default="outputs/metrics")
    a.add_argument("--out_dir", type=str, default="outputs/paper")
    a.add_argument("--ablation_pred_len", type=int, default=168)
    a.add_argument("--curve_pred_len", type=int, default=168)

    def _cmd_all(args: argparse.Namespace) -> int:
        c = argparse.Namespace(
            metrics_dir=args.metrics_dir,
            out_dir=args.out_dir,
            ablation_pred_len=int(args.ablation_pred_len),
        )
        c2 = _cmd_merge(c)
        if c2 != 0:
            return c2
        c3 = argparse.Namespace(
            metrics_dir=args.metrics_dir,
            out_dir=args.out_dir,
            curve_pred_len=int(args.curve_pred_len),
            ablation_pred_len=int(args.ablation_pred_len),
        )
        return _cmd_plots(c3)

    a.set_defaults(func=_cmd_all)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
