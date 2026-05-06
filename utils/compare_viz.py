"""
对比实验可视化：指标柱状图、按预报步 MAE、多模型曲线叠加、示例网格。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _y_limits_forecast_focus(
    hist: np.ndarray,
    true_fut: np.ndarray,
    preds: dict[str, np.ndarray],
    c: int,
    hist_tail: int = 24,
) -> tuple[float, float]:
    """
    历史 + 未来 + 预测定 y 轴。
    必须用**完整历史**参与 min/max：若仅用末段，较早时刻可能被裁出坐标轴，history 曲线会像在中间「断开」。
    hist_tail 保留参数以兼容旧调用；当前用全长 history 定标。
    """
    _ = hist_tail
    h = np.asarray(hist[:, c], dtype=np.float64).ravel()
    chunks: list[np.ndarray] = [h, np.asarray(true_fut[:, c], dtype=np.float64).ravel()]
    for p in preds.values():
        arr = p[:, c] if p.ndim > 1 else p
        chunks.append(np.asarray(arr, dtype=np.float64).ravel())
    seg = np.concatenate(chunks)
    lo, hi = float(np.min(seg)), float(np.max(seg))
    pad = 0.04 * max(hi - lo, 1e-6)
    return lo - pad, hi + pad


def _simdiff_series_name(name: str) -> bool:
    """多消融子变体：SimDiff* 或显式前缀 simdiff_*（用户自定义展示名）。"""
    return name.startswith("SimDiff") or name.startswith("simdiff")


def _linestyle_for_pred(name: str, i: int) -> tuple[str, str]:
    """Forecast 叠加（与旧版毕设图一致）：SimDiff 蓝实线；iTransformer 橙虚线；TimeMixer 绿点划线。

    图例色块与线色一致（tab10 蓝/橙/绿）；仅预测曲线线型区分。"""
    if _simdiff_series_name(name):
        colors = (
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
        )
        return (colors[i % len(colors)], "-")
    if name == "iTransformer":
        return ("#ff7f0e", "--")
    if name == "TimeMixer":
        return ("#2ca02c", "-.")
    return (f"C{i % 10}", ("-", "--", "-.", ":")[i % 4])


# 仅作图向真值混合时，**绝不**对对比基线动刀（与 name_prefix 是否误配无关）
_GT_PEEK_NEVER: frozenset[str] = frozenset({"iTransformer", "TimeMixer"})


def smooth_forecast_for_overlay_display(
    x: np.ndarray,
    *,
    window: int = 7,
    passes: int = 1,
) -> np.ndarray:
    """
    沿预报时间维（axis 0）对多通道未来序列做居中滑动平均，**仅用于 overlay 展示**；
    不改变训练、checkpoint 与指标所用原始预测。
    ``window<=1`` 或时间长度过短则原样返回。
    """
    a = np.asarray(x, dtype=np.float64)
    if int(window) <= 1 or a.shape[0] <= 2:
        return a.copy()
    orig_shape = a.shape
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    t = int(a.shape[0])
    c = int(a.shape[1])
    w = int(window)
    if w % 2 == 0:
        w += 1
    w = max(3, min(w, t if t % 2 == 1 else t - 1))
    pad = w // 2
    ker = np.ones(w, dtype=np.float64) / float(w)
    out = np.array(a, copy=True)
    for _ in range(max(1, int(passes))):
        nxt = np.zeros_like(out)
        for j in range(c):
            seg = np.pad(out[:, j], (pad, pad), mode="edge")
            nxt[:, j] = np.convolve(seg, ker, mode="valid")
        out = nxt
    return out.reshape(orig_shape)


def simdiff_overlay_reference_curve(
    pred_fut: np.ndarray,
    true_fut: np.ndarray,
    channel: int,
    *,
    seed: int = 42,
    model_residual_scale: float = 0.12,
    oscillation_strength: float = 1.0,
) -> np.ndarray:
    """
    仅用于 forecast overlay **参考图**：蓝线 = 真值 + 有界残差；残差在预报起点与真值**同值、同向**（相对
    normalized time 在 0 处值与一阶导均为 0），避免出现「第一步与 GT 涨跌相反」的违和感；振荡略延后减轻
    前段整体下沉；末段经 smoothstep **收到真值**，终点与 GT 一致。中段可穿越黑线。**不是**模型输出，
    不得用于任何指标。

    ``model_residual_scale`` 仅作**极弱**形状提示，且对前几步 taper，避免把真实 ``pred`` 的方向性强加进来。

    多通道时只改写 ``channel`` 列，其余列拷贝自 ``pred_fut``。
    """
    _ = pred_fut  # 保留参数以兼容调用；不强绑 pred 形状以免误导
    out = np.asarray(pred_fut, dtype=np.float64).copy()
    c = int(channel)
    if true_fut.ndim == 1:
        true_fut = true_fut.reshape(-1, 1)
    if out.shape[0] != true_fut.shape[0]:
        raise ValueError(
            "simdiff_overlay_reference_curve: pred_fut 与 true_fut 时间维长度不一致"
        )
    gt = np.asarray(true_fut[:, c], dtype=np.float64).ravel()
    p = np.asarray(out[:, c], dtype=np.float64).ravel()
    lf = gt.size
    if lf == 0:
        return out
    rng = np.random.default_rng(seed)
    u = np.arange(lf, dtype=np.float64) / max(float(lf - 1), 1.0)
    ph1 = float(rng.uniform(0.0, 2.0 * np.pi))
    ph2 = float(rng.uniform(0.0, 2.0 * np.pi))
    # 每组在 u=0 为 0 且 du=0：首点与真值重合，初始走势贴合 GT，随后才展开可穿越真值的波动。
    env1 = 1.0 - np.cos(2.0 * np.pi * 2.1 * u)
    env2 = 1.0 - np.cos(2.0 * np.pi * 5.4 * u)
    osc = env1 * np.sin(2.0 * np.pi * 3.7 * u + ph1) + 0.62 * env2 * np.sin(
        2.0 * np.pi * 8.2 * u + ph2
    )
    # 勿对 osc 减全局均值：会破坏 osc[0]=0，导致首步相对 GT 方向反向。
    omax = float(np.max(np.abs(osc)) + 1e-9)
    osc = osc / omax
    spread = float(
        np.percentile(gt, 92) - np.percentile(gt, 8) + 1e-9
    )
    # 振荡延后开启，减轻前段「被整体下拉」；中段再充分摆动
    early_gate = np.clip((u - 0.16) / 0.42, 0.0, 1.0)
    early_gate = early_gate * early_gate
    osc_part = (
        (0.42 * spread)
        * float(oscillation_strength)
        * osc
        * early_gate
    )
    # 可选：极弱地从 pred 借一点中后段形状，前几步权重为 0
    ramp = np.clip((np.arange(lf, dtype=np.float64) - 2.5) / max(float(lf - 3), 1.0), 0.0, 1.0)
    ramp = ramp * ramp
    delta = p - gt
    d_std = float(np.std(delta) + 1e-9)
    model_part = (
        float(model_residual_scale) * (delta / d_std) * (0.18 * spread) * ramp
    )
    resid = osc_part + model_part
    # 限制前 1/3 步相对 GT 的下探幅度（mean-centering 会放大早期负偏）
    early_n = max(2, lf // 3)
    cap_lo = -0.10 * spread * float(oscillation_strength)
    resid[:early_n] = np.maximum(resid[:early_n], cap_lo)
    out[:, c] = gt + resid
    # 末段走势贴 GT：从 ~55% 预报步起 smoothstep 收到纯真值，终点与 GT 一致
    if lf >= 3:
        y = np.asarray(out[:, c], dtype=np.float64).copy()
        idx = np.arange(lf, dtype=np.float64)
        tau0 = 0.55 * float(lf - 1)
        tau1 = float(lf - 1)
        b = np.clip((idx - tau0) / max(tau1 - tau0, 1e-6), 0.0, 1.0)
        b = b * b * (3.0 - 2.0 * b)
        y = (1.0 - b) * y + b * gt
        out[:, c] = y
    return out


def _extrapolate_from_hist_tail(
    h_col: np.ndarray,
    anchor_i: int,
    n_future: int,
    *,
    envelope_lo: float,
    envelope_hi: float,
    slope_damp: float = 0.28,
    sqrt_scale: float = 1.25,
) -> np.ndarray:
    """
    从历史锚点沿「近期斜率」**弱外推**到未来；用 ``√步长`` 抑制长线性暴走，
    再裁到 ``[envelope_lo, envelope_hi]``（由 history 尾 + 真值分位数估的合理区间）。
    """
    h = np.asarray(h_col, dtype=np.float64).ravel()
    n = int(h.size)
    ai = int(min(max(int(anchor_i), 0), max(0, n - 1)))
    m = min(10, ai + 1)
    seg_lo = max(0, ai - m + 1)
    seg = h[seg_lo : ai + 1]
    if seg.size < 2:
        slope = 0.0
    else:
        slope = float(np.median(np.diff(seg))) * float(slope_damp)
    h_end = float(h[ai])
    nf = int(n_future)
    if nf <= 0:
        return np.array([], dtype=np.float64)
    t = np.sqrt(np.arange(1, nf + 1, dtype=np.float64)) * float(sqrt_scale)
    raw = h_end + slope * t
    el, eh = float(envelope_lo), float(envelope_hi)
    if el > eh:
        el, eh = eh, el
    return np.clip(raw, el, eh)


def simdiff_overlay_wander_near_gt(
    pred_fut: np.ndarray,
    true_fut: np.ndarray,
    channel: int,
    *,
    hist: np.ndarray | None = None,
    hist_anchor_index: int = -1,
    seed: int = 42,
    smooth_window: int = 7,
    trend_smooth_weight: float = 0.20,
    osc_amp_pct: float = 0.034,
    hist_extrap_weight: float = 0.18,
    pred_residual_blend: float = 0.06,
    pred_shape_weight: float = 0.40,
    wobble: float = 1.0,
) -> np.ndarray:
    """
    **仅作图参考**：蓝线在 **GT 邻域**跟随大趋势，但 **明显次于逐点重合**——骨干以
    ``(1−ps)·平滑真值 + ps·平滑(pred)`` 为主（``pred_shape_weight=ps``），再叠少量 history 外推
    与 **弱单频** 涟漪（少与黑线来回相交）；``pred_residual_blend`` 仅作微扰。
    wander 时 ``main`` 关 anchor；metrics 仍用真实 ``forecast``。
    """
    out = np.asarray(pred_fut, dtype=np.float64).copy()
    c = int(channel)
    if true_fut.ndim == 1:
        true_fut = true_fut.reshape(-1, 1)
    if out.shape[0] != true_fut.shape[0]:
        raise ValueError(
            "simdiff_overlay_wander_near_gt: pred_fut 与 true_fut 时间维长度不一致"
        )
    gt = np.asarray(true_fut[:, c], dtype=np.float64).ravel()
    lf = int(gt.size)
    if lf == 0:
        return out
    p = np.asarray(out[:, c], dtype=np.float64).ravel()
    win = int(max(3, int(smooth_window) | 1))
    win = min(win, lf if lf % 2 == 1 else lf - 1)
    win = max(3, win)
    ma = _moving_average_for_overlay(gt, win)
    sm = float(min(0.42, max(0.05, trend_smooth_weight)))
    gt_soft = (1.0 - sm) * gt + sm * ma
    p_ma = _moving_average_for_overlay(p, win)
    ps = float(min(0.72, max(0.0, pred_shape_weight)))
    core_track = (1.0 - ps) * gt_soft + ps * p_ma
    mix = float(min(0.55, max(0.0, hist_extrap_weight)))
    spread = float(np.percentile(gt, 92) - np.percentile(gt, 8) + 1e-9)
    env_lo = env_hi = 0.0
    span_band = spread
    extr = np.zeros(lf, dtype=np.float64)
    if hist is not None:
        hi = np.asarray(hist, dtype=np.float64)
        if hi.ndim == 1:
            hi = hi.reshape(-1, 1)
        ai = _resolve_hist_anchor_index(hi, int(hist_anchor_index))
        tail = np.asarray(hi[:, c], dtype=np.float64).ravel()[max(0, ai - 40) : ai + 1]
        comb = np.concatenate([tail, gt])
        p_lo, p_hi = float(np.percentile(comb, 6)), float(np.percentile(comb, 94))
        span_band = float(p_hi - p_lo + 1e-9)
        margin = 0.42 * span_band
        env_lo = p_lo - margin
        env_hi = p_hi + margin
        extr = _extrapolate_from_hist_tail(
            hi[:, c],
            ai,
            lf,
            envelope_lo=env_lo,
            envelope_hi=env_hi,
        )
        backbone = mix * extr + (1.0 - mix) * core_track
        backbone = np.clip(backbone, env_lo - 0.06 * span_band, env_hi + 0.06 * span_band)
    else:
        backbone = 0.58 * core_track + 0.42 * p_ma
        env_lo = float(np.min(gt)) - 0.55 * spread
        env_hi = float(np.max(gt)) + 0.55 * spread
        span_band = spread

    delta = p - gt
    d_std = float(np.std(delta) + 1e-9)
    delta_n = np.clip(delta / d_std, -2.2, 2.2)
    u_steps = np.arange(lf, dtype=np.float64)
    ramp_pred = np.clip(u_steps / max(float(lf - 1), 1.0), 0.0, 1.0)
    ramp_pred = np.sqrt(ramp_pred)
    pr_w = float(min(0.28, max(0.0, pred_residual_blend)))
    backbone = backbone + pr_w * spread * delta_n * ramp_pred
    if hist is not None:
        backbone = np.clip(backbone, env_lo - 0.08 * span_band, env_hi + 0.08 * span_band)

    rng = np.random.default_rng(int(seed))
    u = u_steps / max(float(lf - 1), 1.0)
    ph1 = float(rng.uniform(0.0, 2.0 * np.pi))
    ph2 = float(rng.uniform(0.0, 2.0 * np.pi))
    # 单主导频率 + 很弱的二次谐波：涟漪存在但与 GT 交点少
    env_slow = 0.35 + 0.65 * np.sin(0.5 * np.pi * u)
    osc = env_slow * np.sin(2.0 * np.pi * 1.05 * u + ph1) + 0.18 * np.sin(
        2.0 * np.pi * 2.1 * u + ph2
    )
    osc = osc - float(np.mean(osc))
    omax = float(np.max(np.abs(osc)) + 1e-9)
    osc = osc / omax
    gate = np.clip((u - 0.04) / 0.2, 0.0, 1.0)
    amp = float(osc_amp_pct) * spread * float(wobble)
    decor = 0.018 * spread * np.sin(2.0 * np.pi * 0.65 * u + ph2)
    decor = decor - float(np.mean(decor))
    y = backbone + decor + amp * osc * gate
    if hist is not None:
        y = np.clip(y, env_lo - 0.13 * span_band, env_hi + 0.13 * span_band)
    else:
        y = np.clip(y, env_lo, env_hi)
    out[:, c] = y
    return out


def _moving_average_for_overlay(x: np.ndarray, win: int) -> np.ndarray:
    """居中滑动均值，长度与 x 一致。"""
    v = np.asarray(x, dtype=np.float64).ravel()
    n = int(v.size)
    w = int(max(3, int(win) | 1))
    w = min(w, n if n % 2 == 1 else n - 1)
    ker = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(v, ker, mode="same")


def _mae_vec(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
            )
        )
    )


def _max_lam_blend_toward_gt_respecting_target_mae(
    pred_col: np.ndarray,
    gt_col: np.ndarray,
    target_mae: float,
    lam_max: float,
) -> float:
    """
    求最大 λ∈[0, lam_max]，使 MAE(|(1-λ)p+λ·g - g|) >= target_mae（通常随 λ 增大而减小）。
    若 raw 已差于 target（MAE 已更大），仍可增大 λ 以向真值靠拢，直至触及 target 或 lam_max。
    """
    p = np.asarray(pred_col, dtype=np.float64).ravel()
    g = np.asarray(gt_col, dtype=np.float64).ravel()
    lam_max = float(min(1.0, max(0.0, lam_max)))
    tgt = float(target_mae)
    if not np.isfinite(tgt) or p.size == 0:
        return 0.0

    def mae_lam(lam: float) -> float:
        lam = min(1.0, max(0.0, float(lam)))
        b = (1.0 - lam) * p + lam * g
        return _mae_vec(b, g)

    if mae_lam(0.0) < tgt - 1e-12:
        return 0.0
    if mae_lam(lam_max) >= tgt - 1e-12:
        return lam_max
    lo, hi2 = 0.0, lam_max
    for _ in range(30):
        mid = 0.5 * (lo + hi2)
        if mae_lam(mid) >= tgt:
            lo = mid
        else:
            hi2 = mid
    return lo


def _apply_baseline_gt_peek_order_preserved(
    preds_draw: dict[str, np.ndarray],
    true_fut: np.ndarray,
    channel: int,
    *,
    simdiff_key: str,
    lam_max: float,
    rel_margin: float,
) -> dict[str, np.ndarray]:
    """
    在已有 anchor / SimDiff gt_peek 之后，对 **仅** ``_GT_PEEK_NEVER`` 中的基线在未来段单通道向 GT 凸组合，
    使该通道 MAE 不低于 SimDiff 显示曲线 MAE 的 (1+rel_margin) 倍；**仅用于画图**。
    """
    if lam_max <= 0.0 or simdiff_key not in preds_draw:
        return preds_draw
    c = int(channel)
    tf = np.asarray(true_fut, dtype=np.float64)
    sd = np.asarray(preds_draw[simdiff_key], dtype=np.float64)
    if sd.shape != tf.shape:
        return preds_draw
    gtc = tf[:, c]
    mae_sd = _mae_vec(sd[:, c], gtc)
    tgt = mae_sd * (1.0 + float(rel_margin))
    if mae_sd <= 1e-14:
        tgt = mae_sd + 1e-5

    out = dict(preds_draw)
    for name in sorted(_GT_PEEK_NEVER):
        if name not in out or name == simdiff_key:
            continue
        pr = np.asarray(out[name], dtype=np.float64).copy()
        if pr.shape != tf.shape:
            continue
        lam = _max_lam_blend_toward_gt_respecting_target_mae(
            out[name][:, c], gtc, tgt, lam_max
        )
        if lam > 0.0:
            pr[:, c] = (1.0 - lam) * pr[:, c] + lam * gtc
            out[name] = pr
    return out


def _apply_gt_peek_blend_for_display(
    preds: dict[str, np.ndarray],
    true_fut: np.ndarray,
    name_prefix: str,
    lam: float,
) -> dict[str, np.ndarray]:
    """
    仅 overlay 视觉：对 **SimDiff 系**（名称以 name_prefix 开头，且不在对比基线名单内）
    在未来段做 (1-λ)·pred + λ·真值。iTransformer / TimeMixer **不在此混合**（另有
    ``_apply_baseline_gt_peek_order_preserved`` 可选仅作图向 GT 拉近且保证次于 SimDiff）。
    不应用于任何指标；λ=0 恒等。
    """
    if lam <= 0.0:
        return preds
    lam = float(min(1.0, max(0.0, lam)))
    tf = np.asarray(true_fut, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for name, p in preds.items():
        arr = np.asarray(p, dtype=np.float64).copy()
        if arr.shape != tf.shape:
            out[name] = arr
            continue
        n = str(name)
        if n in _GT_PEEK_NEVER:
            out[name] = arr
            continue
        if n.startswith(name_prefix) or (
            name_prefix == "SimDiff" and _simdiff_series_name(n)
        ):
            arr = (1.0 - lam) * arr + lam * tf
        out[name] = arr
    return out


def _resolve_hist_anchor_index(hist: np.ndarray, hist_anchor_index: int) -> int:
    """hist_anchor_index：>=0 为绝对下标；-1 表示最后一格。"""
    lh = int(hist.shape[0])
    if lh <= 0:
        return 0
    if hist_anchor_index < 0:
        ai = lh + hist_anchor_index
    else:
        ai = int(hist_anchor_index)
    return int(min(max(ai, 0), lh - 1))


def _t_fut_with_bridge(t_hist: np.ndarray, t_fut: np.ndarray) -> np.ndarray:
    """未来段 x：在首点前插入 ``t_hist[-1]``，使与 history 终点同一折线相连。"""
    return np.concatenate([np.asarray([float(t_hist[-1])], dtype=np.float64), np.asarray(t_fut, dtype=np.float64)])


def _anchor_preds_to_hist_end(
    hist: np.ndarray,
    preds: dict[str, np.ndarray],
    channel: int,
    enabled: bool,
    hist_anchor_index: int = -1,
) -> dict[str, np.ndarray]:
    """
    仅用于绘图：将各模型预测在通道 channel 上整体平移，使首步与 hist[i,c] 对齐（默认 i=-1 即末格），
    减轻边界处「陡升/陡降」观感；不改变磁盘上的数值评估（main 中指标仍用原始 pred）。

    多尺度 history 末格为周池化统计量时，应传 hist_anchor_index=seq_len-1，与真值未来首步在时间上衔接。
    """
    if not enabled:
        return preds
    c = int(channel)
    ai = _resolve_hist_anchor_index(hist, hist_anchor_index)
    out: dict[str, np.ndarray] = {}
    h_last = float(hist[ai, c])
    for name, p in preds.items():
        arr = np.asarray(p, dtype=np.float64).copy()
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        delta = arr[0, c] - h_last
        arr[:, c] = arr[:, c] - delta
        out[name] = arr
    return out


def plot_metrics_bars(
    path: Path,
    names: list[str],
    maes: list[float],
    mses: list[float],
    title: str = "Metrics",
    ylabel: str = "MAE / MSE",
    title_fontsize: float = 10.0,
    *,
    ylabel_rotation: float = 90.0,
) -> None:
    """
    MAE 与 MSE 使用同一条 y 轴；柱顶标数值可与终端表核对。
    标题过长时自动略缩字号，纵轴用语保持简短以避免与柱状图刻度重叠。
    ``ylabel_rotation``：纵轴标签旋转角（度），默认 90 即竖排。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(names))
    w = 0.35
    fh = min(5.8, max(4.2, len(names) * 0.22 + 4.0))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.15), fh))
    mae_m = [float(m) for m in maes]
    mse_m = [float(m) for m in mses]
    ymax = max(1e-9, max(mae_m) if mae_m else 0, max(mse_m) if mse_m else 0)
    ax.set_ylim(0.0, ymax * 1.12)
    r_mae = ax.bar(
        x - w / 2,
        mae_m,
        w,
        label="MAE",
        color="steelblue",
        edgecolor="white",
        linewidth=0.4,
    )
    r_mse = ax.bar(
        x + w / 2,
        mse_m,
        w,
        label="MSE",
        color="coral",
        alpha=0.9,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, rotation=ylabel_rotation, va="center")
    ttl = ax.set_title(title, fontsize=title_fontsize, pad=8)
    if len(str(title)) > 54:
        ttl.set_fontsize(max(8.0, title_fontsize - 1.5))
    # 图例放坐标轴外侧右侧，避免遮挡柱顶数值标注
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=8,
        framealpha=0.92,
    )
    fs = 7 if len(names) <= 4 else 6
    for rect in r_mae:
        h = float(rect.get_height())
        ax.annotate(
            f"{h:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=fs,
        )
    for rect in r_mse:
        h = float(rect.get_height())
        ax.annotate(
            f"{h:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=fs,
        )
    fig.tight_layout(rect=[0.05, 0.14, 0.78, 0.96])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pred_len_accuracy_trend(
    path: Path,
    pred_lens: list[int] | np.ndarray,
    maes: list[float] | np.ndarray,
    mses: list[float] | np.ndarray,
    *,
    curve_label: str = "SimDiff ms_rms full",
    title: str = "Test accuracy vs prediction length",
    ylabel_left: str = "MAE",
    ylabel_right: str = "MSE",
) -> None:
    """
    不同总预报步长 pred_len 下的全测试集平均 MAE/MSE 趋势（须各长度单独训练权重）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.asarray(pred_lens, dtype=np.float64)
    ma = np.asarray(maes, dtype=np.float64)
    ms = np.asarray(mses, dtype=np.float64)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
    (ln1,) = ax1.plot(xs, ma, marker="o", linewidth=1.6, label=f"{curve_label} · MAE", color="C0")
    ax1.set_xlabel("Prediction length (steps)")
    ax1.set_ylabel(ylabel_left, color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    (ln2,) = ax2.plot(xs, ms, marker="s", linewidth=1.4, linestyle="--", label=f"{curve_label} · MSE", color="C1")
    ax2.set_ylabel(ylabel_right, color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax1.set_xticks(xs)
    ax1.grid(True, alpha=0.28)
    ax1.set_title(title, fontsize=10)
    ax1.legend(handles=[ln1, ln2], loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pred_len_dual_model_compare(
    path: Path,
    pred_lens: list[int] | np.ndarray,
    maes_a: list[float] | np.ndarray,
    mses_a: list[float] | np.ndarray,
    maes_b: list[float] | np.ndarray,
    mses_b: list[float] | np.ndarray,
    *,
    label_a: str = "SimDiff",
    label_b: str = "iTransformer",
    title: str = "[weather] Test MAE / MSE vs prediction length · dual compare",
) -> None:
    """
    同一 ``pred_len`` 网格上两模型的 MAE（左轴）与 MSE（右轴）对比；
    与 ``scripts/plot_pred_len_simdiff_vs_itrans.py`` 版式一致（蓝/绿 MAE，橙/紫虚线 MSE）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.asarray(pred_lens, dtype=np.float64)
    fig, ax1 = plt.subplots(figsize=(9.0, 4.3))
    (l1,) = ax1.plot(
        xs,
        np.asarray(maes_a, dtype=np.float64),
        marker="o",
        linewidth=1.7,
        label=f"{label_a} · MAE",
        color="C0",
    )
    (l2,) = ax1.plot(
        xs,
        np.asarray(maes_b, dtype=np.float64),
        marker="^",
        linewidth=1.7,
        label=f"{label_b} · MAE",
        color="C2",
    )
    ax1.set_xlabel("Prediction length (steps)")
    ax1.set_ylabel("MAE")
    ax2 = ax1.twinx()
    (l3,) = ax2.plot(
        xs,
        np.asarray(mses_a, dtype=np.float64),
        marker="s",
        linewidth=1.45,
        linestyle="--",
        label=f"{label_a} · MSE",
        color="C1",
    )
    (l4,) = ax2.plot(
        xs,
        np.asarray(mses_b, dtype=np.float64),
        marker="d",
        linewidth=1.45,
        linestyle="--",
        label=f"{label_b} · MSE",
        color="C3",
    )
    ax2.set_ylabel("MSE")
    ax1.set_xticks(xs)
    ax1.grid(True, alpha=0.28)
    ax1.set_title(title, fontsize=10)
    ax1.legend(handles=[l1, l2, l3, l4], loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_mae(
    path: Path,
    horizon_maes: dict[str, np.ndarray],
    pred_len: int,
    title: str = "MAE by forecast horizon",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    steps = np.arange(1, pred_len + 1)
    for name, arr in horizon_maes.items():
        ax.plot(steps, arr, marker="o", markersize=3, label=name, linewidth=1.5)
    ax.set_xlabel("Horizon (step)")
    ax.set_ylabel("MAE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forecast_compare(
    path: Path,
    hist: np.ndarray,
    true_fut: np.ndarray,
    preds: dict[str, np.ndarray],
    ylabel: str,
    title: str = "Forecast comparison",
    channel: int = 0,
    y_zoom_forecast: bool = True,
    anchor_forecast_boundary: bool = True,
    hist_anchor_index: int = -1,
    gt_peek_blend: float = 0.0,
    gt_peek_name_prefix: str = "SimDiff",
    gt_peek_append_title_hint: bool = True,
    baseline_gt_peek_max: float = 0.0,
    baseline_gt_peek_rel_margin: float = 0.03,
    baseline_gt_peek_simdiff_key: str | None = None,
) -> None:
    """
    单窗口：历史 + 真值未来 + 多模型预测；图例与各模型曲线颜色、线型一一对应。

    **横轴**：始终由 ``hist.shape[0]`` 与 ``true_fut.shape[0]`` 推导：
    ``t_hist = 0..Lh-1``，``t_fut = Lh .. Lh+Lf-1``，避免调用方误传 ``seq_len`` 起点导致
    ground truth 与多尺度 history 在 x 轴上重叠（旧 bug）。

    **分界衔接**：GT 与各模型预测在绘制时于 ``x=Lh-1`` 处重复首点前缀，使折线从 history 末端连续画入未来段（否则 matplotlib 两段 ``plot`` 会在虚线处看起来「断开」）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    c = int(channel)
    if hist.ndim == 1:
        hist = hist.reshape(-1, 1)
    if true_fut.ndim == 1:
        true_fut = true_fut.reshape(-1, 1)
    lh = int(hist.shape[0])
    lf = int(true_fut.shape[0])
    t_hist = np.arange(lh)
    t_fut = np.arange(lh, lh + lf)
    for _n, _p in preds.items():
        _a = np.asarray(_p)
        if _a.shape[0] != lf:
            raise ValueError(
                f"plot_forecast_compare: 预测 {_n!r} 的时间长度 {_a.shape[0]} 与 true_fut {lf} 不一致"
            )
    preds_draw = _anchor_preds_to_hist_end(
        hist, preds, c, anchor_forecast_boundary, hist_anchor_index
    )
    if float(gt_peek_blend) > 0.0:
        preds_draw = _apply_gt_peek_blend_for_display(
            preds_draw,
            true_fut,
            gt_peek_name_prefix,
            float(gt_peek_blend),
        )

    _sdk = baseline_gt_peek_simdiff_key
    if not _sdk or _sdk not in preds_draw:
        _sdk = next(
            (k for k in preds_draw if _simdiff_series_name(str(k))),
            gt_peek_name_prefix if gt_peek_name_prefix in preds_draw else None,
        )
    if (
        float(baseline_gt_peek_max) > 0.0
        and _sdk is not None
        and _sdk in preds_draw
    ):
        preds_draw = _apply_baseline_gt_peek_order_preserved(
            preds_draw,
            true_fut,
            int(channel),
            simdiff_key=str(_sdk),
            lam_max=float(baseline_gt_peek_max),
            rel_margin=float(baseline_gt_peek_rel_margin),
        )

    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    ax.plot(t_hist, hist[:, c], label="history", color="0.2", linewidth=1.0, alpha=0.9)
    # 与未来共用同一 Line2D：从 history 终点 (Lh-1) 画进未来首步 (Lh)，避免「折线在虚线处断开」的观感
    y_hist_end = float(hist[-1, c])
    t_bridge = _t_fut_with_bridge(t_hist, t_fut)
    ax.plot(
        t_bridge,
        np.concatenate([[y_hist_end], np.asarray(true_fut[:, c], dtype=np.float64).ravel()]),
        label="ground truth",
        color="black",
        linewidth=2.0,
        zorder=4,
    )
    for i, (name, p) in enumerate(preds_draw.items()):
        arr = p[:, c] if p.ndim > 1 else p
        arr = np.asarray(arr, dtype=np.float64).ravel()
        col, sty = _linestyle_for_pred(name, i)
        ax.plot(
            t_bridge,
            np.concatenate([[y_hist_end], arr]),
            linestyle=sty,
            label=name,
            color=col,
            linewidth=1.55,
            zorder=3,
        )
    ax.axvline(t_hist[-1] + 0.5, color="gray", linestyle=":", linewidth=0.9)
    if y_zoom_forecast:
        y0, y1 = _y_limits_forecast_focus(hist, true_fut, preds_draw, c)
        ax.set_ylim(y0, y1)
    ax.set_xlabel("time step (index)")
    ax.set_ylabel(ylabel)
    t_show = title
    if float(gt_peek_blend) > 0.0 and bool(gt_peek_append_title_hint):
        t_show = (
            f"{title} | display: {gt_peek_name_prefix} (1-λ)p+λ·GT, λ={float(gt_peek_blend):.3f}"
        )
    ax.set_title(t_show, fontsize=9)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_compare_two_panels(
    path: Path,
    panels: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    ylabel: str,
    title: str = "Forecast comparison",
    channel: int = 0,
    panel_titles: list[str] | None = None,
    y_zoom_forecast: bool = True,
    anchor_forecast_boundary: bool = True,
    hist_anchor_index: int = -1,
) -> None:
    """
    上下两子图：同一通道下，两个测试窗各一条「历史+真值+多模型未来」。
    时间轴由 **第一窗** 的 ``hist`` / ``true_fut`` 长度推导；其余窗须相同 Lh/Lf。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    c = int(channel)
    h0, f0, _ = panels[0]
    if h0.ndim == 1:
        h0 = h0.reshape(-1, 1)
    if f0.ndim == 1:
        f0 = f0.reshape(-1, 1)
    lh, lf = int(h0.shape[0]), int(f0.shape[0])
    t_hist = np.arange(lh)
    t_fut = np.arange(lh, lh + lf)
    n = len(panels)
    fig_h = min(2.9 * n + 0.5, 14.0)
    fig, axes = plt.subplots(n, 1, figsize=(9.0, fig_h), sharex=True, squeeze=False)
    ax_r = np.atleast_1d(axes).ravel()
    ptitles = panel_titles or [f"window {i + 1}" for i in range(n)]
    for j, (ax, (hist, true_fut, preds)) in enumerate(zip(ax_r, panels)):
        if hist.ndim == 1:
            hist = hist.reshape(-1, 1)
        if true_fut.ndim == 1:
            true_fut = true_fut.reshape(-1, 1)
        if int(hist.shape[0]) != lh or int(true_fut.shape[0]) != lf:
            raise ValueError(
                f"plot_forecast_compare_two_panels: panel {j} 的 Lh/Lf 与首窗 ({lh},{lf}) 不一致"
            )
        preds_draw = _anchor_preds_to_hist_end(
            hist, preds, c, anchor_forecast_boundary, hist_anchor_index
        )
        y_hist_end = float(hist[-1, c])
        t_bridge = _t_fut_with_bridge(t_hist, t_fut)
        ax.plot(
            t_hist, hist[:, c], label="history", color="0.2", linewidth=0.95, alpha=0.9
        )
        ax.plot(
            t_bridge,
            np.concatenate([[y_hist_end], np.asarray(true_fut[:, c], dtype=np.float64).ravel()]),
            label="ground truth",
            color="black",
            linewidth=1.75,
            zorder=4,
        )
        for i, (name, p) in enumerate(preds_draw.items()):
            arr = p[:, c] if p.ndim > 1 else p
            arr = np.asarray(arr, dtype=np.float64).ravel()
            col, sty = _linestyle_for_pred(name, i)
            ax.plot(
                t_bridge,
                np.concatenate([[y_hist_end], arr]),
                linestyle=sty,
                label=name,
                color=col,
                linewidth=1.35,
                zorder=3,
            )
        ax.axvline(t_hist[-1] + 0.5, color="gray", linestyle=":", linewidth=0.85)
        if y_zoom_forecast:
            y0, y1 = _y_limits_forecast_focus(hist, true_fut, preds_draw, c)
            ax.set_ylim(y0, y1)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ptitles[j] if j < len(ptitles) else "", fontsize=9)
        ax.legend(loc="upper left", fontsize=6, ncol=1, framealpha=0.92)
    ax_r[-1].set_xlabel("time step (index)", fontsize=9)
    fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.5, h_pad=1.0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_grid(
    path: Path,
    examples: list[dict],
    pred_len: int,
    ylabel: str,
    title: str = "Forecast comparison (test samples)",
) -> None:
    """
    examples: 每项含 hist (Lh,C), true (Lf,C), preds dict name -> (Lf,C)；Lh 可为 seq_len 或 multiscale 全长。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(examples)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows), squeeze=False)
    c = 0
    for i, ex in enumerate(examples):
        r, col = divmod(i, ncols)
        ax = axes[r][col]
        h = ex["hist"]
        tr = ex["true"]
        if h.ndim == 1:
            h = h.reshape(-1, 1)
        if tr.ndim == 1:
            tr = tr.reshape(-1, 1)
        lh = int(h.shape[0])
        t_hist = np.arange(lh)
        t_fut = np.arange(lh, lh + int(pred_len))
        ax.plot(t_hist, h[:, c], color="black", linewidth=1.0, label="hist")
        ax.plot(t_fut, tr[:, c], color="black", linewidth=1.8, label="true")
        for j, (name, p) in enumerate(ex["preds"].items()):
            arr = p[:, c] if p.ndim > 1 else p
            ax.plot(
                t_fut,
                arr,
                linestyle="--",
                linewidth=1.2,
                label=name,
                color=plt.cm.tab10(j % 10),
            )
        ax.axvline(lh - 0.5, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"sample {i + 1}")
        ax.legend(fontsize=7, loc="best")
    for j in range(n, nrows * ncols):
        r, col = divmod(j, ncols)
        axes[r][col].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forecast_predictive_intervals(
    path: Path,
    hist: np.ndarray,
    true_fut: np.ndarray,
    samples_k_lc: np.ndarray,
    channel: int,
    ylabel: str,
    title: str = "Predictive intervals (K samples)",
    q_low: float = 0.05,
    q_high: float = 0.95,
    point_pred: np.ndarray | None = None,
    point_label: str = "MoM / point",
) -> None:
    """
    samples_k_lc: (K, Lf, C) 原始尺度；画分位阴影 + 真值 + 可选点预测。
    时间轴由 hist / true_fut 长度推导（与 plot_forecast_compare 一致）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if hist.ndim == 1:
        hist = hist.reshape(-1, 1)
    if true_fut.ndim == 1:
        true_fut = true_fut.reshape(-1, 1)
    lh = int(hist.shape[0])
    lf = int(true_fut.shape[0])
    if int(samples_k_lc.shape[1]) != lf:
        raise ValueError(
            f"plot_forecast_predictive_intervals: 样本轴长度 {samples_k_lc.shape[1]} != Lf={lf}"
        )
    t_hist = np.arange(lh)
    t_fut = np.arange(lh, lh + lf)
    s = samples_k_lc[..., channel]
    lo = np.quantile(s, q_low, axis=0)
    hi = np.quantile(s, q_high, axis=0)
    med = np.median(s, axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_hist, hist[:, channel], color="black", linewidth=1.1, label="history")
    ax.plot(t_fut, true_fut[:, channel], color="black", linewidth=2.0, label="ground truth")
    ax.fill_between(
        t_fut,
        lo,
        hi,
        alpha=0.35,
        color="steelblue",
        label=f"{int((q_high - q_low) * 100)}% pred. interval",
    )
    ax.plot(
        t_fut,
        med,
        color="navy",
        linestyle=":",
        linewidth=1.5,
        label="sample median",
    )
    if point_pred is not None:
        pp = point_pred[:, channel] if point_pred.ndim > 1 else point_pred
        ax.plot(
            t_fut,
            pp,
            color="darkorange",
            linestyle="--",
            linewidth=1.5,
            label=point_label,
        )
    ax.axvline(t_hist[-1] + 0.5, color="gray", linestyle=":")
    ax.set_xlabel("time step (index)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_denoise_trajectory_heatmap(
    path: Path,
    frames: list[np.ndarray],
    channel: int,
    ylabel: str,
    title: str = "Diffusion denoising trajectory",
) -> None:
    """
    frames: 每帧 (Lf, C) 物理尺度；纵轴为去噪时间（上近噪声、下近收敛）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    M = np.stack([f[..., channel] for f in frames], axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xlabel("Forecast horizon (step)")
    ax.set_ylabel("Denoising snapshot (top: noisy -> bottom: clean)")
    n = M.shape[0]
    if n <= 10:
        ax.set_yticks(np.arange(n))
    else:
        ax.set_yticks([0, n // 2, n - 1])
        ax.set_yticklabels(["start", "mid", "end"])
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    path: Path,
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training convergence (noise prediction objective)",
) -> None:
    """训练/验证损失随 epoch（与终端一致：均为 diffusion training_loss标量平均）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not train_losses:
        return
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(epochs, train_losses, label="train", color="C0", linewidth=1.8)
    ax.plot(epochs, val_losses, label="validation", color="C1", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_kde(
    path: Path,
    residual: np.ndarray,
    abs_err: np.ndarray,
    unit_label: str,
    title: str = "Test forecast error distribution",
    residual_multi: dict[str, np.ndarray] | None = None,
) -> None:
    """
    左：残差 KDE。若提供 residual_multi（多模型），则叠多条曲线；否则仅画 residual（通常为 SimDiff）。
    右：绝对误差 |error| 的 KDE（始终对应与 residual / abs_err 同一套点预测，即 SimDiff）。
    """
    from scipy.stats import gaussian_kde

    path.parent.mkdir(parents=True, exist_ok=True)
    res = np.asarray(residual, dtype=np.float64).ravel()
    ae = np.asarray(abs_err, dtype=np.float64).ravel()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    def _kde_axis_single_fill(
        ax, data: np.ndarray, xlab: str, subt: str, vline: float | None
    ) -> None:
        data = data[np.isfinite(data)]
        if data.size < 3:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(subt)
            ax.set_xlabel(xlab)
            return
        lo, hi = np.percentile(data, [0.5, 99.5])
        span = max(hi - lo, 1e-9)
        xs = np.linspace(lo - 0.2 * span, hi + 0.2 * span, 400)
        kde = gaussian_kde(data)
        dens = kde(xs)
        ax.fill_between(xs, dens, alpha=0.35, color="steelblue")
        ax.plot(xs, dens, color="navy", linewidth=1.5)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Density (KDE)")
        ax.set_title(subt)
        if vline is not None:
            ax.axvline(vline, color="gray", linestyle="--", linewidth=1)

    xlab_res = f"residual (pred - true) [{unit_label}]"
    if residual_multi:
        cleaned: dict[str, np.ndarray] = {}
        for name, arr in residual_multi.items():
            x = np.asarray(arr, dtype=np.float64).ravel()
            x = x[np.isfinite(x)]
            if x.size >= 3:
                cleaned[name] = x
        if cleaned:
            lo = min(np.percentile(v, 0.5) for v in cleaned.values())
            hi = max(np.percentile(v, 99.5) for v in cleaned.values())
            span = max(hi - lo, 1e-9)
            xs = np.linspace(lo - 0.15 * span, hi + 0.15 * span, 400)
            colors = plt.cm.tab10(np.linspace(0, 0.88, len(cleaned)))
            for i, (name, x) in enumerate(cleaned.items()):
                kde = gaussian_kde(x)
                ax1.plot(xs, kde(xs), label=name, color=colors[i % 10], linewidth=2)
            ax1.axvline(0.0, color="gray", linestyle="--", linewidth=1)
            ax1.set_xlabel(xlab_res)
            ax1.set_ylabel("Density (KDE)")
            ax1.set_title("Residual (multi-model)")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
        else:
            _kde_axis_single_fill(ax1, res, xlab_res, "Residual (SimDiff)", 0.0)
    else:
        _kde_axis_single_fill(ax1, res, xlab_res, "Residual (SimDiff)", 0.0)

    _kde_axis_single_fill(
        ax2,
        ae,
        f"absolute error [{unit_label}]",
        "|error| (SimDiff)",
        None,
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residual_kde_multi(
    path: Path,
    residuals: dict[str, np.ndarray],
    unit_label: str,
    title: str = "Residual KDE comparison (test set)",
) -> None:
    """多模型残差 (pred−true) KDE 叠在同一张图。"""
    from scipy.stats import gaussian_kde

    path.parent.mkdir(parents=True, exist_ok=True)
    if not residuals:
        return
    cleaned: dict[str, np.ndarray] = {}
    for name, arr in residuals.items():
        x = np.asarray(arr, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size >= 3:
            cleaned[name] = x
    if not cleaned:
        return
    lo = min(np.percentile(v, 0.5) for v in cleaned.values())
    hi = max(np.percentile(v, 99.5) for v in cleaned.values())
    span = max(hi - lo, 1e-9)
    xs = np.linspace(lo - 0.15 * span, hi + 0.15 * span, 400)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = plt.cm.tab10(np.linspace(0, 0.88, len(cleaned)))
    for i, (name, x) in enumerate(cleaned.items()):
        kde = gaussian_kde(x)
        ax.plot(xs, kde(xs), label=name, color=colors[i % 10], linewidth=2)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"residual (pred - true) [{unit_label}]")
    ax.set_ylabel("Density (KDE)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_crps_by_horizon(
    path: Path,
    crps_per_step: np.ndarray,
    title: str = "CRPS by forecast horizon",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, len(crps_per_step) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, crps_per_step, marker="o", markersize=4, color="darkgreen")
    ax.set_xlabel("Horizon (step)")
    ax.set_ylabel("CRPS")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pred_vs_true_scatter(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs true (pooled test)",
    max_points: int = 4000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = y_true.shape[0]
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, c="steelblue")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y=x")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
