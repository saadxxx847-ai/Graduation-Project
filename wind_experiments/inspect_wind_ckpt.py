#!/usr/bin/env python3
"""检查 Wind SimDiff checkpoint：从 net.in_proj.weight 推断输入通道数 C。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"


def main() -> None:
    names = sys.argv[1:] or [
        "simdiff_weather_best_wind.pt",
        "simdiff_weather_best_wind_mv.pt",
    ]
    for name in names:
        path = CKPT / name
        print(f"\n=== {path.name} ===")
        if not path.is_file():
            print("文件不存在:", path)
            continue
        s = torch.load(path, map_location="cpu", weights_only=False)
        meta = s.get("meta") or {}
        print("meta.input_dim:", meta.get("input_dim"))
        print("meta.temperature_only:", meta.get("temperature_only"))
        sd = s.get("model")
        if not isinstance(sd, dict):
            print("[warn] checkpoint 无 model 字典")
            continue
        keys = [k for k in sd if k.endswith("in_proj.weight")]
        if not keys:
            print("未找到 *in_proj.weight；前若干键:", list(sd.keys())[:15])
            continue
        k = sorted(keys)[0]
        w = sd[k]
        print(f"{k} shape={tuple(w.shape)} -> C = {int(w.shape[1])}")


if __name__ == "__main__":
    main()
