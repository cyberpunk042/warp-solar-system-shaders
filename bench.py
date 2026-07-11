#!/usr/bin/env python3
"""Micro-benchmark: time one or more scenes and print ms/frame.

    python bench.py                       # a default heavy-scene set
    python bench.py ss_nebula black_hole  # specific scenes
    python bench.py --width 640 --height 400 --runs 5 ss_nebula

Each scene is compiled once (warm-up render, not timed), then rendered `--runs`
times; the minimum wall-clock is reported (least noisy estimator). CPU here;
the same kernels run on CUDA when a GPU is present.
"""

import argparse
import time

import warp as wp

from warp_shaders.lod import set_active
from warp_shaders.scene import get_scene

_DEFAULT = ["ss_nebula", "neutron_star", "black_hole", "ss_blackhole",
            "nebula", "solar_system", "terrain", "earth_v2"]


def bench(name, width, height, t, runs, device):
    sc = get_scene(name)
    sc.render(width, height, 0.0, (0.0, 0.0), device)     # warm-up / compile
    times = []
    for _ in range(runs):
        a = time.perf_counter()
        sc.render(width, height, t, (0.0, 0.0), device)
        times.append(time.perf_counter() - a)
    return min(times) * 1000.0


def main():
    ap = argparse.ArgumentParser(description="Time Warp scenes (ms/frame).")
    ap.add_argument("scenes", nargs="*", help="scene names (default: a heavy set)")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--time", type=float, default=1.0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--quality", default="low")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    wp.init()
    set_active(args.quality, args.device)
    scenes = args.scenes or _DEFAULT
    print(f"{'scene':<16} {args.width}x{args.height}  quality={args.quality}  "
          f"device={args.device}  runs={args.runs}")
    for s in scenes:
        try:
            ms = bench(s, args.width, args.height, args.time, args.runs, args.device)
            print(f"  {s:<16} {ms:8.1f} ms/frame")
        except Exception as e:                              # noqa: BLE001
            print(f"  {s:<16} ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
