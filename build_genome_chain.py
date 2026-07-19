"""Build the whole genome ladder as ONE long continuous take -> docs/engine/genome_chain.gif.

The take itself is the ``warp_genome_chain`` scene (see that module): the real gpu_board tokenising into
its own tokens, then base pairs -> helices -> nucleosomes -> fibre -> telomere -> chromosome, every seam a
cross-dissolve so it reads as one shape retransforming. This script just samples that scene at a fixed fps
and encodes a compact GIF with a palette sampled across every stage (so nothing gets quantised away).

    python build_genome_chain.py                 # auto device (CUDA if present, else CPU)
    python build_genome_chain.py --device cuda    # force GPU
    python build_genome_chain.py --fps 18 --width 512 --height 342

For an interactive live view instead of a file, use ``watch.py --scene warp_genome_chain``.
"""
from __future__ import annotations

import argparse
import os
import time as _t

import numpy as np
import warp as wp
from PIL import Image

from warp_shaders.scene import get_scene
from warp_shaders.scenes.warp_genome_chain import TOTAL

OUT = os.path.join("docs", "engine", "genome_chain.gif")


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if wp.get_cuda_device_count() > 0 else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the genome chain to a GIF.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--fps", type=float, default=14.0)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--colors", type=int, default=200)
    ap.add_argument("--hold", type=float, default=1.2, help="seconds to hold on the settled X at the end")
    ap.add_argument("-o", "--out", default=OUT)
    args = ap.parse_args()

    wp.init()
    device = _pick_device(args.device)
    sc = get_scene("warp_genome_chain")
    n = int(round(TOTAL * args.fps))
    print(f"device: {device}  |  {n} frames @ {args.fps} fps  |  {args.width}x{args.height}", flush=True)

    frames = []
    s = _t.time()
    # sample t in [0, TOTAL) — never exactly TOTAL, which would wrap (gt = t % TOTAL) back to the
    # opening board and make the take look like it *ends* on a graphics card.
    eps = min(1e-3, TOTAL / (2.0 * max(n, 1)))
    for k in range(n):
        t = min(TOTAL * (k / max(n - 1, 1)), TOTAL - eps)
        fr = np.clip(sc.render(args.width, args.height, t, (0.0, 0.0), device), 0.0, 1.0)
        frames.append(Image.fromarray((fr * 255.0 + 0.5).astype(np.uint8), "RGB"))
        if k % 20 == 0:
            print(f"  {k + 1}/{n}  t={t:.2f}s", end="\r", flush=True)
    # hold on the settled metaphase X so the take ends on the chromosome, not a wrap
    hold = int(round(args.hold * args.fps))
    frames.extend(frames[-1].copy() for _ in range(hold))
    print(f"\n  rendered {n} frames (+{hold} hold) in {_t.time() - s:.1f}s", flush=True)
    n = len(frames)

    # one global palette sampled ACROSS every stage (card pastels, grey beads, purple chromosome).
    picks = [int(n * fr) for fr in (0.03, 0.12, 0.22, 0.33, 0.45, 0.57, 0.70, 0.82, 0.93, 0.99)]
    montage = Image.fromarray(np.vstack([np.asarray(frames[min(p, n - 1)]) for p in picks]), "RGB")
    pal = montage.convert("P", palette=Image.ADAPTIVE, colors=int(args.colors))
    imgs = [im.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for im in frames]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"WROTE {args.out}  {os.path.getsize(args.out) / 1e6:.1f} MB  {len(imgs)} frames")


if __name__ == "__main__":
    main()
