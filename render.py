#!/usr/bin/env python3
"""Render the solar-system Warp scene to an image or an animated sequence.

Examples
--------
Single frame (auto device — CUDA if present, else CPU)::

    python render.py --time 3.0 --width 1280 --height 720 -o frame.png

A short animation as a GIF::

    python render.py --frames 60 --fps 30 --gif out/spin.gif

Individual PNG frames into a directory::

    python render.py --frames 120 --out-dir out/frames
"""

import argparse
import os

import numpy as np
import warp as wp

from warp_shaders.solar_system import render


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Clamp linear color to [0, 1] and quantize to 8-bit."""
    return (np.clip(frame, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if wp.get_cuda_device_count() > 0 else "cpu"


def save_png(path: str, frame: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(to_uint8(frame), mode="RGB").save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the Warp solar-system scene.")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--time", type=float, default=0.0, help="scene time (seconds) for a single frame")
    ap.add_argument("--mouse", type=float, nargs=2, default=(0.0, 0.0),
                    metavar=("MX", "MY"), help="camera orbit, in pixel coords like iMouse")
    ap.add_argument("--frames", type=int, default=1, help="number of frames to render")
    ap.add_argument("--fps", type=float, default=30.0, help="frames per second (sets time step + GIF timing)")
    ap.add_argument("-o", "--out", default="frame.png", help="output path for a single frame")
    ap.add_argument("--out-dir", default=None, help="directory to write frame_####.png into")
    ap.add_argument("--gif", default=None, help="write the sequence as an animated GIF at this path")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    wp.init()
    device = pick_device(args.device)
    print(f"rendering on device: {device}")

    if args.frames <= 1:
        frame = render(args.width, args.height, args.time, tuple(args.mouse), device)
        save_png(args.out, frame)
        print(f"wrote {args.out}  ({args.width}x{args.height}, t={args.time})")
        return

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    dt = 1.0 / args.fps
    gif_frames = []
    for k in range(args.frames):
        t = k * dt
        frame = render(args.width, args.height, t, tuple(args.mouse), device)
        u8 = to_uint8(frame)
        if args.out_dir:
            save_png(os.path.join(args.out_dir, f"frame_{k:04d}.png"), frame)
        if args.gif:
            from PIL import Image
            gif_frames.append(Image.fromarray(u8, mode="RGB"))
        print(f"  frame {k + 1}/{args.frames}  t={t:.3f}", end="\r", flush=True)
    print()

    if args.gif:
        os.makedirs(os.path.dirname(os.path.abspath(args.gif)), exist_ok=True)
        gif_frames[0].save(
            args.gif, save_all=True, append_images=gif_frames[1:],
            duration=int(1000 / args.fps), loop=0,
        )
        print(f"wrote {args.gif}  ({args.frames} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
