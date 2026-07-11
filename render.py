#!/usr/bin/env python3
"""Render a Warp scene to an image or an animated sequence.

Examples
--------
List available scenes::

    python render.py --list

Single frame (auto device — CUDA if present, else CPU)::

    python render.py --scene neutron_star --time 3.0 --width 1280 --height 720 -o frame.png

A short animation as a GIF::

    python render.py --scene neutron_star --frames 60 --fps 30 --gif out/spin.gif

Individual PNG frames into a directory::

    python render.py --scene neutron_star --frames 120 --out-dir out/frames
"""

import argparse
import os

import numpy as np
import warp as wp

from warp_shaders.scene import get_scene, list_scenes


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
    ap = argparse.ArgumentParser(description="Render a Warp scene.")
    ap.add_argument("--scene", default="neutron_star", help="scene name (see --list)")
    ap.add_argument("--list", action="store_true", help="list available scenes and exit")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--time", type=float, default=0.0, help="scene time (seconds) for a single frame")
    ap.add_argument("--mouse", type=float, nargs=2, default=(0.0, 0.0),
                    metavar=("MX", "MY"), help="camera/pan control, in pixel coords like iMouse")
    ap.add_argument("--frames", type=int, default=1, help="number of frames to render")
    ap.add_argument("--fps", type=float, default=30.0, help="frames per second (sets time step + GIF timing)")
    ap.add_argument("-o", "--out", default="frame.png", help="output path for a single frame")
    ap.add_argument("--out-dir", default=None, help="directory to write frame_####.png into")
    ap.add_argument("--gif", default=None, help="write the sequence as an animated GIF at this path")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--quality", default="auto",
                    choices=["auto", "low", "medium", "high", "ultra"],
                    help="LOD tier for engine/LOD-aware scenes (auto = by device)")
    ap.add_argument("--ss", type=int, default=1,
                    help="supersampling factor for anti-aliasing (render NxN, downsample)")
    args = ap.parse_args()

    wp.init()

    if args.list:
        print("Available scenes:")
        for s in list_scenes():
            print(f"  {s.name:<16} {s.description}")
        return

    scene = get_scene(args.scene)
    device = pick_device(args.device)
    from warp_shaders.lod import set_active
    tier = set_active(args.quality, device)
    ss = max(1, int(args.ss))
    aa = f"  |  {ss}x AA" if ss > 1 else ""
    print(f"scene: {scene.name}  |  device: {device}  |  quality: {tier.name}{aa}")

    def render_frame(t):
        # Supersample: render at ss x resolution, box-average down (universal AA).
        fr = scene.render(args.width * ss, args.height * ss, t, tuple(args.mouse), device)
        if ss > 1:
            h2 = (fr.shape[0] // ss) * ss
            w2 = (fr.shape[1] // ss) * ss
            fr = fr[:h2, :w2].reshape(h2 // ss, ss, w2 // ss, ss, 3).mean(axis=(1, 3))
        return fr

    if args.frames <= 1:
        frame = render_frame(args.time)
        save_png(args.out, frame)
        print(f"wrote {args.out}  ({args.width}x{args.height}, t={args.time})")
        return

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    dt = 1.0 / args.fps
    gif_frames = []
    for k in range(args.frames):
        t = k * dt
        frame = render_frame(t)
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
