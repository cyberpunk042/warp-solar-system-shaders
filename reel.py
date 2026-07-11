#!/usr/bin/env python3
"""Render a showcase **reel** — several scenes stitched into one video.

    python reel.py -o out/showcase.mp4 --width 960 --height 540 --fps 24
    python reel.py -o out/mini.webp --width 480 --height 270 --fps 20 --preset mini

A reel is a list of :class:`Clip` s (scene · seconds · scene-time span · post
look · optional Ken-Burns push-in). Each clip renders its own frames (scenes
animate over their `time`), consecutive clips are joined by a **crossfade**
dissolve, and the whole thing is written with :func:`engine.video.write_video`
(MP4 via ffmpeg when present, else WebP). The default playlist tours the engine's
hero scenes; `--preset mini` is a quick 3-clip smoke reel.
"""

from __future__ import annotations

import argparse
import dataclasses
import os

import numpy as np
import warp as wp

from warp_shaders.engine import post
from warp_shaders.engine.video import crossfade, write_video
from warp_shaders.scene import get_scene


@dataclasses.dataclass
class Clip:
    scene: str
    seconds: float = 3.0
    t0: float = 0.0
    t1: float | None = None          # scene-time at clip end (default t0 + seconds)
    look: str = "clean"
    zoom: tuple = (1.0, 1.0)         # Ken-Burns start/end zoom (1.0 = none)
    fade: float = 0.5               # crossfade seconds joining this clip to the NEXT


# curated tour of the engine's hero scenes
SHOWCASE = [
    Clip("ss_flyby", seconds=6.0, t0=0.0, t1=6.0, look="cinematic"),
    Clip("earth_v2", seconds=3.5, look="cinematic", zoom=(1.0, 1.12)),
    Clip("black_hole", seconds=3.5, look="film"),
    Clip("ss_nebula", seconds=3.0, look="dreamy", zoom=(1.08, 1.0)),
    Clip("se_living", seconds=3.0, look="cinematic", zoom=(1.0, 1.1)),
    Clip("meadow", seconds=3.0, look="film"),
    Clip("aurora", seconds=3.0, look="dreamy", zoom=(1.0, 1.1)),
    Clip("terrain", seconds=3.0, look="cinematic", zoom=(1.1, 1.0)),
]

MINI = [
    Clip("ss_flyby", seconds=2.5, t0=0.0, t1=3.0, look="cinematic"),
    Clip("earth_v2", seconds=2.0, look="cinematic", zoom=(1.0, 1.12)),
    Clip("terrain", seconds=2.0, look="film", zoom=(1.1, 1.0)),
]

PRESETS = {"showcase": SHOWCASE, "mini": MINI}


def _to_u8(img):
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _ken_burns(img, zoom, cx=0.5, cy=0.5):
    """Crop a `1/zoom` window around (cx, cy) and scale back to full — a slow
    push-in that adds motion even to near-static frames."""
    from PIL import Image
    if abs(zoom - 1.0) < 1e-3:
        return img
    h, w = img.shape[:2]
    cw, ch = max(2, int(w / zoom)), max(2, int(h / zoom))
    x0 = int((w - cw) * cx)
    y0 = int((h - ch) * cy)
    pim = Image.fromarray(_to_u8(img))
    crop = pim.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.LANCZOS)
    return np.asarray(crop, np.float32) / 255.0


def render_reel(clips, width, height, fps=24, device="cpu"):
    frames = []
    prev_tail = None
    prev_fade = 0.0                          # the FROM-clip's fade joins it to this one
    for ci, clip in enumerate(clips):
        sc = get_scene(clip.scene)
        n = max(1, int(round(clip.seconds * fps)))
        t1 = clip.t1 if clip.t1 is not None else clip.t0 + clip.seconds
        clip_frames = []
        for k in range(n):
            f = k / max(n - 1, 1)
            st = clip.t0 + (t1 - clip.t0) * f
            img = np.clip(sc.render(width, height, st, (0.0, 0.0), device), 0.0, 1.0)
            if clip.look != "clean":
                img = post.apply_look(img, clip.look, seed=k)
            z = clip.zoom[0] + (clip.zoom[1] - clip.zoom[0]) * f
            img = _ken_burns(np.asarray(img, np.float32), z)
            clip_frames.append(_to_u8(img))
            print(f"  [{ci + 1}/{len(clips)}] {clip.scene}  frame {k + 1}/{n}",
                  end="\r", flush=True)
        # dissolve from the previous clip's tail into this clip's head, using the
        # PREVIOUS clip's fade (it declares how it joins to the next clip)
        if prev_tail is not None and prev_fade > 0.0:
            fn = max(1, int(round(prev_fade * fps)))
            for bf in crossfade(prev_tail, clip_frames[0], fn):
                frames.append(_to_u8(bf / 255.0))
        frames.extend(clip_frames)
        prev_tail = clip_frames[-1].astype(np.float32)
        prev_fade = clip.fade
    print()
    return frames


def main():
    ap = argparse.ArgumentParser(description="Render a multi-scene showcase reel.")
    ap.add_argument("-o", "--out", default="out/showcase.mp4",
                    help="output video (.mp4/.webm via ffmpeg, else .webp/.gif)")
    ap.add_argument("--preset", default="showcase", choices=list(PRESETS))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--quality", default="medium")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    wp.init()
    from warp_shaders.lod import set_active
    device = "cuda" if (args.device == "auto" and wp.get_cuda_device_count() > 0) \
        else ("cuda" if args.device == "cuda" else "cpu")
    set_active(args.quality, device)

    clips = PRESETS[args.preset]
    print(f"reel: {args.preset}  {len(clips)} clips  {args.width}x{args.height} @ {args.fps}fps"
          f"  device={device}")
    frames = render_reel(clips, args.width, args.height, args.fps, device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    wrote = write_video(frames, args.out, fps=args.fps)
    print(f"wrote {wrote}  ({len(frames)} frames, {len(frames) / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
