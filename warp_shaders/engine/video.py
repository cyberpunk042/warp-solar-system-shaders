"""Frame-sequence output — animated containers (MP4 / WebP / GIF / APNG).

`render.py` and the reel turn a scene into a list of ``(H, W, 3)`` frames; this
module writes them to a video the extension chooses:

- ``.mp4`` / ``.mkv`` / ``.webm`` — H.264 / VP9 via **imageio + imageio-ffmpeg**
  (a self-contained static ffmpeg; ``pip install imageio-ffmpeg``). If that stack
  is missing, the writer degrades to an animated **WebP** beside the requested
  path and tells you — so a machine without ffmpeg still produces a video.
- ``.webp`` / ``.gif`` / ``.apng`` — animated stills via **Pillow** (always
  available, no extra deps). WebP is the best quality/size of the three.

Frames may be float ``[0, 1]`` or ``uint8``; they are clamped + quantized once.
`fps` sets playback timing. Returns the path actually written.
"""

from __future__ import annotations

import numpy as np

_VIDEO_EXT = (".mp4", ".mkv", ".webm", ".mov", ".avi")


def _to_u8(frame) -> np.ndarray:
    a = np.asarray(frame)
    if a.dtype == np.uint8:
        return a
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _have_imageio() -> bool:
    try:
        import imageio.v2  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
        return True
    except Exception:
        return False


def _write_ffmpeg(frames, path: str, fps: float, quality: int) -> str:
    import imageio.v2 as imageio
    # even dimensions required by yuv420p H.264
    h, w = frames[0].shape[:2]
    macro = 16
    kw = dict(fps=float(fps), quality=int(quality), macro_block_size=macro)
    with imageio.get_writer(path, **kw) as wr:
        for f in frames:
            wr.append_data(_to_u8(f))
    return path


def _write_pillow(frames, path: str, fps: float) -> str:
    from PIL import Image
    imgs = [Image.fromarray(_to_u8(f), mode="RGB") for f in frames]
    dur = max(1, int(round(1000.0 / max(fps, 1e-3))))
    fmt = None
    low = path.lower()
    if low.endswith(".gif"):
        fmt = "GIF"
    elif low.endswith(".webp"):
        fmt = "WEBP"
    elif low.endswith(".apng") or low.endswith(".png"):
        fmt = "PNG"
    save_kw = dict(save_all=True, append_images=imgs[1:], duration=dur, loop=0)
    if fmt == "WEBP":
        save_kw["quality"] = 90
        save_kw["method"] = 4
    imgs[0].save(path, format=fmt, **save_kw)
    return path


def write_video(frames, path: str, fps: float = 30.0, quality: int = 8) -> str:
    """Write `frames` (an iterable of ``(H, W, 3)`` arrays) to `path`.

    The extension selects the container. Video containers (.mp4/.webm/...) use
    ffmpeg via imageio when installed and otherwise fall back to an animated
    ``.webp`` next to `path`. Returns the path that was actually written."""
    frames = [np.asarray(f) for f in frames]
    if not frames:
        raise ValueError("write_video: no frames")
    low = path.lower()
    if low.endswith(_VIDEO_EXT):
        if _have_imageio():
            return _write_ffmpeg(frames, path, fps, quality)
        alt = path.rsplit(".", 1)[0] + ".webp"
        print(f"[video] imageio-ffmpeg not available; writing {alt} instead of {path}")
        return _write_pillow(frames, alt, fps)
    return _write_pillow(frames, path, fps)


def save_frames(frames, out_dir: str, prefix: str = "frame") -> int:
    """Write a numbered PNG per frame into `out_dir`; returns the count."""
    import os

    from .imageio import save_png
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for k, f in enumerate(frames):
        save_png(os.path.join(out_dir, f"{prefix}_{k:04d}.png"), f)
        n += 1
    return n


def crossfade(a, b, steps: int):
    """Yield `steps` frames blending frame `a` into frame `b` (linear alpha)."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    for k in range(steps):
        t = (k + 1) / (steps + 1)
        yield a * (1.0 - t) + b * t
