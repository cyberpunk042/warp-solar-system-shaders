"""Tests for engine.video — animated container output.

Run: `python -m tests.test_video` (or under pytest). WebP/GIF/APNG go through
Pillow (always available); the MP4 path is exercised only when imageio-ffmpeg is
installed (skipped otherwise, so CI without ffmpeg still passes).
"""

import os
import tempfile

import numpy as np

from warp_shaders.engine import video as V


def _frames(n=6, h=16, w=24):
    out = []
    for k in range(n):
        f = np.zeros((h, w, 3), np.float32)
        f[..., 0] = k / max(n - 1, 1)                 # animate red over the clip
        f[..., 2] = np.linspace(0, 1, w)[None, :]     # a horizontal blue ramp
        out.append(f)
    return out


def _count(path):
    from PIL import Image
    with Image.open(path) as im:
        return getattr(im, "n_frames", 1)


def test_pillow_containers():
    frames = _frames(6)
    d = tempfile.mkdtemp()
    for ext in ("webp", "gif", "apng"):
        p = os.path.join(d, f"clip.{ext}")
        wrote = V.write_video(frames, p, fps=12)
        assert wrote == p and os.path.getsize(p) > 0
        assert _count(p) == 6                         # every frame stored


def test_crossfade():
    a = np.zeros((4, 4, 3), np.float32)
    b = np.ones((4, 4, 3), np.float32)
    mids = list(V.crossfade(a, b, 3))
    assert len(mids) == 3
    # strictly increasing blend, bounded strictly inside (0, 1)
    means = [float(m.mean()) for m in mids]
    assert means == sorted(means)
    assert 0.0 < means[0] and means[-1] < 1.0


def test_save_frames():
    d = tempfile.mkdtemp()
    n = V.save_frames(_frames(5), d)
    assert n == 5
    assert len([f for f in os.listdir(d) if f.endswith(".png")]) == 5


def test_mp4_when_available():
    if not V._have_imageio():
        print("  (mp4 path skipped — imageio-ffmpeg not installed)")
        return
    import imageio.v2 as iio
    d = tempfile.mkdtemp()
    p = os.path.join(d, "clip.mp4")
    wrote = V.write_video(_frames(8, 32, 48), p, fps=24)
    assert wrote == p and os.path.getsize(p) > 0
    back = [f for f in iio.get_reader(p)]
    assert len(back) == 8                             # all frames survive the encode
    assert back[0].shape[2] == 3


if __name__ == "__main__":
    test_pillow_containers()
    print("  pillow containers (webp/gif/apng, frame counts): OK")
    test_crossfade()
    print("  crossfade (monotone blend, bounded): OK")
    test_save_frames()
    print("  save_frames (png sequence): OK")
    test_mp4_when_available()
    print("  mp4 encode/decode roundtrip: OK")
    print("ALL PASSED")
