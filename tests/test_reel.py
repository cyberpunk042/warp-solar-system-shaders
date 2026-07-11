"""Smoke test for reel.py — clip stitching + Ken-Burns + frame budget.

Run: `python -m tests.test_reel` (or under pytest). Renders a tiny 2-clip reel of
a cheap scene so the stitching math (clip frames + crossfade frames) is exercised
end to end without a heavy render.
"""

import numpy as np
import warp as wp

import reel
from reel import Clip, _ken_burns, render_reel

wp.init()


def test_ken_burns():
    img = np.random.default_rng(0).random((20, 30, 3)).astype(np.float32)
    assert _ken_burns(img, 1.0) is img                    # no-op at zoom 1
    z = _ken_burns(img, 1.5)
    assert z.shape == img.shape and np.all(np.isfinite(z)) and z.max() <= 1.0


def test_reel_frame_budget():
    from warp_shaders.lod import set_active
    set_active("low", "cpu")
    fps = 4.0
    clips = [Clip("pbr_demo", seconds=0.5, look="clean", fade=0.5),
             Clip("pbr_demo", seconds=0.5, look="cinematic", zoom=(1.0, 1.2), fade=0.0)]
    frames = render_reel(clips, 48, 32, fps=fps, device="cpu")
    # 2 + 2 clip frames, plus a 2-frame crossfade joining them
    assert len(frames) == 6
    for f in frames:
        assert f.shape == (32, 48, 3) and f.dtype == np.uint8


if __name__ == "__main__":
    test_ken_burns()
    print("  ken burns (no-op at 1.0, in-range zoom): OK")
    test_reel_frame_budget()
    print("  reel frame budget (clips + crossfade): OK")
    print("ALL PASSED")
