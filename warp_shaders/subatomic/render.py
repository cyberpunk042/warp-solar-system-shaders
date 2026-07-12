"""Shared render harness for the sub-atomic scenes — an orbiting camera and the
common HDR finish (bloom + ACES tonemap) so every particle reads consistently.
"""

import math

from ..engine import post
from ..engine.uniforms import make_camera


def orbit_camera(width, height, time, mouse, dist=4.0, fov=42.0,
                 target=(0.0, 0.0, 0.0), auto=0.25, el0=0.35):
    """A slowly auto-orbiting camera; the mouse takes over azimuth/elevation."""
    az = auto * time + 0.6
    el = el0
    if mouse[0] > 0.0 or mouse[1] > 0.0:
        az = (float(mouse[0]) / max(width, 1)) * 6.2831853
        el = (float(mouse[1]) / max(height, 1) - 0.5) * 3.0
    ce = math.cos(el)
    eye = (dist * ce * math.sin(az), dist * math.sin(el), dist * ce * math.cos(az))
    return make_camera(eye, target, fov_deg=fov, aspect=width / height)


def finish(hdr, width, height, threshold=1.1, strength=0.55, exposure=1.06,
           passes=3):
    """Common HDR → display finish for particle fields (glowing emission)."""
    r = max(2, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=threshold, strength=strength, radius=r,
                     passes=passes)
    return post.tonemap(hdr, mode="aces", exposure=exposure)
