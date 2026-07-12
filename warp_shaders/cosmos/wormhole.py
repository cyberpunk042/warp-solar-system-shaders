"""Ellis wormhole — a throat that shows *another universe*, gravitationally lensed.

An impact-parameter model of the Ellis drainhole (see
``docs/research/19-extraordinary-cosmos.md``): the throat is a sphere of radius
`a` at the origin. Per ray we take the closest approach `b` to the centre:

- rays that **miss** (`b ≥ a`) are gravitationally **lensed** — bent toward the
  throat by ``~a²/b²`` — and sample **background A** (this universe), so A's sky
  wraps into an Einstein ring around the rim;
- rays that **enter** (`b < a`) cross the throat and sample **background B** (the
  other universe), fish-eye-mapped so B's whole sky is compressed into the disc;
- a bright **exotic-matter rim** glows where `b ≈ a`.

Two procedural nebula+starfield skies stand in for the two universes. This is a
screen-space approximation of the geodesics — no ODE integration — but it gives
the iconic *Interstellar* portal cheaply.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3


@wp.func
def _neb(rd: wp.vec3, tint: wp.vec3, seed: float) -> wp.vec3:
    """A procedural nebula sky (direction -> colour) tinted by `tint`, plus stars."""
    n = fbm3(rd * 2.1 + wp.vec3(seed, seed * 1.7, seed * 0.5), 5)
    fil = fbm3(rd * 5.5 + wp.vec3(seed * 2.3, seed * 0.3, seed), 4)
    d = wp.clamp(n * 1.5 - 0.35, 0.0, 1.0)
    cloud = wp.pow(d, 1.8) * (0.5 + 0.7 * fil)
    glow = wp.cw_mul(tint, wp.vec3(cloud, cloud * 0.82, cloud * 0.7))
    return glow * 0.9 + stars(rd)


@wp.func
def _bg_a(rd: wp.vec3) -> wp.vec3:
    return _neb(rd, wp.vec3(0.40, 0.55, 1.05), 3.0)      # cool blue universe


@wp.func
def _bg_b(rd: wp.vec3) -> wp.vec3:
    return _neb(rd, wp.vec3(1.05, 0.55, 0.28), 11.0)     # warm amber universe


@wp.kernel
def wormhole_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, center: wp.vec3,
                    a: float, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)
    rob = ro - center

    # closest approach of the ray to the throat centre
    t_ca = -wp.dot(rob, rd)
    ca = rob + rd * t_ca
    b = wp.length(ca)
    outward = wp.normalize(ca + wp.vec3(1.0e-5, 0.0, 0.0))   # radial, centre->ray

    col = wp.vec3(0.0, 0.0, 0.0)
    if t_ca <= 0.0:
        col = _bg_a(rd)                                      # throat is behind us
    elif b < a:
        # entered the throat -> the other universe, fish-eyed (rim -> B's horizon)
        f = b / a
        emergent = wp.normalize(rd + outward * (f * f * 1.4))
        col = _bg_b(emergent)
    else:
        # missed -> gravitational lensing of this universe (bend toward throat)
        defl = a * a / (b * b)
        bent = wp.normalize(rd - outward * defl * 1.3)
        col = _bg_a(bent)

    # exotic-matter rim: a bright ring where the ray grazes the throat
    if t_ca > 0.0:
        rim = wp.exp(-((b - a) * (b - a)) / (0.03 * a * a + 1.0e-5))
        swirl = 0.6 + 0.4 * fbm3(outward * 6.0 + wp.vec3(0.0, time * 0.5, 0.0), 3)
        col = col + wp.vec3(0.7, 0.9, 1.0) * (rim * 1.6 * swirl)

    img[i, j] = col


def render_wormhole(width, height, time, mouse, device, a=1.6):
    """Render one frame of an Ellis wormhole (throat radius `a`) at the origin."""
    az = 0.5 + time * 0.06 + float(mouse[0]) * 0.01
    elev = 0.12 + float(mouse[1]) * 0.01
    dist = 7.5
    eye = (math.sin(az) * dist * math.cos(elev), math.sin(elev) * dist,
           math.cos(az) * dist * math.cos(elev))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=40.0, aspect=width / height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(wormhole_kernel, dim=(height, width),
              inputs=[img, cam, wp.vec3(0.0, 0.0, 0.0), float(a), float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.1)
