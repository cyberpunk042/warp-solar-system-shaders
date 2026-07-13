"""Lightning — a fractal stepped-leader bolt from storm cloud to ground.

The iconic discharge. A **dielectric-breakdown** channel (`electric.generate_bolt`:
recursive midpoint displacement + branching) leaps from the cloud base to a strike
point, blue-white and forked, flickering as each return stroke re-lights it. The
strike throws a flash across the dark storm sky and the wet ground below. Animate
over ``--frames`` to watch strike after strike walk the horizon. See
``docs/research/38-electricity.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electric as el
from ..engine import post
from ..scene import Scene

_STRIKE_EVERY = 1.3


@wp.func
def _sky(rd: wp.vec3, flash: float) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    base = wp.vec3(0.012, 0.014, 0.024) * (1.0 - up) + wp.vec3(0.028, 0.032, 0.050) * up
    # heavy storm cloud darkening the upper sky, with a little churn
    band = wp.smoothstep(0.15, 0.6, rd[1])
    churn = 0.5 + 0.5 * wp.sin(rd[0] * 7.0) * wp.sin(rd[1] * 9.0 + 1.3)
    cloud = wp.vec3(0.014, 0.014, 0.020) * (band * (0.6 + 0.4 * churn))
    return base - cloud + wp.vec3(0.10, 0.12, 0.18) * (flash * 0.14)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), pts: wp.array(dtype=wp.vec3),
                   npts: int, eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3,
                   width: int, height: int, tanfov: float, flash: float,
                   strike: wp.vec3, width_b: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    col = _sky(rd, flash)
    # ground plane at y=0, lit by the flash near the strike point
    if rd[1] < -0.002:
        t = -eye[1] / rd[1]
        if t > 0.0 and t < 200.0:
            gp = eye + rd * t
            dxz = wp.length(wp.vec2(gp[0] - strike[0], gp[2] - strike[2]))
            lit = flash * wp.exp(-dxz * dxz * 0.02)
            fog = wp.exp(-t * 0.03)
            ground = wp.vec3(0.03, 0.035, 0.05) + wp.vec3(0.5, 0.6, 0.9) * lit
            col = ground * fog + col * (1.0 - fog)

    # the bolt: dense glowing points along the fractal channel
    g = float(0.0)
    core = float(0.0)
    for k in range(npts):
        g += el.pt_glow(eye, rd, pts[k], width_b)
        core += el.pt_glow(eye, rd, pts[k], width_b * 0.4)
    glow = wp.clamp(g, 0.0, 3.0)
    boltcol = wp.vec3(0.45, 0.62, 1.0) * (glow * flash * 1.4)
    boltcol += wp.vec3(0.9, 0.95, 1.0) * (wp.clamp(core, 0.0, 4.0) * flash * 2.2)
    img[i, j] = col + boltcol


def _render(width, height, time, mouse, device):
    strike_idx = int(math.floor(time / _STRIKE_EVERY))
    local = time - float(strike_idx) * _STRIKE_EVERY
    # return-stroke flash: a bright onset with a couple of re-strike flickers, decaying
    flash = math.exp(-local * 4.5) * (0.6 + 0.4 * abs(math.sin(local * 42.0)))
    flash = min(1.0, flash + 0.04)

    rng = np.random.RandomState(strike_idx * 2654435761 & 0x7FFFFFFF)
    sx = float(rng.uniform(-2.2, 2.2))
    sz = float(rng.uniform(-1.2, 1.2))
    top = (sx * 0.3 + float(rng.uniform(-0.6, 0.6)), 6.4, sz * 0.3)
    strike = wp.vec3(sx, 0.0, sz)
    pts = el.generate_bolt(top, (sx, 0.0, sz), seed=strike_idx, gens=6,
                           jitter=1.05, branch_prob=0.55)
    parr, npts = el.upload_points(pts, device)

    az = 0.35 + math.sin(time * 0.08) * 0.15 + float(mouse[0]) * 0.01
    el_ang = 0.12 + float(mouse[1]) * 0.005
    dist = 11.0
    eye = wp.vec3(dist * math.cos(el_ang) * math.sin(az),
                  2.6 + dist * math.sin(el_ang),
                  dist * math.cos(el_ang) * math.cos(az))
    tgt = wp.vec3(0.0, 3.0, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, parr, npts, eye, fwd, right, up, width, height, tanfov,
                      float(flash), strike, 0.055], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.6, radius=r, passes=4, octaves=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="lightning",
    description="a fractal stepped-leader lightning bolt from storm cloud to ground — a "
                "branching dielectric-breakdown channel, blue-white and forked, flashing "
                "across a dark storm sky and lighting the wet ground. Animate with --frames "
                "to watch strike after strike walk the horizon.",
    renderer=_render,
)
