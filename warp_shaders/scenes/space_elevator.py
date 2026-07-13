"""A space elevator — a cable to orbit.

A cable anchored at the equator and run past **geostationary** orbit (~36 000 km),
held taut by a counterweight so centrifugal force balances gravity: climbers ride to
orbit without rockets. Here the cable rises from a blue planet's limb to its
counterweight, a climber ascending, against the stars. See
``docs/research/29-megastructures-and-far-future.md``. --frames climbs the elevator.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..procedural.noise import fbm3
from ..subatomic.field import sd_capsule, void
from ..scene import Scene

_RP = 2.0
_ANCHOR = wp.constant(wp.vec3(0.0, 2.0, 0.0))
_TOP = wp.constant(wp.vec3(0.0, 5.4, 0.0))


@wp.func
def _sdbox(p: wp.vec3, c: wp.vec3, b: wp.vec3) -> float:
    d = wp.vec3(wp.abs(p[0] - c[0]) - b[0], wp.abs(p[1] - c[1]) - b[1],
                wp.abs(p[2] - c[2]) - b[2])
    return wp.length(wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0))) \
        + wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0)


@wp.kernel
def elev_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                rgt: wp.vec3, upv: wp.vec3, aspect: float, climb: float,
                time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    rd = wp.normalize(fwd * 1.6 + rgt * (u * aspect) + upv * v)
    sun = wp.normalize(wp.vec3(0.7, 0.35, 0.55))

    col = void(rd)
    # planet (analytic sphere)
    gp = _rs(eye, rd, _RP)
    tp = float(1e30)
    if gp[0] > 0.0 and gp[0] < 1.0e29:
        tp = gp[0]

    # cable + counterweight + climber via a short sphere-trace
    climber = wp.vec3(0.0, 2.05 + climb * 3.15, 0.0)
    t = float(0.1)
    tstruct = float(1e30)
    sid = int(0)
    for _ in range(90):
        p = eye + rd * t
        dc = sd_capsule(p, _ANCHOR, _TOP, 0.012)
        dw = _sdbox(p, wp.vec3(0.0, 5.4, 0.0), wp.vec3(0.14, 0.1, 0.14))
        dm = _sdbox(p, climber, wp.vec3(0.05, 0.06, 0.05))
        d = wp.min(dc, wp.min(dw, dm))
        if d < 0.004:
            tstruct = t
            if d == dc:
                sid = 1
            elif d == dw:
                sid = 2
            else:
                sid = 3
            break
        t += wp.max(d * 0.7, 0.004)
        if t > 20.0 or t > tp:
            break

    if tstruct < tp and tstruct < 1e29:
        p = eye + rd * tstruct
        if sid == 1:                                    # cable: thin sunlit thread
            hl = wp.max(wp.dot(wp.normalize(p - _ANCHOR), sun), 0.0)
            col = wp.vec3(0.5, 0.55, 0.6) * (0.3 + 0.7 * hl) + wp.vec3(0.8, 0.85, 1.0) * 0.2
        elif sid == 2:                                  # counterweight station
            col = wp.vec3(0.6, 0.62, 0.68) + wp.vec3(0.9, 0.7, 0.3) * 0.2
        else:                                           # climber (lit capsule car)
            col = wp.vec3(1.0, 0.85, 0.5) * 1.3
    elif tp < 1e29:
        p = eye + rd * tp
        n = wp.normalize(p)
        dif = wp.max(wp.dot(n, sun), 0.0)
        land = fbm3(n * 3.0, 5)
        ocean = wp.vec3(0.04, 0.15, 0.38)
        green = wp.vec3(0.15, 0.35, 0.15)
        surf = ocean * (1.0 - wp.smoothstep(0.5, 0.58, land)) \
            + green * wp.smoothstep(0.5, 0.58, land)
        cloud = wp.smoothstep(0.55, 0.75, fbm3(n * 4.0 + wp.vec3(time * 0.02, 0.0, 0.0), 4))
        surf = surf * (1.0 - cloud) + wp.vec3(0.9, 0.92, 0.95) * cloud
        night = wp.vec3(1.0, 0.8, 0.4) * wp.pow(1.0 - dif, 3.0) * 0.25 * land
        col = surf * (0.08 + 1.05 * dif) + night
        # atmosphere rim
        rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 2.5)
        col = col + wp.vec3(0.3, 0.55, 1.0) * rim * dif * 0.8

    # atmosphere halo for grazing rays that miss the surface
    if tp > 1e29:
        ga = _rs(eye, rd, _RP * 1.12)
        if ga[0] > 0.0 and ga[0] < 1.0e29:
            pa = eye + rd * ga[0]
            na = wp.normalize(pa)
            glow = wp.max(wp.dot(na, sun), 0.0)
            col = col + wp.vec3(0.3, 0.55, 1.0) * glow * 0.5

    img[i, j] = col


def _render(width, height, time, mouse, device):
    climb = 0.5 + 0.5 * float(np.sin(time * 0.4))
    eye = wp.vec3(6.4, 3.1, 3.6)
    tgt = wp.vec3(0.0, 3.0, 0.0)
    fwd = wp.normalize(tgt - eye)
    rgt = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    upv = wp.cross(rgt, fwd)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(elev_kernel, dim=(height, width),
              inputs=[img, eye, fwd, rgt, upv, float(width / height), float(climb),
                      float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="space_elevator",
    description="A space elevator — a cable anchored at a blue planet's equator rising "
                "past geostationary orbit to a counterweight, a lit climber ascending, "
                "against the stars. --frames climbs the elevator.",
    renderer=_render,
)
