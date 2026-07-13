"""Electric motor — current becomes torque, the rotor spins.

A DC motor laid open: two magnet poles (N red / S blue) straddle a rotor whose copper
windings carry current. The force on a current-carrying wire in a field (F = I L × B)
pushes the rotor around; a commutator flips the current each half-turn so the push keeps
the same sense, and it spins up. The windings glow as current flows and the whole rotor
turns. Animate with ``--frames``. See ``docs/research/38-electricity.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electric as el
from ..engine import post
from ..procedural.sdf import op_union, sd_box, sd_cylinder
from ..scene import Scene

_MAXD = 30.0


@wp.func
def _rotz(p: wp.vec3, a: float) -> wp.vec3:
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.vec3(c * p[0] + s * p[1], -s * p[0] + c * p[1], p[2])


@wp.func
def _rotor(p: wp.vec3, ang: float) -> float:
    core = sd_cylinder(wp.vec3(p[0], p[2], p[1]), 0.5, 0.42)          # rotor iron (axis z)
    bars = float(1e9)
    for k in range(6):
        a = ang + float(k) * 1.0472                                   # 60 deg apart
        q = _rotz(p, a)
        b = sd_box(q - wp.vec3(0.44, 0.0, 0.0), wp.vec3(0.06, 0.09, 0.5))
        bars = wp.min(bars, b)
    return op_union(core, bars)


@wp.func
def _map(p: wp.vec3, ang: float) -> float:
    rotor = _rotor(p, ang)
    shaft = sd_cylinder(wp.vec3(p[0], p[2], p[1]), 0.95, 0.09)        # shaft through z
    poleN = sd_box(p - wp.vec3(-0.95, 0.0, 0.0), wp.vec3(0.22, 0.7, 0.55)) - 0.02
    poleS = sd_box(p - wp.vec3(0.95, 0.0, 0.0), wp.vec3(0.22, 0.7, 0.55)) - 0.02
    base = sd_box(p - wp.vec3(0.0, -0.95, 0.0), wp.vec3(1.4, 0.09, 0.7)) - 0.02
    return op_union(op_union(rotor, shaft), op_union(op_union(poleN, poleS), base))


@wp.func
def _normal(p: wp.vec3, ang: float) -> wp.vec3:
    e = 0.0016
    dx = _map(p + wp.vec3(e, 0.0, 0.0), ang) - _map(p - wp.vec3(e, 0.0, 0.0), ang)
    dy = _map(p + wp.vec3(0.0, e, 0.0), ang) - _map(p - wp.vec3(0.0, e, 0.0), ang)
    dz = _map(p + wp.vec3(0.0, 0.0, e), ang) - _map(p - wp.vec3(0.0, 0.0, e), ang)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanfov: float,
                   ang: float, cur: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(150):
        p = eye + rd * t
        d = _map(p, ang)
        if d < 0.0008 * t + 0.0004:
            hit = 1
            break
        t += d * 0.85
        if t > _MAXD:
            break

    up_ = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    col = wp.vec3(0.015, 0.016, 0.026) * (1.0 - up_) + wp.vec3(0.03, 0.03, 0.05) * up_
    if hit == 1:
        p = eye + rd * t
        n = _normal(p, ang)
        ld = wp.normalize(wp.vec3(0.3, 0.7, 0.9))
        diff = wp.max(wp.dot(n, ld), 0.0)
        rr = wp.length(wp.vec2(p[0], p[1]))
        if p[0] < -0.6:
            base = wp.vec3(0.85, 0.22, 0.18)                          # N pole red
        elif p[0] > 0.6:
            base = wp.vec3(0.2, 0.4, 0.95)                            # S pole blue
        elif rr > 0.3 and rr < 0.55 and wp.abs(p[2]) < 0.55:
            pulse = 0.5 + 0.5 * wp.sin(wp.atan2(p[1], p[0]) * 3.0 - ang * 3.0)
            base = wp.vec3(0.85, 0.5, 0.2) * (0.4 + 0.6 * cur * pulse)  # glowing copper windings
        else:
            base = wp.vec3(0.4, 0.4, 0.45)                            # steel
        col = base * (0.14 + 0.8 * diff)
    img[i, j] = col


def _render(width, height, time, mouse, device):
    ang = time * 6.0                                                  # spins up
    cur = 0.6 + 0.4 * math.sin(time * 12.0)

    az = 0.55 + math.sin(time * 0.1) * 0.15 + float(mouse[0]) * 0.01
    el_ang = 0.24 + float(mouse[1]) * 0.005
    dist = 5.2
    eye = wp.vec3(dist * math.cos(el_ang) * math.sin(az), dist * math.sin(el_ang) + 0.3,
                  dist * math.cos(el_ang) * math.cos(az))
    tgt = wp.vec3(0.0, 0.0, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov, float(ang), float(cur)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(post.bloom(img.numpy(), threshold=1.3, strength=0.3,
                                   radius=max(2, int(min(width, height) * 0.012)),
                                   passes=3, octaves=4), mode="aces", exposure=1.05,
                        preserve_hue=True)


SCENE = Scene(
    name="electric_motor",
    description="a DC motor laid open — two magnet poles (N red / S blue) straddling a "
                "rotor whose copper windings glow with current and spin under the F=ILxB "
                "force, a commutator keeping the torque one-way. Animate with --frames.",
    renderer=_render,
)
