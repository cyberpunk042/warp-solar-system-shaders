"""A generation ship — a world in a bottle between the stars.

A rotating-habitat starship carrying a whole ecosystem on a centuries-long voyage:
its inhabitants are born, live and die aboard, never seeing departure or arrival. A
long hull with spinning habitat **rings** (windows lit), a glowing fusion engine
trailing behind, crossing interstellar space. See
``docs/research/29-megastructures-and-far-future.md``. --frames spins the rings.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.noise import fbm3
from ..subatomic.field import sd_capsule, void
from ..scene import Scene


@wp.func
def _ring(p: wp.vec3, zc: float, R: float, tube: float) -> float:
    q = wp.vec2(wp.length(wp.vec2(p[0], p[1])) - R, p[2] - zc)
    return wp.length(q) - tube


@wp.func
def _map(p: wp.vec3) -> float:
    hull = sd_capsule(p, wp.vec3(0.0, 0.0, -1.5), wp.vec3(0.0, 0.0, 1.7), 0.24)
    spine = sd_capsule(p, wp.vec3(0.0, 0.0, -1.9), wp.vec3(0.0, 0.0, 2.0), 0.06)
    r1 = _ring(p, 0.5, 0.72, 0.07)
    r2 = _ring(p, -0.3, 0.72, 0.07)
    dome = wp.length(p - wp.vec3(0.0, 0.0, 1.75)) - 0.3
    return wp.min(wp.min(hull, spine), wp.min(wp.min(r1, r2), dome))


@wp.kernel
def ship_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                rgt: wp.vec3, upv: wp.vec3, aspect: float, spin: float,
                time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    rd = wp.normalize(fwd * 1.7 + rgt * (u * aspect) + upv * v)
    sun = wp.normalize(wp.vec3(0.5, 0.6, 0.3))

    col = void(rd)
    t = float(0.1)
    hit = int(0)
    hp = wp.vec3(0.0, 0.0, 0.0)
    for _ in range(120):
        p = eye + rd * t
        d = _map(p)
        if d < 0.003:
            hit = 1
            hp = p
            break
        t += wp.max(d * 0.8, 0.004)
        if t > 14.0:
            break

    if hit == 1:
        e = 0.004
        n = wp.normalize(wp.vec3(
            _map(hp + wp.vec3(e, 0.0, 0.0)) - _map(hp - wp.vec3(e, 0.0, 0.0)),
            _map(hp + wp.vec3(0.0, e, 0.0)) - _map(hp - wp.vec3(0.0, e, 0.0)),
            _map(hp + wp.vec3(0.0, 0.0, e)) - _map(hp - wp.vec3(0.0, 0.0, e))))
        dif = wp.max(wp.dot(n, sun), 0.0)
        metal = wp.vec3(0.32, 0.34, 0.4) * (0.25 + 0.8 * dif)
        rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
        metal = metal + wp.vec3(0.5, 0.6, 0.8) * rim * 0.4
        # window lights on the rotating rings
        rr = wp.length(wp.vec2(hp[0], hp[1]))
        if rr > 0.6:
            ang = wp.atan2(hp[1], hp[0]) + spin
            win = wp.smoothstep(0.5, 0.9, wp.sin(ang * 40.0))
            metal = metal + wp.vec3(1.0, 0.85, 0.5) * win * 0.9
        col = metal
    else:
        # fusion engine plume behind the ship (additive glow along -z axis)
        rxy2 = rd[0] * rd[0] + rd[1] * rd[1]
        ts = -(eye[0] * rd[0] + eye[1] * rd[1]) / (rxy2 + 1e-6)
        if ts > 0.0:
            pz = eye[2] + rd[2] * ts
            cxy = wp.length(wp.vec2(eye[0] + rd[0] * ts, eye[1] + rd[1] * ts))
            if pz < -1.8:
                fall = wp.exp((pz + 1.8) * 0.8)               # fades behind
                width_p = 0.12 + (-1.8 - pz) * 0.08
                glow = wp.exp(-(cxy / width_p) * (cxy / width_p))
                col = col + wp.vec3(0.5, 0.75, 1.0) * glow * fall * 1.6

    img[i, j] = col


def _render(width, height, time, mouse, device):
    eye = wp.vec3(3.8, 1.5, 2.4)
    tgt = wp.vec3(0.0, 0.0, -0.2)
    fwd = wp.normalize(tgt - eye)
    rgt = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    upv = wp.cross(rgt, fwd)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(ship_kernel, dim=(height, width),
              inputs=[img, eye, fwd, rgt, upv, float(width / height), float(time * 0.6),
                      float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.011))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="generation_ship",
    description="A generation ship — a long hull with spinning habitat rings (windows "
                "lit), a glowing fusion engine trailing behind, carrying a whole "
                "ecosystem across interstellar space on a centuries-long voyage. "
                "--frames spins the rings.",
    renderer=_render,
)
