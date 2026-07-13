"""Transformer — two coils coupled through an iron core by changing flux.

Faraday's law made visible: AC in the **primary** winding drives a changing magnetic
**flux** that circulates the laminated iron core; that changing flux induces AC in the
**secondary**, stepping the voltage by the turns ratio. The flux is drawn as a bright
loop racing around the core; the two windings pulse with their (phase-shifted) currents.
Animate with ``--frames``. See ``docs/research/38-electricity.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electric as el
from ..engine import post
from ..procedural.sdf import op_subtract, op_union, sd_box, sd_cylinder
from ..scene import Scene

_MAXD = 34.0


@wp.func
def _core(p: wp.vec3) -> float:
    outer = sd_box(p, wp.vec3(1.25, 1.0, 0.22)) - 0.02
    window = sd_box(p, wp.vec3(0.55, 0.62, 0.5))          # the window of an O core
    return op_subtract(outer, window)


@wp.func
def _coil(p: wp.vec3, cx: float) -> float:
    # a stack of winding rings around a vertical leg (a fat cylinder that bulges out
    # past the thinner core so it reads as copper windings)
    q = p - wp.vec3(cx, 0.0, 0.0)
    return sd_cylinder(q, 0.72, 0.40) - 0.01


@wp.func
def _map(p: wp.vec3) -> float:
    core = _core(p)
    prim = _coil(p, -0.9)
    sec = _coil(p, 0.9)
    return op_union(core, op_union(prim, sec))


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.0016
    dx = _map(p + wp.vec3(e, 0.0, 0.0)) - _map(p - wp.vec3(e, 0.0, 0.0))
    dy = _map(p + wp.vec3(0.0, e, 0.0)) - _map(p - wp.vec3(0.0, e, 0.0))
    dz = _map(p + wp.vec3(0.0, 0.0, e)) - _map(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), fpts: wp.array(dtype=wp.vec3), nf: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3,
                   width: int, height: int, tanfov: float, time: float, prim_i: float,
                   sec_i: float, fluxlvl: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(150):
        p = eye + rd * t
        d = _map(p)
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
        n = _normal(p)
        ld = wp.normalize(wp.vec3(0.3, 0.7, 0.9))
        diff = wp.max(wp.dot(n, ld), 0.0)
        rr = wp.length(wp.vec2(p[0] + 0.9, p[2]))
        rr2 = wp.length(wp.vec2(p[0] - 0.9, p[2]))
        if rr < 0.44 and wp.abs(p[1]) < 0.74:
            ring = 0.5 + 0.5 * wp.sin(p[1] * 40.0)                    # winding ribs
            base = wp.vec3(0.8, 0.45, 0.18) * (0.4 + 0.6 * prim_i * ring)   # primary (copper)
        elif rr2 < 0.44 and wp.abs(p[1]) < 0.74:
            ring = 0.5 + 0.5 * wp.sin(p[1] * 40.0)
            base = wp.vec3(0.75, 0.5, 0.22) * (0.4 + 0.6 * sec_i * ring)    # secondary
        else:
            base = wp.vec3(0.32, 0.33, 0.36)                          # laminated iron core
        col = base * (0.16 + 0.8 * diff)

    # flux loop racing around the core (bright travelling points)
    g = float(0.0)
    for k in range(nf):
        g += el.pt_glow(eye, rd, fpts[k], 0.05)
    col += wp.vec3(0.5, 0.75, 1.0) * (wp.clamp(g, 0.0, 3.0) * fluxlvl * 1.3)
    img[i, j] = col


def _render(width, height, time, mouse, device):
    prim_i = 0.5 + 0.5 * math.sin(time * 6.0)
    sec_i = 0.5 + 0.5 * math.sin(time * 6.0 - 2.6)                    # induced, phase-shifted
    fluxlvl = 0.5 + 0.5 * abs(math.sin(time * 6.0))

    # sample a moving bright arc of the flux loop threading both windows of the core
    phase = time * 3.0
    pts = []
    for s in range(10):
        u = (s / 10.0) * 2.0 * math.pi + phase
        # a rounded-rectangle path around the left window centre (x=-0.9)
        pts.append([-0.9 + 0.42 * math.cos(u), 0.62 * math.sin(u), 0.0])
    fpts = np.asarray(pts, dtype=np.float32)
    fpar, nf = el.upload_points(fpts, device)

    az = 0.6 + math.sin(time * 0.1) * 0.15 + float(mouse[0]) * 0.01
    el_ang = 0.2 + float(mouse[1]) * 0.005
    dist = 5.6
    eye = wp.vec3(dist * math.cos(el_ang) * math.sin(az), dist * math.sin(el_ang) + 0.2,
                  dist * math.cos(el_ang) * math.cos(az))
    tgt = wp.vec3(0.0, 0.0, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, fpar, nf, eye, fwd, right, up, width, height, tanfov, time,
                      float(prim_i), float(sec_i), float(fluxlvl)], device=device)
    wp.synchronize_device(device)
    return post.tonemap(post.bloom(img.numpy(), threshold=1.1, strength=0.4,
                                   radius=max(2, int(min(width, height) * 0.015)),
                                   passes=3, octaves=4), mode="aces", exposure=1.05,
                        preserve_hue=True)


SCENE = Scene(
    name="transformer",
    description="a transformer — a primary and secondary coil on a laminated iron core, "
                "AC in the primary driving a changing flux (a bright loop racing the core) "
                "that induces a phase-shifted current in the secondary. Faraday's law made "
                "visible. Animate with --frames.",
    renderer=_render,
)
