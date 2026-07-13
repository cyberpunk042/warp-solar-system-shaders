"""Meson renderer — a **quark + antiquark** bound by a single gluon flux tube.

Mesons are the two-body hadrons: one quark and one antiquark, colour + anti-colour
(red + anti-red = colour-neutral), held together by one QCD flux string. This
renders the pair orbiting a common centre inside a confinement bag, reusing the
same emission primitives as the nucleon (``field.quark_emit`` / ``tube_emit`` /
``bag_glow``). Heavier quarkonia (cc̄, bb̄) are drawn more compact and brighter —
the heavy quarks sit deeper in the potential. See ``docs/research/21-standard-model.md``.
"""

import numpy as np
import warp as wp

from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import value3
from . import render as _render_mod
from .field import bag_glow, flavor_color, quark_emit, tube_emit, void

_BOUND = 1.5
_TUBE_R = 0.11


@wp.func
def _mpos(k: int, time: float, sep: float) -> wp.vec3:
    """Quark (k=0) and antiquark (k=1) on opposite ends of a slowly tumbling,
    rattling dumbbell."""
    ang = float(k) * 3.14159265 + time * 0.7
    jit = 0.10 * (value3(wp.vec3(time * 0.8, float(k) * 4.1, 0.0)) - 0.5)
    rad = sep * (1.0 + jit)
    y = 0.18 * wp.sin(time * 0.9 + float(k) * 3.1)
    return wp.vec3(rad * wp.cos(ang), y, rad * wp.sin(ang))


@wp.kernel
def meson_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, cq: wp.vec3,
                 caq: wp.vec3, qsize: float, sep: float, conf: float,
                 gain: float, tint: wp.vec3, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    g = _rs(ro, rd, _BOUND)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return

    q0 = _mpos(0, time, sep)
    q1 = _mpos(1, time, sep)
    gcol = wp.vec3(0.8, 0.92, 1.0)                    # gluon tube tint

    t0 = wp.max(g[0], 0.0)
    t1 = g[1]
    nstep = 54
    dt = (t1 - t0) / float(nstep)
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(nstep):
        p = ro + rd * t
        e = quark_emit(p, q0, qsize, cq, time, 1.3)          # quark
        e = e + quark_emit(p, q1, qsize, caq, time, 6.7)     # antiquark
        e = e + tube_emit(p, q0, q1, _TUBE_R, gcol, time)    # single flux string
        e = e + bag_glow(p, wp.vec3(0.0, 0.0, 0.0), conf, tint) * 0.3
        col = col + e * dt
        t += dt

    img[i, j] = col * gain + void(rd)


# name -> (quark flavour, antiquark flavour, separation, qsize, gain, warm)
#   flavours: 0=up 1=down 2=charm 3=strange 4=top 5=bottom
_MESON = {
    "pion":    (0, 1, 0.92, 0.27, 2.0, 1.0),         # π⁺  u d̄  (140 MeV)
    "kaon":    (0, 3, 0.86, 0.27, 2.0, 0.6),         # K⁺  u s̄  (494 MeV)
    "jpsi":    (2, 2, 0.64, 0.23, 2.2, 1.0),         # J/ψ c c̄  (3097 MeV)
    "upsilon": (5, 5, 0.54, 0.21, 2.4, 0.4),         # Υ   b b̄  (9460 MeV)
}


@wp.func
def _anti(c: wp.vec3) -> wp.vec3:
    """Anti-colour tint for the antiquark — shift toward the complementary hue."""
    comp = wp.vec3(1.0, 1.0, 1.0) - c
    return c * 0.35 + comp * 0.65


def render_meson(width, height, time, mouse, device, name="pion"):
    fq, faq, sep, qsize, gain, warm = _MESON[name]
    cq = flavor_color(fq)
    caq = _anti(flavor_color(faq))
    tint = wp.vec3(1.0, 0.9, 0.75) * warm + wp.vec3(0.8, 0.9, 1.0) * (1.0 - warm)
    conf = sep + 0.35

    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=4.0, fov=40.0,
                                   el0=0.4)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(meson_kernel, dim=(height, width),
              inputs=[img, cam, cq, caq, float(qsize), float(sep), float(conf),
                      float(gain), tint, float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.6, strength=0.42,
                              exposure=0.84)
