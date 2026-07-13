"""Nucleon renderer — three colour-charged quarks in a confinement bag, bound by
gluon flux tubes, ray-marched as a volumetric emission field.

Shared by the ``proton`` (uud) and ``neutron`` (udd) scenes. See
``docs/research/21-standard-model.md``.
"""

import numpy as np
import warp as wp

from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import value3
from . import render as _render_mod
from .field import bag_glow, color_charge, quark_emit, tube_emit, void

_BOUND = 1.9
_CONF = 1.0
_QSIZE = 0.34
_TUBE_R = 0.1


@wp.func
def _qpos(k: int, time: float) -> wp.vec3:
    """Position of quark ``k`` confined in the nucleon: a slowly rotating triad
    with a little quantum jitter (asymptotic freedom — they rattle inside)."""
    ang = float(k) * 2.0943951 + time * 0.5
    jit = 0.16 * (value3(wp.vec3(time * 0.7, float(k) * 3.1, 0.0)) - 0.5)
    rad = _CONF * (0.62 + jit)
    z = _CONF * 0.26 * wp.sin(time * 0.9 + float(k) * 2.1)
    return wp.vec3(rad * wp.cos(ang), z, rad * wp.sin(ang))


@wp.kernel
def nucleon_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera,
                   b0: float, b1: float, b2: float, warm: float,
                   anti: int, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    g = _rs(ro - wp.vec3(0.0, 0.0, 0.0), rd, _BOUND)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return

    q0 = _qpos(0, time)
    q1 = _qpos(1, time)
    q2 = _qpos(2, time)
    c0 = color_charge(0)
    c1 = color_charge(1)
    c2 = color_charge(2)
    if anti == 1:                                    # antiquarks carry anti-colour
        c0 = wp.vec3(1.0, 1.0, 1.0) - c0
        c1 = wp.vec3(1.0, 1.0, 1.0) - c1
        c2 = wp.vec3(1.0, 1.0, 1.0) - c2
    gcol = wp.vec3(0.8, 0.92, 1.0)                    # gluon tube tint
    tint = wp.vec3(1.0, 0.9, 0.75) * warm + wp.vec3(0.8, 0.9, 1.0) * (1.0 - warm)
    if anti == 1:
        tint = wp.vec3(0.75, 0.6, 1.0)               # violet anti-matter bag

    t0 = wp.max(g[0], 0.0)
    t1 = g[1]
    nstep = 56
    dt = (t1 - t0) / float(nstep)
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(nstep):
        p = ro + rd * t
        e = quark_emit(p, q0, _QSIZE, c0, time, 1.3) * b0
        e = e + quark_emit(p, q1, _QSIZE, c1, time, 5.1) * b1
        e = e + quark_emit(p, q2, _QSIZE, c2, time, 9.7) * b2
        e = e + tube_emit(p, q0, q1, _TUBE_R, gcol, time)
        e = e + tube_emit(p, q1, q2, _TUBE_R, gcol, time)
        e = e + tube_emit(p, q2, q0, _TUBE_R, gcol, time)
        e = e + bag_glow(p, wp.vec3(0.0, 0.0, 0.0), _CONF, tint)
        col = col + e * dt
        t += dt

    img[i, j] = col * 1.9 + void(rd)


def render_nucleon(width, height, time, mouse, device, is_proton=True, anti=False):
    """Render a proton (uud) or neutron (udd). Down quarks render dimmer; the
    confinement bag is warm for the charged proton, cool for the neutral neutron.
    ``anti=True`` renders the antiparticle (anti-colour quarks, violet bag).
    """
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=4.2, fov=40.0,
                                   el0=0.72)
    if is_proton:                                    # u, u, d
        b0, b1, b2, warm = 1.0, 1.0, 0.72, 1.0
    else:                                            # u, d, d
        b0, b1, b2, warm = 1.0, 0.72, 0.72, 0.0
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(nucleon_kernel, dim=(height, width),
              inputs=[img, cam, float(b0), float(b1), float(b2), float(warm),
                      int(1 if anti else 0), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.4, strength=0.42,
                              exposure=0.92)
