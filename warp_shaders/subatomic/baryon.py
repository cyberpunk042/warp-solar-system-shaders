"""Baryon renderer — three quarks of chosen **flavour** in a confinement bag.

The proton (uud) and neutron (udd) are the two lightest baryons; swap in strange
quarks and you get the **hyperons** — Λ (uds), Σ (uus), Ξ (uss, "cascade"),
Ω⁻ (sss) — and three up quarks give the Δ⁺⁺ resonance (uuu). This renderer is the
nucleon field with each quark tinted by its **flavour** (`field.flavor_color`) as
well as carrying a QCD colour charge, so the three still sum colour-neutral. See
``docs/research/21-standard-model.md``.
"""

import numpy as np
import warp as wp

from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import value3
from . import render as _render_mod
from .field import bag_glow, flavor_color, quark_emit, tube_emit, void

_BOUND = 1.9
_CONF = 1.0
_QSIZE = 0.32
_TUBE_R = 0.1


@wp.func
def _qpos(k: int, time: float) -> wp.vec3:
    ang = float(k) * 2.0943951 + time * 0.5
    jit = 0.16 * (value3(wp.vec3(time * 0.7, float(k) * 3.1, 0.0)) - 0.5)
    rad = _CONF * (0.62 + jit)
    z = _CONF * 0.26 * wp.sin(time * 0.9 + float(k) * 2.1)
    return wp.vec3(rad * wp.cos(ang), z, rad * wp.sin(ang))


@wp.kernel
def baryon_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, c0: wp.vec3,
                  c1: wp.vec3, c2: wp.vec3, warm: float, time: float,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    g = _rs(ro, rd, _BOUND)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return

    q0 = _qpos(0, time)
    q1 = _qpos(1, time)
    q2 = _qpos(2, time)
    gcol = wp.vec3(0.8, 0.92, 1.0)
    tint = wp.vec3(1.0, 0.9, 0.75) * warm + wp.vec3(0.8, 0.9, 1.0) * (1.0 - warm)

    t0 = wp.max(g[0], 0.0)
    t1 = g[1]
    nstep = 56
    dt = (t1 - t0) / float(nstep)
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(nstep):
        p = ro + rd * t
        e = quark_emit(p, q0, _QSIZE, c0, time, 1.3)
        e = e + quark_emit(p, q1, _QSIZE, c1, time, 5.1)
        e = e + quark_emit(p, q2, _QSIZE, c2, time, 9.7)
        e = e + tube_emit(p, q0, q1, _TUBE_R, gcol, time)
        e = e + tube_emit(p, q1, q2, _TUBE_R, gcol, time)
        e = e + tube_emit(p, q2, q0, _TUBE_R, gcol, time)
        e = e + bag_glow(p, wp.vec3(0.0, 0.0, 0.0), _CONF, tint)
        col = col + e * dt
        t += dt
    img[i, j] = col * 1.9 + void(rd)


# name -> (three quark flavours, warm 0..1 from net charge)
#   flavours: 0=up 1=down 2=charm 3=strange 4=top 5=bottom
_BARYON = {
    "lambda": (0, 1, 3, 0.0),      # Λ⁰  u d s   (1116 MeV, neutral)
    "sigma":  (0, 0, 3, 1.0),      # Σ⁺  u u s   (1189 MeV, +1)
    "xi":     (0, 3, 3, 0.0),      # Ξ⁰  u s s   (1315 MeV, cascade)
    "omega":  (3, 3, 3, 0.4),      # Ω⁻  s s s   (1672 MeV, −1, all-strange)
    "delta":  (0, 0, 0, 1.0),      # Δ⁺⁺ u u u   (1232 MeV, +2 resonance)
}


def render_baryon(width, height, time, mouse, device, name="lambda"):
    f0, f1, f2, warm = _BARYON[name]
    # each quark: its flavour colour, nudged by a QCD colour charge so the triad
    # still reads as red/green/blue-ish neutral
    c0 = flavor_color(f0) * 0.8 + wp.vec3(0.3, 0.0, 0.0)
    c1 = flavor_color(f1) * 0.8 + wp.vec3(0.0, 0.3, 0.0)
    c2 = flavor_color(f2) * 0.8 + wp.vec3(0.0, 0.0, 0.3)
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=4.2, fov=40.0,
                                   el0=0.72)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(baryon_kernel, dim=(height, width),
              inputs=[img, cam, c0, c1, c2, float(warm), float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.4, strength=0.42,
                              exposure=0.92)
