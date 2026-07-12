"""Hydrogen orbital renderer — the electron's probability cloud |ψ_{nlm}|²
ray-marched as a volumetric emission field, around a bright nucleus.

The real radial (Laguerre) × angular (spherical-harmonic) shapes appear: the 1s
sphere, the 2p dumbbell, the 3d cloverleaf, complete with their nodes. See
``docs/research/21-standard-model.md``.
"""

import numpy as np
import warp as wp

from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from . import render as _render_mod
from .field import orbital_psi2, void


@wp.func
def _cloud_ramp(x: float) -> wp.vec3:
    """Density → electron-cloud colour: faint indigo → blue → cyan → white."""
    x = wp.clamp(x, 0.0, 1.0)
    a = wp.vec3(0.05, 0.06, 0.22)
    b = wp.vec3(0.15, 0.35, 0.9)
    c = wp.vec3(0.5, 0.85, 1.0)
    d = wp.vec3(1.0, 1.0, 1.0)
    if x < 0.4:
        return a + (b - a) * (x / 0.4)
    if x < 0.75:
        return b + (c - b) * ((x - 0.4) / 0.35)
    return c + (d - c) * ((x - 0.75) / 0.25)


@wp.kernel
def orbital_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, orb: int,
                   a0: float, bound: float, bright: float, nucleus: float,
                   width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    g = _rs(ro, rd, bound)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return

    t0 = wp.max(g[0], 0.0)
    t1 = g[1]
    nstep = 80
    dt = (t1 - t0) / float(nstep)
    acc = float(0.0)
    t = t0 + dt * 0.5
    for _ in range(nstep):
        p = ro + rd * t
        acc = acc + orbital_psi2(p, orb, a0) * dt
        t += dt

    col = _cloud_ramp(acc * bright) * wp.clamp(acc * bright * 2.0, 0.0, 1.4)

    # the nucleus (a tiny, bright proton) at the centre
    if nucleus > 0.0:
        oc = -ro
        tc = wp.max(wp.dot(oc, rd), 0.0)
        dc = wp.length(-(ro + rd * tc))
        core = wp.exp(-(dc / 0.12) * (dc / 0.12) * 4.0)
        col = col + wp.vec3(1.0, 0.85, 0.7) * (core * nucleus)

    img[i, j] = col + void(rd)


def render_orbital(width, height, time, mouse, device, orb=0, a0=0.32,
                   bound=5.0, bright=6.0, nucleus=1.4, dist=5.2, el0=0.32):
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=dist,
                                   fov=42.0, el0=el0)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(orbital_kernel, dim=(height, width),
              inputs=[img, cam, int(orb), float(a0), float(bound), float(bright),
                      float(nucleus), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.3, strength=0.4,
                              exposure=1.0)


# per-orbital tuning: (a0, bound, brightness) so each reads at a good exposure
_ORB = {
    0: (0.55, 4.0, 2.6),      # 1s  — compact sphere
    1: (0.42, 6.5, 2.4),      # 2s  — sphere + shell node
    2: (0.42, 6.5, 4.2),      # 2p  — dumbbell
    3: (0.33, 9.0, 4.5),      # 3p
    4: (0.32, 10.0, 5.5),     # 3d z²
    5: (0.32, 10.0, 5.5),     # 3d cloverleaf
}


def render_named(width, height, time, mouse, device, orb, nucleus=1.4):
    a0, bound, bright = _ORB.get(orb, (0.4, 6.0, 5.0))
    dist = bound * 0.9 + 1.5
    return render_orbital(width, height, time, mouse, device, orb=orb, a0=a0,
                          bound=bound, bright=bright, nucleus=nucleus, dist=dist,
                          el0=0.3)
