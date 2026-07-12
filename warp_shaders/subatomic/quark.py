"""Single-quark renderer — a colour-charged, flavour-tinted plasma orb.

A free quark can't be isolated (confinement), so each of the six flavours is shown
as one glowing plasma orb: **size ∝ log(mass)**, tinted by flavour, its QCD colour
charge **cycling** red→green→blue over time, with faint gluon wisps radiating
outward (the field that would bind it). See ``docs/research/21-standard-model.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from . import render as _render_mod
from .field import flavor_color, quark_emit, tube_emit, void

# flavour → (name, mass in MeV) — PDG
_FLAV = {
    0: ("up", 2.2), 1: ("down", 4.7), 2: ("charm", 1270.0),
    3: ("strange", 93.0), 4: ("top", 173000.0), 5: ("bottom", 4180.0),
}


def _size(mass_mev):
    return 0.34 + 0.13 * math.log10(mass_mev)          # ~0.34 (up) .. ~0.93 (top)


@wp.kernel
def quark_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, flav: int,
                 radius: float, charge: wp.vec3, time: float,
                 width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    bound = radius * 2.6
    g = _rs(ro, rd, bound)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return

    fcol = flavor_color(flav)
    center = wp.vec3(0.0, 0.0, 0.0)
    t0 = wp.max(g[0], 0.0)
    t1 = g[1]
    nstep = 52
    dt = (t1 - t0) / float(nstep)
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(nstep):
        p = ro + rd * t
        # flavour-tinted plasma orb (the particle's identity)
        e = quark_emit(p, center, radius, fcol, time, float(flav) * 3.3) * 2.1
        # three gluon wisps radiating out (the confining field) — the gluons carry
        # the cycling colour charge (red→green→blue)
        for k in range(3):
            ang = float(k) * 2.0943951 + time * 0.5
            tip = wp.vec3(wp.cos(ang) * radius * 2.1, wp.sin(ang * 1.3) * radius * 1.2,
                          wp.sin(ang) * radius * 2.1)
            e = e + tube_emit(p, center, tip, radius * 0.16, charge, time) * 0.3
        col = col + e * dt
        t += dt

    img[i, j] = col * 1.5 + void(rd)


def render_quark(width, height, time, mouse, device, flav=0):
    name, mass = _FLAV[flav]
    radius = _size(mass)
    # colour charge cycles R→G→B over ~6 s
    ph = (time * 0.5) % 3.0
    reds = [(1.0, 0.2, 0.2), (0.25, 1.0, 0.3), (0.3, 0.45, 1.0)]
    a = reds[int(ph)]
    b = reds[(int(ph) + 1) % 3]
    f = ph - float(int(ph))
    charge = wp.vec3(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f,
                     a[2] + (b[2] - a[2]) * f)
    cam = _render_mod.orbit_camera(width, height, time, mouse,
                                   dist=radius * 4.5 + 1.6, fov=40.0, el0=0.3)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(quark_kernel, dim=(height, width),
              inputs=[img, cam, int(flav), float(radius), charge, float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.5, strength=0.45,
                              exposure=0.82)
