"""Exotic / charged atoms — the ion and positronium.

- **ion**: an atom that has lost an electron — a nucleus with a depleted electron
  cloud, the ejected electron streaking away, and a net **positive** charge halo
  (a cation, the moment of ionisation).
- **positronium**: a hydrogen-like atom of an **electron + positron** bound by
  their mutual attraction, orbiting their common centre (equal masses, so both
  circle the midpoint) inside a shared probability cloud — a matter/antimatter
  atom that annihilates in ~0.1 ns.

Reuses the hydrogen orbital density from ``field.orbital_psi2``. See
``docs/research/21-standard-model.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from . import render as _render_mod
from .field import orbital_psi2, void

_BOUND = 4.6


@wp.func
def _pt(p: wp.vec3, c: wp.vec3, col: wp.vec3, s: float) -> wp.vec3:
    d = wp.length(p - c)
    core = wp.exp(-(d / s) * (d / s) * 4.0)
    return (wp.vec3(1.0, 1.0, 1.0) * 0.5 + col * 0.5) * core


@wp.kernel
def ion_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, a0: float,
               elx: float, escape: float, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    g = _rs(ro, rd, _BOUND)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return
    t0 = wp.max(g[0], 0.0)
    t1 = g[1]
    nstep = 70
    dt = (t1 - t0) / float(nstep)
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(nstep):
        p = ro + rd * t
        # remaining (depleted) electron cloud — dimmer on the side the e⁻ left
        dens = orbital_psi2(p, 1, a0)
        side = 0.5 + 0.5 * wp.clamp(-p[0] * 0.9, -0.5, 0.5)
        col = col + wp.vec3(0.3, 0.55, 1.0) * (dens * 2.6 * side * dt)
        # net positive-charge halo (warm, radial)
        r = wp.length(p) + 1e-4
        halo = wp.exp(-r * 1.4) * (0.4 + 0.4 * wp.sin(r * 5.0 - time * 2.0))
        col = col + wp.vec3(1.0, 0.5, 0.24) * (halo * 0.11 * dt)
        # the ejected electron streaking away (+x) with a short trail
        col = col + _pt(p, wp.vec3(elx, 0.0, 0.0), wp.vec3(0.4, 0.8, 1.0), 0.13) * (escape * dt * 5.0)
        col = col + _pt(p, wp.vec3(elx * 0.7, 0.0, 0.0), wp.vec3(0.4, 0.8, 1.0), 0.09) * (escape * dt * 2.0)
        t += dt

    # bright nucleus (net-charged core)
    tc = wp.max(wp.dot(-ro, rd), 0.0)
    dc = wp.length(-(ro + rd * tc))
    core = wp.exp(-(dc / 0.12) * (dc / 0.12) * 4.0)
    col = col + wp.vec3(1.0, 0.82, 0.6) * (core * 1.5)
    img[i, j] = col + void(rd)


@wp.kernel
def positronium_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, ex: float,
                       ez: float, a0: float, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    g = _rs(ro, rd, _BOUND)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return
    t0 = wp.max(g[0], 0.0)
    t1 = g[1]
    ep = wp.vec3(ex, 0.0, ez)
    pp = wp.vec3(-ex, 0.0, -ez)
    nstep = 60
    dt = (t1 - t0) / float(nstep)
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(nstep):
        p = ro + rd * t
        # shared bound cloud around the common centre
        dens = orbital_psi2(p, 0, a0)
        col = col + wp.vec3(0.5, 0.45, 0.72) * (dens * 1.4 * dt)
        # the electron (cyan) and positron (orange) orbiting the midpoint
        col = col + _pt(p, ep, wp.vec3(0.4, 0.8, 1.0), 0.15) * dt * 5.0
        col = col + _pt(p, pp, wp.vec3(1.0, 0.6, 0.25), 0.15) * dt * 5.0
        # the bond — an emission line between them
        ba = pp - ep
        h = wp.clamp(wp.dot(p - ep, ba) / wp.dot(ba, ba), 0.0, 1.0)
        axis = ep + ba * h
        rr = wp.length(p - axis)
        bond = wp.exp(-(rr / 0.06) * (rr / 0.06))
        col = col + wp.vec3(0.8, 0.7, 1.0) * (bond * 0.5 * dt)
        t += dt
    img[i, j] = col + void(rd)


def render_ion(width, height, time, mouse, device, period=5.0):
    prog = (time % period) / period
    elx = 0.5 + prog * 3.4                                # ejected electron flies out
    escape = min(prog * 3.0, 1.0)
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=6.0, fov=42.0,
                                   el0=0.3, auto=0.12)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(ion_kernel, dim=(height, width),
              inputs=[img, cam, 0.5, float(elx), float(escape), float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.3, strength=0.45,
                              exposure=1.0)


def render_positronium(width, height, time, mouse, device):
    ang = time * 1.4
    r = 1.15
    ex = r * math.cos(ang)
    ez = r * math.sin(ang)
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=5.4, fov=42.0,
                                   el0=0.5, auto=0.1)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(positronium_kernel, dim=(height, width),
              inputs=[img, cam, float(ex), float(ez), 0.7, float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.3, strength=0.48,
                              exposure=1.0)
