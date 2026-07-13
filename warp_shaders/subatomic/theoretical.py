"""Hypothetical / theoretical particles — never observed, but predicted.

- **tachyon**: a particle with imaginary rest mass that would move *faster* than
  light, dragging a **Cherenkov shock cone** of blueshifted light behind it (a
  Mach cone of the electromagnetic field).
- **graviton**: the conjectured spin-2 quantum of gravity — a ripple in the fabric
  of spacetime, drawn as a grid stretched and squeezed by the passing wave's
  **quadrupole** (plus-polarisation) strain.
- **magnetic_monopole**: an isolated north (or south) magnetic charge, with
  **radial** magnetic field lines streaming out in every direction (a real
  magnet's field always loops; a monopole's does not).

See ``docs/research/21-standard-model.md``.
"""

import numpy as np
import warp as wp

from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import fbm3, value3
from . import render as _render_mod
from .field import void

_BOUND = 4.0


# --------------------------------------------------------------------- tachyon
@wp.kernel
def tachyon_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, apexz: float,
                   time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    g = _rs(ro, rd, _BOUND)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return
    apex = wp.vec3(0.0, 0.0, apexz)
    axis = wp.vec3(0.0, 0.0, -1.0)                    # cone opens backward
    t0 = wp.max(g[0], 0.0)
    t1 = g[1]
    nstep = 60
    dt = (t1 - t0) / float(nstep)
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(nstep):
        p = ro + rd * t
        w = p - apex
        along = wp.dot(w, axis)
        if along > 0.0:
            rperp = wp.length(w - axis * along)
            ang = wp.atan2(rperp, along)
            mach = 0.5
            shell = wp.exp(-((ang - mach) / 0.055) * ((ang - mach) / 0.055))
            fade = wp.exp(-along * 0.7)
            turb = 0.6 + 0.6 * fbm3(w * 3.0 + wp.vec3(0.0, 0.0, time * 1.5), 3)
            col = col + wp.vec3(0.4, 0.58, 1.0) * (shell * fade * turb * dt * 1.4)
        # the tachyon itself at the apex — a blueshifted point
        dc = wp.length(p - apex)
        col = col + wp.vec3(0.7, 0.85, 1.0) * (wp.exp(-dc * dc * 16.0) * dt * 2.6)
        t += dt
    img[i, j] = col + void(rd)


# -------------------------------------------------------------------- graviton
@wp.func
def _gridline(a: float, b: float) -> float:
    fa = wp.abs(a - wp.floor(a + 0.5))
    fb = wp.abs(b - wp.floor(b + 0.5))
    line = wp.exp(-fa * fa * 90.0) + wp.exp(-fb * fb * 90.0)
    return wp.min(line, 1.0)


@wp.kernel
def graviton_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                    width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    if rd[1] > -0.001:                               # only the grid plane below
        img[i, j] = void(rd)
        return
    t = -ro[1] / rd[1]
    p = ro + rd * t
    r = wp.length(wp.vec2(p[0], p[2])) + 1e-4
    phi = wp.atan2(p[2], p[0])
    # spin-2 quadrupole strain: a ripple modulated by cos(2φ) (plus polarisation)
    ripple = wp.sin(r * 2.6 - time * 3.0) * wp.exp(-r * 0.22)
    strain = ripple * wp.cos(2.0 * phi)
    # radially stretch / squeeze the grid coordinates by the strain
    s = 1.0 + 0.5 * strain
    gx = p[0] * s
    gz = p[2] * s
    line = _gridline(gx * 0.9, gz * 0.9)
    stretch = wp.clamp(strain, 0.0, 1.0)
    squeeze = wp.clamp(-strain, 0.0, 1.0)
    base = wp.vec3(0.10, 0.14, 0.28)
    col = base * 0.4 + wp.vec3(0.4, 0.6, 1.0) * (line * (0.4 + stretch * 1.4)) \
        + wp.vec3(1.0, 0.5, 0.35) * (line * squeeze * 1.2)
    # the graviton quantum: a bright packet at the centre
    col = col + wp.vec3(0.8, 0.9, 1.0) * (wp.exp(-r * r * 3.0) * 1.4)
    fog = 1.0 - wp.exp(-t * 0.04)
    col = col * (1.0 - fog) + wp.vec3(0.02, 0.03, 0.06) * fog
    img[i, j] = col


# ------------------------------------------------------------- magnetic monopole
@wp.kernel
def monopole_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
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
    t0 = wp.max(g[0], 0.0)
    t1 = g[1]
    nstep = 58
    dt = (t1 - t0) / float(nstep)
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(nstep):
        p = ro + rd * t
        r = wp.length(p) + 1e-4
        dir = p / r
        # discrete radial field lines: bright where the direction lands near a
        # cell centre of a spherical grid (a hedgehog of outgoing B-field)
        cell = value3(dir * 5.0)
        linev = wp.pow(wp.max(cell, 0.0), 6.0)
        flow = 0.5 + 0.5 * wp.sin(r * 6.0 - time * 4.0)   # field streaming outward
        fall = wp.exp(-r * 1.0)
        col = col + wp.vec3(0.4, 0.6, 1.0) * (linev * flow * fall * dt * 5.0)
        # the monopole core
        col = col + wp.vec3(1.0, 0.95, 0.85) * (wp.exp(-r * r * 12.0) * dt * 5.0)
        t += dt
    img[i, j] = col + void(rd)


def render_tachyon(width, height, time, mouse, device):
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=5.6, fov=44.0,
                                   el0=0.14, auto=0.08)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(tachyon_kernel, dim=(height, width),
              inputs=[img, cam, 1.4, float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.4, strength=0.45,
                              exposure=0.82)


def render_graviton(width, height, time, mouse, device):
    import math
    az = 0.6 + time * 0.05 + float(mouse[0]) * 0.008
    eye = (math.sin(az) * 6.5, 4.2, math.cos(az) * 6.5)
    from ..engine.uniforms import make_camera
    cam = make_camera(eye, (0.0, -0.2, 0.0), fov_deg=48.0, aspect=width / height)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(graviton_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.2, strength=0.4,
                              exposure=1.05)


def render_monopole(width, height, time, mouse, device):
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=5.4, fov=42.0,
                                   el0=0.3, auto=0.15)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(monopole_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.3, strength=0.5,
                              exposure=1.0)
