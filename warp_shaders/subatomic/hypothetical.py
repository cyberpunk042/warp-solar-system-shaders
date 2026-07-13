"""Dark-sector hypotheticals — the axion and the WIMP (dark matter).

- **axion**: an ultralight pseudoscalar proposed to solve the strong-CP problem
  and a leading cold-dark-matter candidate. Nearly invisible; its one handle is the
  **Primakoff effect** — in a magnetic field an axion can convert into a photon,
  seen here as rare flashes inside a faint shimmer.
- **dark_matter** (WIMP): massive but non-luminous — it neither emits nor absorbs
  light. The only sign of it here is **gravitational lensing**: the background
  starlight is bent into arcs around an unseen mass.

See ``docs/research/21-standard-model.md``.
"""

import numpy as np
import warp as wp

from ..engine.intersect import ray_sphere_o as _rs
from ..engine.sky import starfield
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import fbm3, value3
from . import render as _render_mod
from .field import void

_BOUND = 3.6


@wp.kernel
def axion_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
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
    nstep = 50
    dt = (t1 - t0) / float(nstep)
    col = wp.vec3(0.0, 0.0, 0.0)
    # three Primakoff conversion sites, each flashing on its own phase
    s0 = wp.vec3(0.8, 0.4, -0.5)
    s1 = wp.vec3(-0.7, -0.3, 0.6)
    s2 = wp.vec3(0.1, 0.7, 0.9)
    f0 = wp.max(wp.sin(time * 2.3) - 0.6, 0.0) * 2.5
    f1 = wp.max(wp.sin(time * 1.7 + 2.0) - 0.6, 0.0) * 2.5
    f2 = wp.max(wp.sin(time * 3.1 + 4.0) - 0.6, 0.0) * 2.5
    t = t0 + dt * 0.5
    for _ in range(nstep):
        p = ro + rd * t
        r = wp.length(p) + 1e-4
        # a faint, ghostly pseudoscalar shimmer (axions are almost invisible)
        shim = fbm3(p * 2.2 + wp.vec3(0.0, 0.0, time * 0.5), 4)
        col = col + wp.vec3(0.3, 0.26, 0.55) * (wp.max(shim, 0.0) * wp.exp(-r * 0.9) * 0.06 * dt)
        # faint horizontal magnetic field lines (the conversion medium)
        fld = wp.exp(-wp.abs(p[1] - wp.floor(p[1] + 0.5)) * 7.0)
        col = col + wp.vec3(0.18, 0.32, 0.5) * (fld * wp.exp(-r * 1.0) * 0.03 * dt)
        # rare photon flashes where an axion converts
        col = col + wp.vec3(1.0, 0.9, 0.55) * (wp.exp(-wp.length(p - s0) * wp.length(p - s0) * 30.0) * f0 * dt * 4.0)
        col = col + wp.vec3(1.0, 0.9, 0.55) * (wp.exp(-wp.length(p - s1) * wp.length(p - s1) * 30.0) * f1 * dt * 4.0)
        col = col + wp.vec3(1.0, 0.9, 0.55) * (wp.exp(-wp.length(p - s2) * wp.length(p - s2) * 30.0) * f2 * dt * 4.0)
        t += dt
    img[i, j] = col + void(rd)


@wp.kernel
def dark_matter_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, klens: float,
                       time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    # the invisible mass sits at the origin; bend the ray toward it (weak lensing)
    fwd = wp.normalize(-ro)
    perp = rd - fwd * wp.dot(rd, fwd)
    b = wp.length(perp) + 0.02                        # angular impact parameter
    defl = klens / (b * b + 0.02)
    rd2 = wp.normalize(rd - wp.normalize(perp) * defl)
    col = starfield(rd2) * 1.4
    # a whisper of a halo so the mass isn't perfectly invisible
    col = col + wp.vec3(0.10, 0.12, 0.2) * (wp.exp(-b * b * 6.0) * 0.5)
    img[i, j] = col


def render_axion(width, height, time, mouse, device):
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=5.0, fov=42.0,
                                   el0=0.25, auto=0.1)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(axion_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.0, strength=0.5,
                              exposure=1.1)


def render_dark_matter(width, height, time, mouse, device):
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=5.0, fov=46.0,
                                   el0=0.2, auto=0.06)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(dark_matter_kernel, dim=(height, width),
              inputs=[img, cam, 0.16, float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.4, strength=0.35,
                              exposure=1.0)
