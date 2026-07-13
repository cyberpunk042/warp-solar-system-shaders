"""Particle-detector scenes — a bubble chamber and a collider collision.

- **bubble_chamber**: the historical detector image — charged particles leave
  **curved tracks** in a magnetic field (curvature ∝ 1/momentum; opposite charges
  bend opposite ways; low-momentum particles spiral in as they lose energy), all
  radiating from an interaction vertex, over the pale blue liquid.
- **particle_collision**: a modern collider **event display** — two beams meet at
  the centre and the energy sprays out as a fan of tracks in every direction, with
  a bright collision flash at the vertex.

See ``docs/research/21-standard-model.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine.uniforms import Camera, camera_ray_dir
from ..particles import emitter, flux
from . import render as _render_mod
from .field import void


@wp.func
def _track_color(k: int) -> wp.vec3:
    m = k % 6
    if m == 0:
        return wp.vec3(0.5, 0.75, 1.0)      # blue
    if m == 1:
        return wp.vec3(0.5, 1.0, 0.7)       # green
    if m == 2:
        return wp.vec3(1.0, 0.85, 0.4)      # amber
    if m == 3:
        return wp.vec3(1.0, 0.55, 0.7)      # pink
    if m == 4:
        return wp.vec3(0.7, 0.9, 1.0)       # pale
    return wp.vec3(0.9, 0.7, 1.0)           # violet


# ---------------------------------------------------------------- collider event
@wp.kernel
def collision_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                     width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    col = void(rd)
    vtx = wp.vec3(0.0, 0.0, 0.0)
    # spray of tracks in golden-angle directions, each bent a little by the field
    for k in range(16):
        gk = float(k) + 0.5
        y = 1.0 - 2.0 * gk / 16.0
        rr = wp.sqrt(wp.max(1.0 - y * y, 0.0))
        th = 2.399963 * gk + time * 0.05
        d = wp.vec3(rr * wp.cos(th), y, rr * wp.sin(th))
        bend = wp.vec3(-d[2], 0.0, d[0]) * 0.25         # solenoid B-field curl
        endp = d * 1.9 + bend
        col = col + _track_color(k) * (flux(ro, rd, vtx, endp, time) * 1.3)
    # the two incoming beams along the pipe axis (±z), faint
    col = col + wp.vec3(0.8, 0.9, 1.0) * (flux(ro, rd, wp.vec3(0.0, 0.0, -2.6), vtx, time) * 0.35)
    col = col + wp.vec3(0.8, 0.9, 1.0) * (flux(ro, rd, wp.vec3(0.0, 0.0, 2.6), vtx, time) * 0.35)
    # the collision flash at the vertex
    col = col + wp.vec3(1.0, 0.96, 0.86) * (emitter(ro, rd, vtx, 0.16) * 1.6)
    img[i, j] = col


# ---------------------------------------------------------------- bubble chamber
@wp.kernel
def bubble_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
                  width: int, height: int):
    i, j = wp.tid()
    x = ((2.0 * (float(j) + 0.5) / float(width)) - 1.0) * aspect
    y = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0

    # pale-blue liquid, slightly vignetted
    col = wp.vec3(0.06, 0.12, 0.22) * (1.0 - 0.35 * (x * x + y * y))
    vx = 0.0
    vy = -0.75
    for k in range(9):
        a0 = 1.5708 + (float(k) - 4.0) * 0.17           # fan of initial directions
        sgn = 1.0
        if (k % 2) == 1:
            sgn = -1.0                                    # alternate charge sign
        rad = 0.55 + 0.32 * float(k % 4)                 # curvature radius ∝ momentum
        dx = wp.cos(a0)
        dy = wp.sin(a0)
        cx = vx - dy * sgn * rad                          # circle centre ⟂ to v
        cy = vy + dx * sgn * rad
        rp = wp.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy))
        dd = wp.abs(rp - rad)
        line = wp.exp(-dd * dd * 2600.0)                  # a thin track
        distv = wp.sqrt((x - vx) * (x - vx) + (y - vy) * (y - vy))
        # only the outgoing branch (above the vertex) + fade as it loses energy
        ahead = wp.smoothstep(-0.1, 0.15, y - vy)
        fade = wp.exp(-distv * 0.7)
        col = col + _track_color(k) * (line * ahead * (0.4 + fade) * 1.1)
    # a neutral-decay "V" off to the side (two tracks from a bare point)
    for k2 in range(2):
        s2 = 1.0
        if k2 == 1:
            s2 = -1.0
        vx2 = 0.55
        vy2 = 0.2
        a2 = 1.2 + s2 * 0.5
        cx2 = vx2 - wp.sin(a2) * s2 * 0.7
        cy2 = vy2 + wp.cos(a2) * s2 * 0.7
        rp2 = wp.sqrt((x - cx2) * (x - cx2) + (y - cy2) * (y - cy2))
        dd2 = wp.abs(rp2 - 0.7)
        line2 = wp.exp(-dd2 * dd2 * 2600.0)
        ah2 = wp.smoothstep(-0.05, 0.1, y - vy2) * wp.exp(-((x - vx2) * (x - vx2) + (y - vy2) * (y - vy2)) * 1.5)
        col = col + wp.vec3(1.0, 0.85, 0.5) * (line2 * ah2 * 1.0)
    img[i, j] = col


def render_collision(width, height, time, mouse, device):
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=5.4, fov=44.0,
                                   el0=0.28, auto=0.12)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(collision_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.3, strength=0.5,
                              exposure=1.02)


def render_bubble_chamber(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(bubble_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.3, strength=0.35,
                              exposure=1.05)
