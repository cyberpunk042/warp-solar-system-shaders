"""Electron–positron annihilation — e⁻ + e⁺ → γ γ.

Matter meets antimatter: an electron (cool cyan) and a positron (warm orange) drift
together, annihilate in a blinding flash at the centre, and their rest-mass energy
leaves as **two back-to-back gamma photons** (E = mc², 511 keV each, momentum-
conserving so they fly apart in exactly opposite directions) drawn as travelling
transverse-EM wave-streaks. Loops with `time`. See ``docs/research/21-standard-model.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from . import render as _render_mod
from .field import void

_BOUND = 3.4


@wp.func
def _emit_pt(p: wp.vec3, c: wp.vec3, col: wp.vec3, s: float) -> wp.vec3:
    d = wp.length(p - c)
    core = wp.exp(-(d / s) * (d / s) * 4.0)
    return (wp.vec3(1.0, 1.0, 1.0) * 0.5 + col * 0.5) * core


@wp.kernel
def annih_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sep: float,
                 flash: float, gpos: float, gshow: float, time: float,
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
    nstep = 56
    dt = (t1 - t0) / float(nstep)
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(nstep):
        p = ro + rd * t
        # incoming electron (cyan, −x) + positron (orange, +x)
        col = col + _emit_pt(p, wp.vec3(-sep, 0.0, 0.0), wp.vec3(0.4, 0.8, 1.0), 0.16) * dt * 4.0
        col = col + _emit_pt(p, wp.vec3(sep, 0.0, 0.0), wp.vec3(1.0, 0.6, 0.25), 0.16) * dt * 4.0
        # annihilation flash at the origin
        dc = wp.length(p)
        col = col + wp.vec3(1.0, 0.98, 0.95) * (wp.exp(-dc * dc * 6.0) * flash * dt * 6.0)
        # two back-to-back gamma photons (transverse-EM streaks along ±y)
        if gshow > 0.0:
            ax = wp.length(wp.vec2(p[0], p[2]))          # distance from the y-axis
            wave = 0.5 + 0.5 * wp.sin(p[1] * 9.0 - time * 8.0)
            for s in range(2):
                yc = gpos
                if s == 1:
                    yc = -gpos
                dy = wp.abs(p[1] - yc)
                streak = wp.exp(-(ax / 0.10) * (ax / 0.10)) * wp.exp(-(dy / 0.5) * (dy / 0.5))
                col = col + wp.vec3(0.8, 0.85, 1.0) * (streak * wave * gshow * dt * 5.0)
        t += dt

    img[i, j] = col + void(rd)


def render_annihilation(width, height, time, mouse, device, period=6.0):
    prog = (time % period) / period
    if prog < 0.5:
        sep = (0.5 - prog) * 3.4                          # e⁻/e⁺ drift in
        flash = 0.0
        gshow = 0.0
        gpos = 0.0
    else:
        sep = 0.0
        q = (prog - 0.5) / 0.5                            # 0..1 after collision
        flash = math.exp(-((q) / 0.12) * (q / 0.12))      # brief flash
        gshow = min(q * 4.0, 1.0) * (1.0 - q * 0.6)       # photons fade as they leave
        gpos = 0.4 + q * 2.8                              # fly apart along ±y
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=5.2, fov=42.0,
                                   el0=0.22, auto=0.08)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(annih_kernel, dim=(height, width),
              inputs=[img, cam, float(sep), float(flash), float(gpos), float(gshow),
                      float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.3, strength=0.5,
                              exposure=1.0)
