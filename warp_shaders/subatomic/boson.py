"""Boson renderers — the force carriers and the Higgs.

- **photon** γ: a travelling transverse EM **wave packet** (oscillating E ⊥ B).
- **gluon** g: a **colour + anticolour** double-helix field (it carries both).
- **W / Z**: heavy, short-lived weak bosons — a dense core that **decays** into a
  back-to-back pair of jets on a timer (W charged/orange, Z neutral/blue).
- **Higgs** H: the excitation of the all-pervading Higgs field — a golden core over
  a faint field lattice, decaying to two photon jets (H→γγ).

See ``docs/research/21-standard-model.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import fbm3, value3
from . import render as _render_mod
from .field import sd_capsule, void


# ---------------------------------------------------------------- photon
@wp.kernel
def photon_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, flow: float,
                  time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    g = _rs(ro, rd, 3.8)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return
    t0 = wp.max(g[0], 0.0)
    dt = (g[1] - t0) / 72.0
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(72):
        p = ro + rd * t
        z = p[2]
        phase = 3.0 * z - 6.0 * time
        amp = 0.85
        env = wp.exp(-(z - flow) * (z - flow) * 0.16)     # travelling wave packet
        ex = amp * wp.sin(phase) * env
        by = amp * wp.sin(phase) * env
        de = wp.length(wp.vec2(p[0] - ex, p[1]))
        db = wp.length(wp.vec2(p[0], p[1] - by))
        e = wp.vec3(1.0, 0.85, 0.4) * (wp.exp(-(de / 0.13) * (de / 0.13)) * env)
        e = e + wp.vec3(0.4, 0.75, 1.0) * (wp.exp(-(db / 0.13) * (db / 0.13)) * env)
        axis = wp.length(wp.vec2(p[0], p[1]))
        e = e + wp.vec3(0.5, 0.5, 0.6) * (wp.exp(-(axis / 0.05) * (axis / 0.05)) * env * 0.3)
        col = col + e * (dt * 3.2)
        t += dt
    img[i, j] = col + void(rd)


def render_photon(width, height, time, mouse, device):
    flow = math.fmod(time * 1.6, 6.4) - 3.2               # packet sweeps along z
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=5.5, fov=42.0,
                                   el0=0.22, auto=0.12)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(photon_kernel, dim=(height, width),
              inputs=[img, cam, float(flow), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.3, strength=0.5,
                              exposure=1.02)


# ---------------------------------------------------------------- gluon
@wp.kernel
def gluon_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                 width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    g = _rs(ro, rd, 2.6)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return
    t0 = wp.max(g[0], 0.0)
    dt = (g[1] - t0) / 64.0
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(64):
        p = ro + rd * t
        z = p[2]
        ph = 2.6 * z + time * 2.0
        r = 0.55
        # two intertwined strands: colour (red) + anticolour (cyan), phase offset π
        ax = wp.vec3(r * wp.cos(ph), r * wp.sin(ph), z)
        bx = wp.vec3(r * wp.cos(ph + 3.1416), r * wp.sin(ph + 3.1416), z)
        da = wp.length(p - ax)
        db = wp.length(p - bx)
        env = wp.exp(-(z * z) * 0.12)
        e = wp.vec3(1.0, 0.3, 0.3) * (wp.exp(-(da / 0.15) * (da / 0.15)) * env)
        e = e + wp.vec3(0.3, 0.9, 1.0) * (wp.exp(-(db / 0.15) * (db / 0.15)) * env)
        col = col + e * (dt * 3.4)
        t += dt
    img[i, j] = col + void(rd)


def render_gluon(width, height, time, mouse, device):
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=4.6, fov=42.0,
                                   el0=0.18)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(gluon_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.3, strength=0.5,
                              exposure=1.02)


# ---------------------------------------------------------------- weak (W / Z) + Higgs
@wp.kernel
def decay_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, core_col: wp.vec3,
                 jet_col: wp.vec3, jetlen: float, coreglow: float, lattice: int,
                 time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    g = _rs(ro, rd, 3.6)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return
    t0 = wp.max(g[0], 0.0)
    dt = (g[1] - t0) / 60.0
    col = wp.vec3(0.0, 0.0, 0.0)
    # two back-to-back jet directions (tilted), rotating slowly
    a = time * 0.4
    jdir = wp.vec3(wp.cos(a), 0.35, wp.sin(a))
    jdir = jdir / wp.length(jdir)
    t = t0 + dt * 0.5
    for _ in range(60):
        p = ro + rd * t
        rr = wp.length(p)
        # massive core
        core = wp.exp(-(rr / 0.5) * (rr / 0.5) * 2.0)
        turb = 0.6 + 0.7 * fbm3(p * 2.5 + wp.vec3(0.0, 0.0, time), 4)
        e = core_col * (core * turb * coreglow)
        # two back-to-back decay jets (capsules from the core outward)
        d1 = sd_capsule(p, wp.vec3(0.0, 0.0, 0.0), jdir * jetlen, 0.12)
        d2 = sd_capsule(p, wp.vec3(0.0, 0.0, 0.0), jdir * (-jetlen), 0.12)
        jt = wp.exp(-(d1 / 0.12) * (d1 / 0.12) * 3.0) + wp.exp(-(d2 / 0.12) * (d2 / 0.12) * 3.0)
        flow = 0.6 + 0.4 * wp.sin(rr * 8.0 - time * 9.0)
        e = e + jet_col * (jt * flow * 0.7)
        # faint Higgs-field lattice (a slab of grid lines in the y=0 plane)
        if lattice == 1:
            gy = wp.exp(-(p[1] * p[1]) * 8.0)
            gx = value3(wp.vec3(p[0] * 3.0, 0.0, p[2] * 3.0))
            grid = wp.pow(gx, 8.0)
            e = e + wp.vec3(0.5, 0.42, 0.2) * (grid * gy * 0.5)
        col = col + e * dt
        t += dt
    img[i, j] = col * 1.8 + void(rd)


def _render_decay(width, height, time, mouse, device, core_col, jet_col,
                  coreglow, lattice, period=6.0):
    prog = (time % period) / period                      # 0..1 decay cycle
    jetlen = 0.4 + 2.6 * prog                             # jets shoot out then reset
    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=5.0, fov=42.0,
                                   el0=0.3)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(decay_kernel, dim=(height, width),
              inputs=[img, cam, wp.vec3(*core_col), wp.vec3(*jet_col), float(jetlen),
                      float(coreglow), int(lattice), float(time), int(width),
                      int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.4, strength=0.5,
                              exposure=0.95)


def render_w(width, height, time, mouse, device):
    return _render_decay(width, height, time, mouse, device,
                         core_col=(1.0, 0.55, 0.25), jet_col=(1.0, 0.7, 0.35),
                         coreglow=1.4, lattice=0)


def render_z(width, height, time, mouse, device):
    return _render_decay(width, height, time, mouse, device,
                         core_col=(0.55, 0.7, 1.0), jet_col=(0.7, 0.85, 1.0),
                         coreglow=1.4, lattice=0)


def render_higgs(width, height, time, mouse, device):
    return _render_decay(width, height, time, mouse, device,
                         core_col=(1.0, 0.85, 0.4), jet_col=(1.0, 0.95, 0.7),
                         coreglow=1.7, lattice=1)
