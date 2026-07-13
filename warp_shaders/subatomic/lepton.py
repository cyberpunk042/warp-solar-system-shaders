"""Lepton renderer — charged leptons (e, μ, τ) and neutrinos (νₑ, ν_μ, ν_τ).

Charged leptons are point particles shown as a bright core wrapped in an animated
**electromagnetic field** (radial filaments + outgoing Coulomb ripples), coloured
by generation and sized by mass. Neutrinos are nearly invisible — a faint,
elongated shimmer that **oscillates** between flavours as it streaks through. See
``docs/research/21-standard-model.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import fbm3
from . import render as _render_mod
from .field import void

# kind: 0=electron,1=muon,2=tau, 3=nu_e,4=nu_mu,5=nu_tau ; (name, mass MeV, gen)
_LEP = {
    0: ("electron", 0.511, 0), 1: ("muon", 105.7, 1), 2: ("tau", 1776.9, 2),
    3: ("neutrino_e", 0.0000001, 0), 4: ("neutrino_mu", 0.0000001, 1),
    5: ("neutrino_tau", 0.0000001, 2),
}


@wp.func
def _gen_color(gen: int) -> wp.vec3:
    if gen == 0:
        return wp.vec3(0.45, 0.8, 1.0)               # gen I — cyan
    if gen == 1:
        return wp.vec3(0.55, 1.0, 0.65)              # gen II — green
    return wp.vec3(0.8, 0.6, 1.0)                     # gen III — violet


@wp.kernel
def lepton_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, charged: int,
                  fcol: wp.vec3, core_r: float, core_i: float, time: float,
                  osc: float, anti: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    g = _rs(ro, rd, 3.2)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return

    t0 = wp.max(g[0], 0.0)
    t1 = g[1]
    nstep = 54
    dt = (t1 - t0) / float(nstep)
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(nstep):
        p = ro + rd * t
        r = wp.length(p) + 1.0e-4
        dir = p / r
        if charged == 1:
            # radial EM field filaments + Coulomb ripples; antimatter runs the
            # ripples the other way (charge conjugation cue)
            fil = fbm3(dir * 4.0 + wp.vec3(0.0, 0.0, time * 0.4), 4)
            streak = wp.pow(wp.max(fil, 0.0), 2.5)
            rdir = r * 6.5 - time * 3.0
            if anti == 1:
                rdir = r * 6.5 + time * 3.0
            ripple = 0.45 + 0.55 * wp.sin(rdir)
            fall = wp.exp(-r * 1.15)
            col = col + fcol * (streak * ripple * fall * 0.9 * dt)
        else:
            # neutrino: a faint ghostly shimmer streaking along the travel axis (z)
            axial = wp.exp(-(dir[0] * dir[0] + dir[1] * dir[1]) * 5.0)
            shim = 0.4 + 0.6 * wp.sin(p[2] * 5.0 - time * 4.0)
            wisp = 0.6 + 0.7 * fbm3(p * 2.2 + wp.vec3(0.0, 0.0, time), 3)
            fall = wp.exp(-r * 0.6)
            col = col + fcol * (axial * shim * wisp * fall * 0.5 * dt)
        t += dt

    # bright core (the charged point lepton); neutrinos have only a wisp
    if core_i > 0.0:
        tc = wp.max(wp.dot(-ro, rd), 0.0)
        dc = wp.length(-(ro + rd * tc))
        x = dc / core_r
        core = wp.exp(-x * x * 6.0) + 0.3 * wp.exp(-x * x * 1.5)
        cc = wp.vec3(1.0, 1.0, 1.0) * 0.6 + fcol * 0.4
        col = col + cc * (core * core_i)

    img[i, j] = col + void(rd)


def render_lepton(width, height, time, mouse, device, kind=0, anti=False):
    name, mass, gen = _LEP[kind]
    charged = 1 if kind < 3 else 0
    # neutrino flavour oscillation: blend the generation colour toward its
    # neighbours over time (νₑ ↔ ν_μ ↔ ν_τ)
    base = [(0.45, 0.8, 1.0), (0.55, 1.0, 0.65), (0.8, 0.6, 1.0)]
    if charged:
        c = base[gen]
        core_r = 0.1 + 0.02 * math.log10(mass * 1000.0)     # mass → core size
        core_i = 1.4 + 0.25 * math.log10(mass / 0.5)
        # antimatter (e⁺): a warm positive-charge field instead of the cool e⁻
        if anti:
            c = (0.95, 0.55, 0.22)
        fcol = wp.vec3(*c)
    else:
        ph = (time * 0.25 + float(gen)) % 3.0
        a = base[int(ph)]
        b = base[(int(ph) + 1) % 3]
        f = ph - float(int(ph))
        fcol = wp.vec3(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f,
                       a[2] + (b[2] - a[2]) * f)
        core_r, core_i = 0.1, 0.0

    cam = _render_mod.orbit_camera(width, height, time, mouse, dist=4.4, fov=40.0,
                                   el0=0.28)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(lepton_kernel, dim=(height, width),
              inputs=[img, cam, int(charged), fcol, float(core_r), float(core_i),
                      float(time), 0.0, int(1 if anti else 0), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    thr = 1.4 if charged else 1.0
    return _render_mod.finish(hdr, width, height, threshold=thr, strength=0.5,
                              exposure=1.0)
