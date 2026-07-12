"""Beta-minus decay — n → p + e⁻ + ν̄ₑ, animated.

The Standard Model's weak force in action: inside a **neutron** (udd) a **down
quark flips to an up quark** (blue→red), emitting a **W⁻** boson; the W⁻ then
**decays** into an **electron** (cyan) and an **antineutrino** (a faint wisp) that
fly apart — leaving a **proton** (uud). Loops every ~8 s; animate with
``--frames``. See ``docs/research/21-standard-model.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..scene import Scene
from ..subatomic import render as _render_mod
from ..subatomic.field import bag_glow, color_charge, quark_emit, tube_emit, void

_CONF = 1.0
_QSIZE = 0.34


@wp.kernel
def beta_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera,
                q0: wp.vec3, q1: wp.vec3, q2: wp.vec3, qcol2: wp.vec3,
                bound_i: float, wpos: wp.vec3, wI: float, epos: wp.vec3, eI: float,
                nupos: wp.vec3, nuI: float, warm: float, time: float,
                width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    g = _rs(ro, rd, 4.2)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return
    t0 = wp.max(g[0], 0.0)
    dt = (g[1] - t0) / 66.0
    c0 = color_charge(0)
    c1 = color_charge(1)
    tint = wp.vec3(1.0, 0.9, 0.75) * warm + wp.vec3(0.8, 0.9, 1.0) * (1.0 - warm)
    col = wp.vec3(0.0, 0.0, 0.0)
    t = t0 + dt * 0.5
    for _ in range(66):
        p = ro + rd * t
        e = quark_emit(p, q0, _QSIZE, c0, time, 1.3)
        e = e + quark_emit(p, q1, _QSIZE, c1, time, 5.1) * 0.72
        e = e + quark_emit(p, q2, _QSIZE, qcol2, time, 9.7)
        # confinement tubes (fade as the nucleon settles)
        if bound_i > 0.01:
            gt = wp.vec3(0.8, 0.92, 1.0)
            e = e + (tube_emit(p, q0, q1, 0.1, gt, time)
                     + tube_emit(p, q1, q2, 0.1, gt, time)
                     + tube_emit(p, q2, q0, 0.1, gt, time)) * bound_i
        e = e + bag_glow(p, wp.vec3(0.0, 0.0, 0.0), _CONF, tint)
        # the W- boson (orange), while it exists
        if wI > 0.01:
            dw = wp.length(p - wpos)
            e = e + wp.vec3(1.0, 0.55, 0.25) * (wp.exp(-(dw / 0.3) * (dw / 0.3) * 2.0) * wI)
        # the electron (cyan core)
        if eI > 0.01:
            de = wp.length(p - epos)
            e = e + wp.vec3(0.5, 0.82, 1.0) * (wp.exp(-(de / 0.22) * (de / 0.22) * 3.0) * eI)
        # the antineutrino (faint wisp)
        if nuI > 0.01:
            dn = wp.length(p - nupos)
            e = e + wp.vec3(0.7, 0.75, 0.9) * (wp.exp(-(dn / 0.4) * (dn / 0.4)) * nuI)
        col = col + e * dt
        t += dt
    img[i, j] = col * 1.8 + void(rd)


def _triad(k, t):
    ang = float(k) * 2.0943951 + t * 0.4
    return np.array([0.6 * math.cos(ang), 0.24 * math.sin(t + k * 2.1),
                     0.6 * math.sin(ang)], np.float32)


def _render(width, height, time, mouse, device, period=8.0):
    prog = (time % period) / period
    t_flip, t_wdec = 0.33, 0.55

    q0 = _triad(0, time); q1 = _triad(1, time); q2 = _triad(2, time)
    # q2: down (blue) → up (red) at the flip
    blue = np.array([0.3, 0.45, 1.0], np.float32)
    red = np.array([1.0, 0.2, 0.2], np.float32)
    f = float(np.clip((prog - t_flip + 0.04) / 0.08, 0.0, 1.0))
    qcol2 = blue * (1.0 - f) + red * f

    wdir = np.array([1.0, 0.5, 0.35], np.float32); wdir /= np.linalg.norm(wdir)
    wdecay_pos = q2 + wdir * 1.3
    wI = 0.0; wpos = wdecay_pos.copy()
    eI = 0.0; nuI = 0.0
    epos = wdecay_pos.copy(); nupos = wdecay_pos.copy()
    bound_i = 1.0

    if prog < t_flip:
        bound_i = 1.0
    elif prog < t_wdec:                                   # W travelling out
        s = (prog - t_flip) / (t_wdec - t_flip)
        wpos = q2 + wdir * (0.4 + s * 0.9)
        wI = 1.4 * math.sin(s * math.pi)
        bound_i = 1.0
    else:                                                 # W decays → e + nu fly apart
        s = (prog - t_wdec) / (1.0 - t_wdec)
        edir = np.array([1.4, 0.7, 0.1], np.float32); edir /= np.linalg.norm(edir)
        ndir = np.array([0.4, 1.0, -0.7], np.float32); ndir /= np.linalg.norm(ndir)
        epos = wdecay_pos + edir * (s * 2.4)
        nupos = wdecay_pos + ndir * (s * 2.6)
        eI = 2.0; nuI = 0.8
        bound_i = 1.0

    az = 0.5 + time * 0.15
    eye = (4.6 * math.cos(0.3) * math.sin(az), 4.6 * math.sin(0.3),
           4.6 * math.cos(0.3) * math.cos(az))
    cam = make_camera(eye, (0.4, 0.2, 0.2), fov_deg=44.0, aspect=width / height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(beta_kernel, dim=(height, width),
              inputs=[img, cam, wp.vec3(*q0), wp.vec3(*q1), wp.vec3(*q2),
                      wp.vec3(*qcol2), float(bound_i), wp.vec3(*wpos), float(wI),
                      wp.vec3(*epos), float(eI), wp.vec3(*nupos), float(nuI),
                      float(1.0), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    return _render_mod.finish(hdr, width, height, threshold=1.4, strength=0.5,
                              exposure=0.95)


SCENE = Scene(
    name="beta_decay",
    description="Beta-minus decay n→p+e⁻+ν̄ₑ — a down quark flips to up (blue→red) "
                "emitting a W⁻ that decays into an electron (cyan) and an "
                "antineutrino (wisp) flying apart. --frames animates the loop.",
    renderer=_render,
)
