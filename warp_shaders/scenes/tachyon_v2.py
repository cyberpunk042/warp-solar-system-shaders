"""Tachyon v2 — a rising ray, a central orb, two spiralling rays into a cone.

An alternative, more iconic tachyon: a bright **ray climbing bottom-up** along the axis
with a luminous **orb** at the centre, and **two rays spiralling** around it, flaring
outward toward a **conic destination** at the top. The motion lives in the glow — pulses
travelling up the beam and along the spirals — and in the slow rotation of the two helices.
A companion to `tachyon` (the Cherenkov-cone version). Animate with ``--frames``. See
``docs/research/21-standard-model.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electric as el
from ..engine import post
from ..scene import Scene

_TOP = 2.3
_BOT = -2.3
_TURNS = 3.5
_FLARE = 1.15          # how wide the cone opens at the top


def _helix(strand, n, time, rot):
    pts = []
    for k in range(n):
        u = k / float(n - 1)
        h = _BOT + (_TOP - _BOT) * u
        ang = strand + u * _TURNS * 2.0 * math.pi + time * rot
        rad = 0.14 + _FLARE * u                      # flares outward -> cone at the top
        pts.append([rad * math.cos(ang), h, rad * math.sin(ang)])
    return pts


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), beam: wp.array(dtype=wp.vec3), nb: int,
                   spir: wp.array(dtype=wp.vec3), ns: int, pulse: wp.array(dtype=wp.vec3),
                   npz: int, eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3,
                   width: int, height: int, tanfov: float, time: float, pulselvl: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    # deep void with a faint vertical shimmer
    up_ = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    col = wp.vec3(0.006, 0.008, 0.020) * (1.0 - up_) + wp.vec3(0.012, 0.016, 0.035) * up_

    # central rising ray (cyan) — a glowing filament up the axis
    gb = float(0.0)
    for k in range(nb):
        gb += el.pt_glow(eye, rd, beam[k], 0.03)
    col += wp.vec3(0.35, 0.85, 1.0) * (wp.clamp(gb, 0.0, 3.0) * 0.9)

    # the two spiralling rays (violet-blue) with bright white cores
    gs = float(0.0)
    cs = float(0.0)
    for k in range(ns):
        gs += el.pt_glow(eye, rd, spir[k], 0.045)
        cs += el.pt_glow(eye, rd, spir[k], 0.018)
    col += wp.vec3(0.6, 0.4, 1.0) * (wp.clamp(gs, 0.0, 3.0) * 1.1)
    col += wp.vec3(0.85, 0.9, 1.0) * (wp.clamp(cs, 0.0, 4.0) * 0.9)

    # travelling pulses (bright) racing up the beam + spirals
    gp = float(0.0)
    for k in range(npz):
        gp += el.pt_glow(eye, rd, pulse[k], 0.06)
    col += wp.vec3(0.9, 0.95, 1.0) * (wp.clamp(gp, 0.0, 4.0) * (1.2 + 0.8 * pulselvl))

    # the central orb — a pulsing luminous core
    orbr = 0.34 + 0.05 * wp.sin(time * 3.0)
    col += wp.vec3(0.7, 0.85, 1.0) * (el.corona(eye, rd, wp.vec3(0.0, 0.0, 0.0), orbr) * (1.4 + 0.5 * pulselvl))
    col += wp.vec3(1.0, 1.0, 1.0) * (el.corona(eye, rd, wp.vec3(0.0, 0.0, 0.0), 0.13) * 2.2)
    img[i, j] = col


def _render(width, height, time, mouse, device):
    rot = 0.8
    beam = [[0.0, _BOT + (_TOP - _BOT) * (k / 139.0), 0.0] for k in range(140)]
    spir = _helix(0.0, 240, time, rot) + _helix(math.pi, 240, time, rot)

    pulses = []
    # pulses climbing the central beam
    for pk in range(3):
        u = ((time * 0.6 + pk / 3.0) % 1.0)
        pulses.append([0.0, _BOT + (_TOP - _BOT) * u, 0.0])
    # pulses running up each spiral
    for strand in (0.0, math.pi):
        for pk in range(2):
            u = ((time * 0.5 + pk / 2.0) % 1.0)
            h = _BOT + (_TOP - _BOT) * u
            ang = strand + u * _TURNS * 2.0 * math.pi + time * rot
            rad = 0.14 + _FLARE * u
            pulses.append([rad * math.cos(ang), h, rad * math.sin(ang)])

    barr, nb = el.upload_points(np.asarray(beam, dtype=np.float32), device)
    sarr, ns = el.upload_points(np.asarray(spir, dtype=np.float32), device)
    parr, npz = el.upload_points(np.asarray(pulses, dtype=np.float32), device)
    pulselvl = 0.5 + 0.5 * math.sin(time * 4.0)

    az = 0.5 + math.sin(time * 0.12) * 0.25 + float(mouse[0]) * 0.01
    el_ang = 0.06 + float(mouse[1]) * 0.005
    dist = 7.4
    eye = wp.vec3(dist * math.cos(el_ang) * math.sin(az), 0.4 + dist * math.sin(el_ang),
                  dist * math.cos(el_ang) * math.cos(az))
    tgt = wp.vec3(0.0, 0.3, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(52.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, barr, nb, sarr, ns, parr, npz, eye, fwd, right, up, width,
                      height, tanfov, time, float(pulselvl)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.022))
    hdr = post.bloom(hdr, threshold=0.8, strength=0.65, radius=r, passes=4, octaves=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="tachyon_v2",
    description="a tachyon reimagined — a bright ray climbing the axis with a luminous orb "
                "at the centre and two rays spiralling around it, flaring outward toward a "
                "conic destination, pulses of glow racing up the beam and the spirals. "
                "Animate with --frames.",
    renderer=_render,
)
