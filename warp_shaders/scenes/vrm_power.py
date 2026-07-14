"""vrm_power — the multiphase VRM delivering current to the die (simulation of reality, B3).

The board's power stage, made concrete. A 12-phase **buck converter** (`sim/vrm.py`) is solved to
steady state; the real per-phase **inductor currents** then pulse the VRM chokes on the board. Because
the phases are **interleaved** (phase k fires at offset k/12 of the switching cycle), the current
sweeps across the choke bank as a running "chaser" — the signature look of a multiphase VRM — and the
summed current glows into the **GPU die** as the ~1 V rail it drinks. The conversion V_out = D·Vin,
the ripple, and the L–C energy are the ones verified in `tests/test_vrm.py`.

`time` advances the switching phase; the currents are read from the solved steady-state cycle.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..scene import Scene
from ..sim.vrm import BuckConverter
from .gpu_board import board_map, board_shade

_MAXD = 40.0
_CYCLE = 10.0
_NPH = 12                    # VRM phases
_FSW = 1.0

# choke positions on the board's VRM bank (board-local x, z): two rows of six on the right side
_CHOKES = [(1.75 + 0.32 * i, z) for z in (0.55, -0.55) for i in range(6)]
_DIE = (-0.75, 0.05)


def _phase_table():
    """Solve the 12-phase buck to steady state; return per-phase current over one switching cycle."""
    b = BuckConverter(n_phases=_NPH, Vin=12.0, D=0.5, L=1.0, C=0.6, Rload=0.5, fsw=_FSW, dt=1e-3)
    b.run(20000)                                       # settle to steady state
    npc = int(round(1.0 / (_FSW * b.dt)))              # steps per switching cycle
    tab = np.empty((npc, _NPH), np.float32)
    for i in range(npc):
        b.step()
        tab[i] = b.iL
    tab = np.clip(tab, 0.0, None)                       # show forward current
    m = float(tab.max())
    if m > 1e-6:
        tab /= m
    return tab


_TABLE = _phase_table()
_NPC = _TABLE.shape[0]


def _currents_at(time):
    idx = int((time * _FSW % 1.0) * _NPC) % _NPC
    return np.ascontiguousarray(_TABLE[idx])


@wp.func
def _spin(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.1 * wp.sin(time * 0.2)
    ca = wp.cos(a); sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _fmap(p: wp.vec3, time: float) -> float:
    return board_map(_spin(p, time))


@wp.func
def _fnormal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.0012
    dx = _fmap(p + wp.vec3(e, 0.0, 0.0), time) - _fmap(p - wp.vec3(e, 0.0, 0.0), time)
    dy = _fmap(p + wp.vec3(0.0, e, 0.0), time) - _fmap(p - wp.vec3(0.0, e, 0.0), time)
    dz = _fmap(p + wp.vec3(0.0, 0.0, e), time) - _fmap(p - wp.vec3(0.0, 0.0, e), time)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _fao(p: wp.vec3, n: wp.vec3, time: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _fmap(p + n * hr, time)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), cx: wp.array(dtype=float), cz: wp.array(dtype=float),
                   cur: wp.array(dtype=float), nph: int, itot: float, eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, time: float, tanfov: float,
                   diex: float, diez: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(240):
        p = eye + rd * t
        d = _fmap(p, time)
        if d < 0.0007 * t + 0.0004:
            hit = 1
            break
        t += d * 0.8
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    sp = _spin(p, time)
    n = _fnormal(p, time)
    ao = _fao(p, n, time)
    col = board_shade(sp, n, rd, ao, time)
    face = wp.clamp(n[1], 0.0, 1.0)

    # per-phase choke glow (amber current), pulsing interleaved across the bank
    glow = float(0.0)
    for k in range(nph):
        dx = sp[0] - cx[k]
        dz = sp[2] - cz[k]
        glow += cur[k] * wp.exp(-(dx * dx + dz * dz) / 0.022)
    col = col + wp.vec3(1.0, 0.6, 0.18) * (glow * 3.0 * face) \
        + wp.vec3(1.0, 1.0, 1.0) * (glow * glow * 1.5 * face)

    # the die drinks the summed current — a bright pulsing warm core
    ddx = sp[0] - diex
    ddz = sp[2] - diez
    die = wp.exp(-(ddx * ddx + ddz * ddz) / 0.45) * itot
    col = col + wp.vec3(1.0, 0.8, 0.5) * (die * 1.3 * face)

    img[i, j] = col


def _render(width, height, time, mouse, device):
    cur = _currents_at(time)
    itot = float(cur.sum() / _NPH)
    cx = wp.array(np.array([c[0] for c in _CHOKES], np.float32), dtype=float, device=device)
    cz = wp.array(np.array([c[1] for c in _CHOKES], np.float32), dtype=float, device=device)
    curr = wp.array(cur, dtype=float, device=device)

    az = 0.62 + float(mouse[0]) * 0.006
    el = 0.6
    dist = 8.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.2, -0.1, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, cx, cz, curr, _NPH, itot, eye, fwd, right, up, width, height,
                      float(time), tanfov, float(_DIE[0]), float(_DIE[1])],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="vrm_power",
    description="the RTX board's 12-phase VRM delivering current to the GPU die — a real buck "
                "converter (sim/vrm.py) solved to steady state, its interleaved per-phase inductor "
                "currents pulsing the choke bank in a running chaser and glowing into the die as the "
                "~1 V rail. V_out=D·Vin, ripple, and L-C energy verified in tests/test_vrm.py.",
    renderer=_render,
)
