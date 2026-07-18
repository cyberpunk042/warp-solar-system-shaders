"""heat_die — the RTX board heating up, by the real heat equation (simulation of reality, B5).

The consequence layer, made concrete. Almost all the power the card draws becomes heat at the die, so
a real **heat-diffusion** simulation (`sim/heat.py`) runs on the board plane with the **die floorplan
as its power source** (a concentrated hotspot over the compute-dense die, a broader warm die body, a
warm VRM). From cold, the die heats, a **hotspot forms**, and the heat **spreads** into the PCB and
the surrounding memory before the cooler (a lumped convective term) carries it away to a bounded
steady state — then, under "idle", it cools back down. The temperature glows on the board as a
blackbody heat ramp (dark → red → orange → yellow → white). The diffusion, the steady state, and the
first-law energy balance are the ones verified in `tests/test_heat.py`.

`time` runs one load→idle cycle; the field is re-simulated from cold each frame so the frame *is* the
temperature field at that instant.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..scene import Scene
from ..sim.heat import HeatField
from .gpu_board import board_map, board_shade

_MAXD = 40.0
_CYCLE = 10.0
_NG = 150                    # heat grid resolution
_MAXSTEP = 2600             # steps over one cycle (heat up, then cool)
_TAMB = 20.0
_TRANGE = 72.0             # temperature above ambient that maps to "white hot"

_BX, _BZ = 3.7, 1.5


def _b2u(x):
    return (x + _BX) / (2.0 * _BX)


def _b2v(z):
    return (z + _BZ) / (2.0 * _BZ)


def _build_temp(step):
    """Simulate the board thermal field to `step`: source ON for the first half (load), OFF after
    (idle → cool). Returns the temperature grid."""
    h = HeatField(n=_NG, alpha=0.25, kappa=0.02, t_amb=_TAMB, boundary="neumann")
    # die floorplan power map (board coords): a hot compute core + broader die + a warm VRM
    h.add_source_gauss(_b2u(-0.75), _b2v(0.05), power=3.2, radius=0.055)   # compute hotspot
    h.add_source_rect(_b2u(-1.7), _b2v(-0.85), _b2u(0.2), _b2v(0.95), 0.55)  # the die body
    h.add_source_rect(_b2u(1.7), _b2v(-1.1), _b2u(3.3), _b2v(1.1), 0.35)    # the VRM warms too
    q_saved = h.q.copy()
    half = _MAXSTEP // 2
    on = min(step, half)
    h.run(on)
    if step > half:
        h.q[:] = 0.0                        # idle: source off, cooler pulls it back to ambient
        h.run(step - half)
    return np.ascontiguousarray(h.T.astype(np.float32))


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


@wp.func
def _heat_ramp(x: float) -> wp.vec3:
    """Blackbody-ish heat ramp: 0 -> dark, .3 red, .6 orange, .85 yellow, 1 white."""
    r = wp.clamp(x * 2.2, 0.0, 1.0)
    g = wp.clamp((x - 0.35) * 2.2, 0.0, 1.0)
    b = wp.clamp((x - 0.72) * 3.2, 0.0, 1.0)
    return wp.vec3(r, g, b)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), temp: wp.array2d(dtype=float), gn: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, bx: float, bz: float, tamb: float, trange: float):
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

    fu = (sp[0] + bx) / (2.0 * bx)
    fv = (sp[2] + bz) / (2.0 * bz)
    if fu >= 0.0 and fu < 1.0 and fv >= 0.0 and fv < 1.0:
        gi = int(fu * float(gn))
        gj = int(fv * float(gn))
        heat = wp.clamp((temp[gi, gj] - tamb) / trange, 0.0, 1.2)
        ramp = _heat_ramp(heat)
        # hot silicon glows: darken the cool board slightly, add the incandescent ramp
        col = col * (1.0 - 0.35 * wp.clamp(heat, 0.0, 1.0)) + ramp * (heat * 1.6)

    img[i, j] = col


def _progress_step(time):
    u = (float(time) % _CYCLE) / _CYCLE
    return int(u * float(_MAXSTEP))


def _render(width, height, time, mouse, device):
    step = _progress_step(time)
    T = _build_temp(step)
    temp = wp.array(T, dtype=float, device=device)

    az = 0.55 + float(mouse[0]) * 0.006
    el = 0.6
    dist = 8.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(-0.2, -0.1, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, temp, _NG, eye, fwd, right, up, width, height, float(time), tanfov,
                      float(_BX), float(_BZ), float(_TAMB), float(_TRANGE)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="heat_die",
    description="the real RTX board heating up by the heat equation (sim/heat.py): the die floorplan "
                "is the power source, a hotspot forms over the compute die and spreads into the PCB "
                "and memory before the cooler bounds it, then it cools on idle — temperature glowing "
                "as a blackbody heat ramp. Diffusion, steady state and first-law balance verified in "
                "tests/test_heat.py.",
    renderer=_render,
)
