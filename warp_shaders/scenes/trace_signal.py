"""trace_signal — a real signal ringing down a trace on the RTX board (simulation of reality, B2).

The transmission-line substrate made concrete. A voltage pulse is launched from the **PCIe edge** and
propagates down a copper trace toward the **GPU die** as a real solution of the telegrapher's
equations (`sim/tline.py`). The trace ends are mismatched (open), so the pulse **reflects** and rings
back and forth — the classic signal-integrity picture — the voltage V(z) glowing along the trace on
the board (warm where V is high, cool where it swings negative). The wave speed, the reflection
coefficient, and the energy are the ones verified in `tests/test_tline.py` (open Γ=+1, matched Γ=0,
v=1/√(LC)). This is the solved signal, not a decorative streak.

`time` advances the line's step count; the signal is recomputed from the launch each frame.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..scene import Scene
from ..sim.tline import TransmissionLine
from .gpu_board import board_map, board_shade

_MAXD = 40.0
_CYCLE = 10.0
_NL = 384                    # transmission-line nodes
_MAXSTEP = 760              # steps across one cycle (pulse traverses ~2x)

# the trace on the board (board-local x,z): a long signal trace across the clean front edge,
# where a travelling pulse reads clearly against the plain green solder mask
_AX, _AZ = -2.9, -1.34      # near end
_BX, _BZ = 2.9, -1.34       # far end
_TRACE_HW = 0.075           # trace half-width (board units) for the glow footprint


def _build_signal(step):
    """Run the telegrapher solver `step` steps; return normalised V(z) along the trace."""
    tl = TransmissionLine(n=_NL, sc=0.9, R=0.006)     # a touch of series loss so the ringing decays
    tl.source(rs=np.inf)                              # open near end (edge) -> reflects
    tl.load(np.inf)                                   # open far end (die)  -> reflects (Gamma = +1)
    tl.launch_pulse(center=0.08, width=0.02, amp=1.0)
    tl.run(step)
    v = tl.V.astype(np.float32)
    m = float(np.abs(v).max())
    if m > 1e-6:
        v = v / m
    return np.ascontiguousarray(v)


@wp.func
def _spin(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.1 * wp.sin(time * 0.25)
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
def _render_kernel(img: wp.array2d(dtype=wp.vec3), volt: wp.array(dtype=float), nl: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, ax: float, az: float, bx: float, bz: float, hw: float):
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

    # distance from this board point to the trace segment A->B (in the board x-z plane)
    apx = sp[0] - ax
    apz = sp[2] - az
    abx = bx - ax
    abz = bz - az
    denom = abx * abx + abz * abz
    s = wp.clamp((apx * abx + apz * abz) / denom, 0.0, 1.0)
    cx = ax + s * abx
    cz = az + s * abz
    dist = wp.sqrt((sp[0] - cx) * (sp[0] - cx) + (sp[2] - cz) * (sp[2] - cz))
    face = wp.clamp(n[1], 0.0, 1.0)
    if dist < hw and face > 0.3:
        gi = int(s * float(nl - 1))
        vv = volt[gi]
        prof = wp.clamp(1.0 - dist / hw, 0.0, 1.0)     # taper across the trace width
        mag = wp.abs(vv) * prof
        warm = wp.vec3(1.0, 0.72, 0.32)
        cool = wp.vec3(0.32, 0.72, 1.0)
        tint = wp.vec3(0.0, 0.0, 0.0)
        if vv > 0.0:
            tint = warm
        else:
            tint = cool
        # a copper trace is always visible as a gold line; the travelling pulse rides it, white-hot
        base = wp.vec3(0.55, 0.42, 0.16) * prof
        col = col * (1.0 - 0.85 * prof) + base + tint * (mag * 3.2) + wp.vec3(1.0, 1.0, 1.0) * (mag * mag * 2.6)

    img[i, j] = col


def _progress_step(time):
    u = (float(time) % _CYCLE) / _CYCLE
    return int(4 + u * float(_MAXSTEP))


def _render(width, height, time, mouse, device):
    step = _progress_step(time)
    v = _build_signal(step)
    volt = wp.array(v, dtype=float, device=device)

    az = 0.5 + float(mouse[0]) * 0.006
    el = 0.66
    dist = 8.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.1, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, volt, _NL, eye, fwd, right, up, width, height, float(time), tanfov,
                      float(_AX), float(_AZ), float(_BX), float(_BZ), float(_TRACE_HW)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="trace_signal",
    description="a real voltage pulse launched from the PCIe edge and ringing down a copper trace to "
                "the GPU die on the RTX board — a live solution of the telegrapher's equations "
                "(sim/tline.py), reflecting off the mismatched (open) ends, V(z) glowing along the "
                "trace. Reflection Γ and wave speed verified in tests/test_tline.py.",
    renderer=_render,
)
