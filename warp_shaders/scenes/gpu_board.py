"""GPU board — an advanced professional graphics board, no cover.

The real thing, stripped of every cosmetic: no shroud, no fans, no heatsink — just
the dense populated PCB of a workstation-class card (in the spirit of an RTX 6000
Pro Blackwell). One enormous exposed GPU die dominates the centre, flip-chip on its
substrate and ringed on three sides by a full set of GDDR7 memory packages. A heavy
multi-phase VRM — bank after bank of chokes with their MOSFET power stages and driver
ICs — turns the 12 V input into the hundreds of amps the die drinks, steadied by
arrays of MLCC and POSCAP capacitors packed around the die. A 12VHPWR connector feeds
it, a gold PCIe x16 edge plugs it in, and copper routing threads the whole board.
See ``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_subtract, op_union, sd_box, sd_cylinder
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 60.0
_GPUX = -0.75            # GPU centre x
_DIEH = wp.vec3(0.86, 0.05, 0.76)   # die half-extents


@wp.func
def _rot(p: wp.vec3, time: float, amt: float) -> wp.vec3:
    # amt scales the board's pose: 1 = the standalone tilted/rotated presentation; 0 = the raw board_map
    # frame (no rotation, no tilt) — used by the genome chain so the board overlays its own token voxels.
    a = (0.12 + 0.05 * wp.sin(time * 0.3)) * amt
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.16 * amt
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


@wp.func
def _pcb(q: wp.vec3) -> float:
    board = sd_box(q, wp.vec3(3.7, 0.06, 1.5)) - 0.01
    notch = sd_box(q - wp.vec3(0.9, 0.0, -1.5), wp.vec3(0.09, 0.2, 0.16))   # PCIe key
    board = op_subtract(board, notch)
    h0 = sd_cylinder(q - wp.vec3(-3.4, 0.0, 1.25), 0.2, 0.12)               # mount holes
    h1 = sd_cylinder(q - wp.vec3(3.45, 0.0, 1.25), 0.2, 0.12)
    board = op_subtract(board, h0)
    return op_subtract(board, h1)


@wp.func
def _gpu(q: wp.vec3) -> float:
    sub = sd_box(q - wp.vec3(_GPUX, 0.11, 0.05), wp.vec3(1.12, 0.05, 1.0)) - 0.01
    die = sd_box(q - wp.vec3(_GPUX, 0.18, 0.05), _DIEH) - 0.004
    return op_union(sub, die)


@wp.func
def _die_top(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(_GPUX, 0.18, 0.05), _DIEH) - 0.004


@wp.func
def _mem(q: wp.vec3) -> float:
    # GDDR7 packages ringing the GPU: top row, bottom row, left column
    tx = wp.clamp(wp.floor((q[0] - (-1.9)) / 0.56 + 0.5), 0.0, 4.0)
    top = sd_box(q - wp.vec3(-1.9 + 0.56 * tx, 0.12, 1.2), wp.vec3(0.24, 0.055, 0.22)) - 0.006
    bx = wp.clamp(wp.floor((q[0] - (-1.9)) / 0.56 + 0.5), 0.0, 4.0)
    bot = sd_box(q - wp.vec3(-1.9 + 0.56 * bx, 0.12, -1.1), wp.vec3(0.24, 0.055, 0.22)) - 0.006
    lz = wp.clamp(wp.floor((q[2] - (-0.55)) / 0.55 + 0.5), 0.0, 2.0)
    lft = sd_box(q - wp.vec3(-2.35, 0.12, -0.55 + 0.55 * lz), wp.vec3(0.2, 0.055, 0.24)) - 0.006
    return wp.min(wp.min(top, bot), lft)


@wp.func
def _chokes(q: wp.vec3) -> float:
    # two dense rows of VRM inductors on the right
    x0 = 0.7
    xi = wp.clamp(wp.floor((q[0] - x0) / 0.4 + 0.5), 0.0, 6.0)
    cx = x0 + 0.4 * xi
    r0 = sd_box(q - wp.vec3(cx, 0.16, 1.2), wp.vec3(0.15, 0.1, 0.15)) - 0.01
    r1 = sd_box(q - wp.vec3(cx, 0.16, 0.72), wp.vec3(0.15, 0.1, 0.15)) - 0.01
    return wp.min(r0, r1)


@wp.func
def _mosfets(q: wp.vec3) -> float:
    x0 = 0.7
    xi = wp.clamp(wp.floor((q[0] - x0) / 0.4 + 0.5), 0.0, 6.0)
    cx = x0 + 0.4 * xi
    m0 = sd_box(q - wp.vec3(cx, 0.11, 0.97), wp.vec3(0.14, 0.045, 0.08)) - 0.005
    m1 = sd_box(q - wp.vec3(cx, 0.11, 0.47), wp.vec3(0.14, 0.045, 0.08)) - 0.005
    return wp.min(m0, m1)


@wp.func
def _mlcc(q: wp.vec3) -> float:
    # dense MLCC field on the substrate border, skipping the die footprint
    xi = wp.floor(q[0] / 0.12 + 0.5)
    zi = wp.floor(q[2] / 0.12 + 0.5)
    cx = 0.12 * xi
    cz = 0.12 * zi
    onsub = wp.abs(cx - _GPUX) < 1.05 and wp.abs(cz - 0.05) < 0.92
    ondie = wp.abs(cx - _GPUX) < 0.92 and wp.abs(cz - 0.05) < 0.82
    if onsub and not ondie:
        return sd_box(q - wp.vec3(cx, 0.1, cz), wp.vec3(0.035, 0.02, 0.02))
    return 1e9


@wp.func
def _poscap(q: wp.vec3) -> float:
    # POSCAP array just off the die's -z edge + right edge
    xi = wp.clamp(wp.floor((q[0] - (-1.4)) / 0.4 + 0.5), 0.0, 3.0)
    a = sd_box(q - wp.vec3(-1.4 + 0.4 * xi, 0.1, -0.72), wp.vec3(0.16, 0.045, 0.08)) - 0.004
    zi = wp.clamp(wp.floor((q[2] - (-0.4)) / 0.4 + 0.5), 0.0, 2.0)
    b = sd_box(q - wp.vec3(0.55, 0.1, -0.4 + 0.4 * zi), wp.vec3(0.08, 0.045, 0.16)) - 0.004
    return wp.min(a, b)


@wp.func
def _bulk(q: wp.vec3) -> float:
    # bulk electrolytic cans near the power input
    c0 = sd_cylinder(q - wp.vec3(2.7, 0.2, -0.2), 0.16, 0.19)
    c1 = sd_cylinder(q - wp.vec3(3.2, 0.2, -0.2), 0.16, 0.19)
    return wp.min(c0, c1)


@wp.func
def _power(q: wp.vec3) -> float:
    # 12VHPWR / 12V-2x6 connector
    return sd_box(q - wp.vec3(3.15, 0.18, 0.95), wp.vec3(0.55, 0.13, 0.24)) - 0.01


@wp.func
def _ctrl(q: wp.vec3) -> float:
    a = sd_box(q - wp.vec3(1.4, 0.12, -1.05), wp.vec3(0.22, 0.05, 0.18)) - 0.008   # VRM controller
    b = sd_box(q - wp.vec3(2.3, 0.11, -1.05), wp.vec3(0.14, 0.04, 0.12)) - 0.006   # BIOS / support
    dv = wp.clamp(wp.floor((q[0] - 0.7) / 0.4 + 0.5), 0.0, 6.0)                     # gate-driver ICs
    drv = sd_box(q - wp.vec3(0.7 + 0.4 * dv, 0.1, 0.0), wp.vec3(0.06, 0.035, 0.09)) - 0.004
    return wp.min(wp.min(a, b), drv)


@wp.func
def _smd(q: wp.vec3) -> float:
    # dense field of tiny SMD resistors/caps filling the support-circuit strip on the left
    inleft = q[0] > -3.55 and q[0] < -2.62 and wp.abs(q[2]) < 1.28
    if not inleft:
        return 1e9
    xr = q[0] - 0.3 * wp.floor(q[0] / 0.3 + 0.5)
    zr = q[2] - 0.34 * wp.floor(q[2] / 0.34 + 0.5)
    return sd_box(wp.vec3(xr, q[1] - 0.09, zr), wp.vec3(0.05, 0.018, 0.1))


@wp.func
def _headers(q: wp.vec3) -> float:
    fan = sd_box(q - wp.vec3(-3.0, 0.14, 1.18), wp.vec3(0.2, 0.09, 0.12)) - 0.01     # fan header
    rgb = sd_box(q - wp.vec3(-3.0, 0.14, -1.18), wp.vec3(0.16, 0.08, 0.1)) - 0.01    # ARGB header
    return wp.min(fan, rgb)


@wp.func
def board_map(q: wp.vec3) -> float:
    """The whole populated board as one SDF, in board-local coords (no camera
    rotation applied). Reusable by other scenes (e.g. the nuke) that want to put
    the *real* board in world space and light / heat / destroy it."""
    d = op_union(_pcb(q), _gpu(q))
    d = op_union(d, _mem(q))
    d = op_union(d, _chokes(q))
    d = op_union(d, _mosfets(q))
    d = op_union(d, _mlcc(q))
    d = op_union(d, _poscap(q))
    d = op_union(d, _bulk(q))
    d = op_union(d, _power(q))
    d = op_union(d, _ctrl(q))
    d = op_union(d, _smd(q))
    return op_union(d, _headers(q))


@wp.func
def _map(p: wp.vec3, time: float, amt: float) -> float:
    return board_map(_rot(p, time, amt))


@wp.func
def _normal(p: wp.vec3, time: float, amt: float) -> wp.vec3:
    e = 0.0011
    dx = _map(p + wp.vec3(e, 0.0, 0.0), time, amt) - _map(p - wp.vec3(e, 0.0, 0.0), time, amt)
    dy = _map(p + wp.vec3(0.0, e, 0.0), time, amt) - _map(p - wp.vec3(0.0, e, 0.0), time, amt)
    dz = _map(p + wp.vec3(0.0, 0.0, e), time, amt) - _map(p - wp.vec3(0.0, 0.0, e), time, amt)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, time: float, amt: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _map(p + n * hr, time, amt)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.func
def _routing(lx: float, lz: float) -> float:
    # dense fine copper routing on exposed board areas (many parallel + orthogonal runs)
    # dense differential-pair routing: two close runs per period, both axes
    a = lx * 9.0
    fa = a - wp.floor(a)
    b = lz * 9.0
    fb = b - wp.floor(b)
    c = float(0.0)
    if fa < 0.1 or (fa > 0.2 and fa < 0.3):
        c = 1.0
    if (fb < 0.1 or (fb > 0.2 and fb < 0.3)) and lx > 0.5:
        c = 1.0
    return c


@wp.func
def _silk(lx: float, lz: float) -> float:
    """White silkscreen: board border, component courtyards, row ticks, labels."""
    w = 0.018
    s = float(0.0)
    # board border inset
    if wp.abs(wp.abs(lx) - 3.55) < w and wp.abs(lz) < 1.44:
        s = 1.0
    if wp.abs(wp.abs(lz) - 1.42) < w and wp.abs(lx) < 3.57:
        s = 1.0
    # die courtyard outline
    ddx = wp.abs(lx - _GPUX)
    ddz = wp.abs(lz - 0.05)
    if wp.abs(ddx - 1.2) < w and ddz < 1.06:
        s = 1.0
    if wp.abs(ddz - 1.06) < w and ddx < 1.2:
        s = 1.0
    # memory-row bracket lines (top + bottom) with per-module designator ticks
    if wp.abs(lz - 1.46) < 0.5 and lx > -2.2 and lx < 0.6:
        if wp.abs(lz - 1.47) < w:
            s = 1.0
        tk = lx - 0.56 * wp.floor(lx / 0.56 + 0.5)
        if wp.abs(tk) < 0.02 and wp.abs(lz - 1.44) < 0.06:
            s = 1.0
    # VRM label bracket (top-right)
    if wp.abs(lz - 1.47) < w and lx > 0.6 and lx < 3.4:
        s = 1.0
    # a solid silkscreen label block near the power connector
    if lx > 2.5 and lx < 3.5 and lz > 0.45 and lz < 0.62:
        s = 1.0
    # fiducial dots at two corners
    if wp.length(wp.vec2(lx + 3.3, lz + 1.2)) < 0.05:
        s = 1.0
    if wp.length(wp.vec2(lx - 3.3, lz - 1.2)) < 0.05:
        s = 1.0
    return s


@wp.func
def _dhash(a: float, b: float) -> float:
    h = wp.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return h - wp.floor(h)


@wp.func
def _die_look(dx: float, dz: float, time: float) -> wp.vec3:
    """Die-shot floorplan in die-local coords dx,dz in [-1,1]: SM/GPC compute
    array, a central L2-cache spine, and a memory-controller / PHY perimeter."""
    ax = wp.abs(dx)
    az = wp.abs(dz)
    # perimeter ring: memory controllers + I/O PHY, fine stripes perpendicular to edge
    if ax > 0.8 or az > 0.82:
        s = dz * 42.0
        if az > 0.82:
            s = dx * 46.0
        st = 0.5 + 0.5 * wp.sin(s)
        return wp.vec3(0.10, 0.20, 0.40) * (0.55 + 0.6 * st)
    # central L2-cache spine: two amber bands either side of centre
    if az < 0.12 or (az > 0.32 and az < 0.44):
        st = 0.5 + 0.5 * wp.sin(dx * 34.0)
        return wp.vec3(0.42, 0.30, 0.10) * (0.7 + 0.4 * st)
    # SM / GPC compute array — coarse clusters subdivided into fine cores
    gcx = dx * 6.0
    gcz = dz * 5.0
    ci = wp.floor(gcx)
    cj = wp.floor(gcz)
    act = 0.45 + 0.55 * (0.5 + 0.5 * wp.sin(time * 1.6 + _dhash(ci, cj) * 6.2832))
    fx = gcx - wp.floor(gcx)
    fz = gcz - wp.floor(gcz)
    core = 1.0
    fcx = dx * 46.0 - wp.floor(dx * 46.0)
    fcz = dz * 40.0 - wp.floor(dz * 40.0)
    if fcx < 0.14 or fcz < 0.14:
        core = 0.45                              # fine core-lane separations
    gpc = 1.0
    if fx < 0.07 or fz < 0.07:
        gpc = 0.25                               # thick GPC-cluster boundaries
    col = wp.vec3(0.10, 0.58, 0.80) * (act * core * gpc)
    return col + wp.vec3(0.02, 0.05, 0.09)


@wp.func
def board_shade(q: wp.vec3, n: wp.vec3, rd: wp.vec3, ao: float, time: float) -> wp.vec3:
    """Material colour for a board hit at board-local point q. Selects the nearest
    component and shades it (die floorplan, GDDR, VRM, caps, connectors, silkscreen,
    routing, PCB). Reusable so other scenes can render the *real* board and then
    overlay their own effects (heat, blasts)."""
    dgpu = _gpu(q)
    dmem = _mem(q)
    dch = _chokes(q)
    dmo = _mosfets(q)
    dml = _mlcc(q)
    dpo = _poscap(q)
    dbk = _bulk(q)
    dpw = _power(q)
    dct = _ctrl(q)
    dsm = _smd(q)
    dhd = _headers(q)
    dpc = _pcb(q)
    m0 = wp.min(wp.min(dgpu, dmem), wp.min(dch, dmo))
    m1 = wp.min(wp.min(dml, dpo), wp.min(dbk, dpw))
    m2 = wp.min(wp.min(dct, dsm), dhd)
    mind = wp.min(wp.min(m0, m1), wp.min(m2, dpc))
    eps = 0.0009
    out = wp.vec3(0.0, 0.0, 0.0)

    if dgpu <= mind + eps:
        if _die_top(q) <= dgpu + eps and q[1] > 0.18:
            ldx = (q[0] - _GPUX) / _DIEH[0]
            ldz = (q[2] - 0.05) / _DIEH[2]
            form = 0.4 + 0.6 * ao                                       # a little PBR form
            out = _die_look(ldx, ldz, time) * form                      # exposed die floorplan
        else:
            out = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.4    # substrate
    elif dmem <= mind + eps:
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        if q[1] > 0.16:
            mk = q[0] / 0.09 - wp.floor(q[0] / 0.09)
            if mk < 0.12:
                col = col * 0.6                                          # package marking
        out = col                                                       # GDDR7 modules
    elif dch <= mind + eps:
        out = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.82        # VRM chokes
    elif dmo <= mind + eps:
        out = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.9         # MOSFET stages
    elif dml <= mind + eps:
        col = ec.lit(n, rd, 6, ao, wp.vec3(0.0, 0.0, 0.0))
        out = wp.cw_mul(col, wp.vec3(0.9, 0.8, 0.65))                    # MLCC caps
    elif dpo <= mind + eps:
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        if q[1] > 0.12:
            col = col + wp.vec3(0.18, 0.12, 0.03)                        # POSCAP top stripe
        out = col
    elif dbk <= mind + eps:
        out = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))              # bulk cans
    elif dpw <= mind + eps:
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        if q[1] > 0.24:
            sx = q[0] / 0.12 - wp.floor(q[0] / 0.12)                    # connector pins
            if sx < 0.5:
                col = col * 0.4
        out = col
    elif dct <= mind + eps:
        out = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.85        # controller ICs / drivers
    elif dsm <= mind + eps:
        # tiny SMD field: alternate dark resistors and tan caps by cell hash
        h = _dhash(wp.floor(q[0] / 0.3 + 0.5), wp.floor(q[2] / 0.34 + 0.5))
        if h > 0.5:
            out = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.9     # SMD resistor
        else:
            col = ec.lit(n, rd, 6, ao, wp.vec3(0.0, 0.0, 0.0))
            out = wp.cw_mul(col, wp.vec3(0.85, 0.78, 0.62))              # SMD cap
    elif dhd <= mind + eps:
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        if q[1] > 0.2:
            hx = q[0] / 0.08 - wp.floor(q[0] / 0.08)
            if hx < 0.5:
                col = col + wp.vec3(0.28, 0.2, 0.05)                     # header pins (gold)
        out = col
    else:
        # dark professional PCB: gold PCIe fingers on -z edge, dense routing elsewhere
        if q[2] < -1.28 and q[1] > -0.02:
            fx = q[0] / 0.1 - wp.floor(q[0] / 0.1)
            if fx > 0.28 and wp.abs(q[0] - 0.9) > 0.15:
                out = ec.lit(n, rd, 2, ao, wp.vec3(0.0, 0.0, 0.0))
            else:
                out = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.4
        elif q[1] > 0.04 and n[1] > 0.5 and _silk(q[0], q[2]) > 0.5:
            out = wp.vec3(0.72, 0.74, 0.72) * (0.6 + 0.4 * ao)          # white silkscreen
        elif q[1] > 0.05 and n[1] > 0.5 and _routing(q[0], q[2]) > 0.5:
            out = ec.lit(n, rd, 2, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.5    # copper routing
        else:
            out = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.26   # dark pro PCB
    return out


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, spin: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(240):
        p = eye + rd * t
        d = _map(p, time, spin)
        if d < 0.0007 * t + 0.0004:
            hit = 1
            break
        t += d * 0.82
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    n = _normal(p, time, spin)
    ao = _ao(p, n, time, spin)
    img[i, j] = board_shade(_rot(p, time, spin), n, rd, ao, time)


def _render(width, height, time, mouse, device, cam=None):
    if cam is None:
        # standalone: the board's own orbiting/tilted 3/4 presentation
        az = 0.14 + float(mouse[0]) * 0.01
        el = 0.78 + float(mouse[1]) * 0.005
        dist = 8.4
        eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                      dist * math.sin(el) + 0.15,
                      dist * math.cos(el) * math.cos(az))
        tgt = wp.vec3(0.0, -0.12, 0.0)
        fwd = wp.normalize(tgt - eye)
        right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
        up = wp.cross(right, fwd)
        tanfov = math.tan(math.radians(42.0) * 0.5)
        spin = 1.0
    else:
        # chain: render from the genome chain's locked splat camera, in the RAW board_map frame (spin=0),
        # so the solid card overlays its own token voxels exactly and can fragment into them in place.
        ro, uu, vv, ww, _dist = cam
        eye = wp.vec3(float(ro[0]), float(ro[1]), float(ro[2]))
        fwd = wp.vec3(float(ww[0]), float(ww[1]), float(ww[2]))
        right = wp.vec3(float(uu[0]), float(uu[1]), float(uu[2]))
        up = wp.vec3(float(vv[0]), float(vv[1]), float(vv[2]))
        tanfov = 0.5 / 1.7          # match the splat stages' zoom=1.7 so the card is the tokens' size
        spin = 0.0

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov, spin],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.7, strength=0.3)


SCENE = Scene(
    name="gpu_board",
    description="an advanced workstation GPU board with no cover — a huge exposed "
                "die ringed by GDDR7 memory, a dense multi-phase VRM (chokes, MOSFETs, "
                "drivers), MLCC/POSCAP/bulk cap arrays, a 12VHPWR connector, gold PCIe "
                "x16 fingers, and heavy copper routing. The real board, no cosmetics.",
    renderer=_render,
)
