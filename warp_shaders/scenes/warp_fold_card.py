"""warp_fold_card — the RTX 6000 Pro board folded in half, and in half again, into a cube.

Warp compression applied to the *actual card*. The real `gpu_board` SDF (the RTX 6000 Pro
Blackwell board) is folded the way you fold paper: **fold it in half, then in half again on the
other axis, and again** — three folds — each one a visible hinge where one half of the card lifts,
swings over, and lands stacked on the other half. Three halvings (x -> z -> x) take the long card
down to a compact **cube** of eight stacked layers of its own board. `time` runs the folds forward
(compress) and back (decompress).

The mechanism: the folds already completed are a static reflected **stack** (a domain unfold); the
one fold currently happening is a rigid **hinge** — the moving half is `board_map` evaluated on the
point rotated back about the crease, so the real board (green solder mask, gold routing, GDDR7, the
die) is what lifts and stacks.
"""

import math

import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..scene import Scene
from .gpu_board import board_map, board_shade

_MAXD = 40.0
_CYCLE = 10.0
_N = 3                       # three folds: in half, in half, in half

# fold schedule (board is x in +/-3.7, z in +/-1.5, thin in y, centred at origin)
_T = 0.26                    # layer spacing -> 8 layers, height ~2.1, ~cube with the ~1.85x1.5 footprint
_C0 = 0.0                    # fold 0: axis x, crease x=0     (right half onto left)
_C1 = 0.0                    # fold 1: axis z, crease z=0
_C2 = -1.85                  # fold 2: axis x, crease x=-1.85 (right half onto left again)
_PIV0 = _T                   # hinge pivot height = top of the stack before that fold
_PIV1 = 2.0 * _T
_PIV2 = 4.0 * _T
# centre of the finished 8-layer block, to recentre it in view as it forms
_BCX = -2.775
_BCY = 4.0 * _T
_BCZ = -0.75


@wp.func
def _unfold(p: wp.vec3, m: int) -> wp.vec3:
    """Undo the ``m`` completed folds (outermost first): reflect-and-drop each stacked half back
    down, so a point in the folded stack maps to its original spot on the flat board."""
    x = p[0]
    y = p[1]
    z = p[2]
    if m > 2:
        if y > _PIV2:                       # fold 2 (axis x)
            x = 2.0 * _C2 - x
            y = 2.0 * _PIV2 - y
    if m > 1:
        if y > _PIV1:                       # fold 1 (axis z)
            z = 2.0 * _C1 - z
            y = 2.0 * _PIV1 - y
    if m > 0:
        if y > _PIV0:                       # fold 0 (axis x)
            x = 2.0 * _C0 - x
            y = 2.0 * _PIV0 - y
    return wp.vec3(x, y, z)


@wp.func
def _crease(m: int) -> float:
    if m == 0:
        return _C0
    if m == 1:
        return _C1
    return _C2


@wp.func
def _pivot(m: int) -> float:
    if m == 0:
        return _PIV0
    if m == 1:
        return _PIV1
    return _PIV2


@wp.func
def _is_x(m: int) -> int:
    if m == 1:
        return 0            # fold 1 is the z-axis fold
    return 1                # folds 0 and 2 are x-axis folds


@wp.func
def _rot_back(p: wp.vec3, m: int, theta: float) -> wp.vec3:
    """Rotate ``p`` back by ``-theta`` about the crease line of fold ``m`` (in the fold axis vs y
    plane) — the inverse of the hinge that swings the moving half up and over."""
    c = _crease(m)
    piv = _pivot(m)
    ca = wp.cos(theta)
    sa = wp.sin(theta)
    if _is_x(m) == 1:
        du = p[0] - c
        dv = p[1] - piv
        return wp.vec3(c + du * ca + dv * sa, piv - du * sa + dv * ca, p[2])
    du = p[2] - c
    dv = p[1] - piv
    return wp.vec3(p[0], piv - du * sa + dv * ca, c + du * ca + dv * sa)


@wp.func
def _axis_coord(p: wp.vec3, m: int) -> float:
    if _is_x(m) == 1:
        return p[0]
    return p[2]


@wp.func
def _fmap(p: wp.vec3, m: int, theta: float) -> float:
    if m >= _N:
        return board_map(_unfold(p, _N))            # fully folded cube
    c = _crease(m)
    base = wp.max(board_map(_unfold(p, m)), _axis_coord(p, m) - c)   # half that stays (<= crease)
    q = _rot_back(p, m, theta)
    flap = wp.max(board_map(_unfold(q, m)), c - _axis_coord(q, m))   # half hinging over (>= crease)
    return wp.min(base, flap)


@wp.func
def _shade_q(p: wp.vec3, m: int, theta: float) -> wp.vec3:
    """Board-local coord at the hit point, from whichever half (base or flap) is nearer."""
    if m >= _N:
        return _unfold(p, _N)
    c = _crease(m)
    base = wp.max(board_map(_unfold(p, m)), _axis_coord(p, m) - c)
    q = _rot_back(p, m, theta)
    flap = wp.max(board_map(_unfold(q, m)), c - _axis_coord(q, m))
    if flap < base:
        return _unfold(q, m)
    return _unfold(p, m)


@wp.func
def _fnormal(p: wp.vec3, m: int, theta: float) -> wp.vec3:
    e = 0.0012
    dx = _fmap(p + wp.vec3(e, 0.0, 0.0), m, theta) - _fmap(p - wp.vec3(e, 0.0, 0.0), m, theta)
    dy = _fmap(p + wp.vec3(0.0, e, 0.0), m, theta) - _fmap(p - wp.vec3(0.0, e, 0.0), m, theta)
    dz = _fmap(p + wp.vec3(0.0, 0.0, e), m, theta) - _fmap(p - wp.vec3(0.0, 0.0, e), m, theta)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _fao(p: wp.vec3, n: wp.vec3, m: int, theta: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _fmap(p + n * hr, m, theta)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3, right: wp.vec3,
                   up: wp.vec3, width: int, height: int, time: float, tanfov: float,
                   off: wp.vec3, m: int, theta: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(320):
        p = eye + rd * t + off
        d = _fmap(p, m, theta)
        if d < 0.0006 * t + 0.0004:
            hit = 1
            break
        t += d * 0.7
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t + off
    n = _fnormal(p, m, theta)
    ao = _fao(p, n, m, theta)
    q = _shade_q(p, m, theta)
    col = board_shade(q, n, rd, ao, time)
    # warm rim glow along the fold creases so the compression reads as energetic
    seam = wp.pow(wp.clamp(1.0 - wp.abs(wp.dot(n, -rd)), 0.0, 1.0), 3.0)
    prog = wp.clamp(float(m) / float(_N), 0.0, 1.0)
    col = col + wp.vec3(1.0, 0.55, 0.2) * (seam * prog * 0.5)
    img[i, j] = col


def _progress(time):
    u = (float(time) % _CYCLE) / _CYCLE
    return 1.0 - abs(2.0 * u - 1.0)


def _fold_state(prog):
    """(recenter offset, active-fold index m, hinge angle theta) for a compression amount in [0,1]."""
    s = prog * float(_N)
    m = int(math.floor(s))
    if m > _N:
        m = _N
    if m >= _N:
        theta = 0.0
    else:
        frac = s - float(m)
        theta = (frac * frac * (3.0 - 2.0 * frac)) * math.pi     # smoothstep ease into each hinge
    off = wp.vec3(_BCX * prog, _BCY * prog, _BCZ * prog)
    return off, m, theta


def _render(width, height, time, mouse, device):
    prog = _progress(time)
    off, m, theta = _fold_state(prog)
    az = 0.6 + float(mouse[0]) * 0.006
    el = 0.42
    dist = 9.2 * (1.0 - prog) + 6.0 * prog
    eye = wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el) + 0.2,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.0, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, float(time), tanfov,
                      off, int(m), float(theta)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="warp_fold_card",
    description="the real RTX 6000 Pro board (gpu_board) folded in half three times — each fold a "
                "visible hinge where one half swings over and stacks — condensing the long card into "
                "a compact cube of eight stacked layers of its own board, then unfolding flat again.",
    renderer=_render,
)
