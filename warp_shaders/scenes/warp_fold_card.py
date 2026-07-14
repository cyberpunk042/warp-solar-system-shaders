"""warp_fold_card — C2: the RTX 6000 Pro board FOLDED (not torn) into a ~20x smaller cube.

Operator spec (verbatim): *"just fold it and dont care about the collision, it build the compression
image in the process and not at the result, a bit like docker image ... you have to fold it right and
squish it just right and you need a really 20x smaller cube of the total surface of the whole item of
compression."* And, sharply: *"YOU CANNOT TEAR THE CARD APPART... YOU MUST FOLD IT... FOLDING."*

So this is **real folding**. The card is one connected sheet. To fold it in half, the far half swings
about the crease line — a rigid hinge, an isometry — and the two halves stay **joined at the crease**
(the material bends around; it is never cut). Fold in half, and in half again, alternating the long
and short axis, **five times** (x, x, z, x, z — the exact schedule `warp_compress.foldcube` uses to
reach the target), stacking gaplessly into a compact laminated **cube**. The only thing ignored is
**self-collision**: as layers pile up they pass through each other's components — that overlap is what
takes the fold past the lossless limit down to the real board's **20.3x smaller total surface**
(22398 -> 1102 exposed voxel-faces, measured by the codec).

Each layer is a real strip of the card: the fold coordinate maps every point of the laminated block
back to where it came from on the flat board, so the green solder mask, the gold routing, the GDDR7
packages and the die are what fold and stack — connected across every crease. `time` runs the folds
forward (compress) and back (decompress): the compressed image is built **in the process**, one
crease-layer at a time, Docker-style.
"""

import math

import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..procedural.sdf import sd_round_box
from ..scene import Scene
from .gpu_board import board_shade

_MAXD = 40.0
_CYCLE = 12.0
_N = 5                          # five folds: in half, again, again, again, again
_TAU = 0.030                    # sheet half-pitch -> layers stack gaplessly (thin, squished "just right")
_XR = 3.7                       # board half-extent x
_ZR = 1.5                       # board half-extent z

# fold schedule (axis 0=x / 2=z, crease on that axis) — x,x,z,x,z, each keeping the (<= crease) half
_AX = wp.constant(wp.vec3(0.0, 0.0, 0.0))    # placeholder; per-fold data via the funcs below
_C0 = 0.0;   _AX0 = 0
_C1 = -1.85; _AX1 = 0
_C2 = 0.0;   _AX2 = 2
_C3 = -2.775; _AX3 = 0
_C4 = -0.75; _AX4 = 2
# centre of the finished laminated block (mean of the kept footprint), to frame it
_CX = -0.5 * (3.7 + 2.775)      # x kept in [-3.7,-2.775]
_CZ = -0.5 * (1.5 + 0.75)       # z kept in [-1.5,-0.75]
_CY = 0.5 * (2 ** _N) * _TAU    # centre of the 2^N-layer stack (top = 2^N*tau)


@wp.func
def _crease(m: int) -> float:
    if m == 0:
        return _C0
    if m == 1:
        return _C1
    if m == 2:
        return _C2
    if m == 3:
        return _C3
    return _C4


@wp.func
def _is_x(m: int) -> int:
    if m == 2:
        return 0                # fold 2 is the z-axis fold
    if m == 4:
        return 0                # fold 4 is the z-axis fold
    return 1                    # folds 0,1,3 are x-axis folds


@wp.func
def _pivh(m: int) -> float:
    # stack top before fold m = 2^m * tau  -> the flap rotates onto the top, gaplessly
    return wp.pow(2.0, float(m)) * _TAU


@wp.func
def _axis_coord(p: wp.vec3, m: int) -> float:
    if _is_x(m) == 1:
        return p[0]
    return p[2]


@wp.func
def _reflect(p: wp.vec3, m: int) -> wp.vec3:
    """The completed 180deg fold m as an isometry: reflect the flap across the crease and drop it onto
    the stack top (rotation about the crease line at height 2^m*tau)."""
    c = _crease(m)
    h = _pivh(m)
    x = p[0]; y = p[1]; z = p[2]
    if _is_x(m) == 1:
        x = 2.0 * c - x
    else:
        z = 2.0 * c - z
    y = 2.0 * h - y
    return wp.vec3(x, y, z)


@wp.func
def _unfold(p: wp.vec3, m: int) -> wp.vec3:
    """Undo the ``m`` completed folds (outermost first) -> the point's home on the flat board."""
    q = p
    for k in range(_N):
        j = _N - 1 - k
        if j < m:
            if q[1] > _pivh(j):
                q = _reflect(q, j)
    return q


@wp.func
def _hinge_back(p: wp.vec3, m: int, theta: float) -> wp.vec3:
    """Rotate ``p`` back by -theta about the crease line of the active fold m (axis-vs-y plane) — the
    inverse of the hinge swinging the far half up and over. At theta=pi this equals ``_reflect``."""
    c = _crease(m)
    h = _pivh(m)
    ca = wp.cos(theta); sa = wp.sin(theta)
    if _is_x(m) == 1:
        du = p[0] - c; dv = p[1] - h
        return wp.vec3(c + du * ca + dv * sa, h - du * sa + dv * ca, p[2])
    du = p[2] - c; dv = p[1] - h
    return wp.vec3(p[0], h - du * sa + dv * ca, c + du * ca + dv * sa)


@wp.func
def _sheet(q: wp.vec3) -> float:
    """The flat card as a thin connected sheet over its full footprint (rounded edges = the fold bend)."""
    return sd_round_box(q, wp.vec3(_XR, _TAU, _ZR), _TAU * 0.9)


@wp.func
def _xhi(m: int) -> float:
    # right edge of the kept x-footprint after m completed folds (x-folds at m=0,1,3)
    if m <= 0:
        return _XR
    if m == 1:
        return _C0
    if m <= 3:
        return _C1
    return _C3


@wp.func
def _zhi(m: int) -> float:
    # right edge of the kept z-footprint after m completed folds (z-folds at m=2,4)
    if m <= 2:
        return _ZR
    if m <= 4:
        return _C2
    return _C4


@wp.func
def _clip_xz(p: wp.vec3, xhi: float, zhi: float) -> float:
    """Distance outside the kept footprint box x in [-XR, xhi], z in [-ZR, zhi] (folded material only
    ever lands inside the kept quadrant — this discards the un-folded remainder)."""
    dx = wp.max(-_XR - p[0], p[0] - xhi)
    dz = wp.max(-_ZR - p[2], p[2] - zhi)
    return wp.max(dx, dz)


@wp.func
def _fmap(p: wp.vec3, m: int, theta: float) -> float:
    if m >= _N:
        d = _sheet(_unfold(p, _N))                          # fully folded cube
        return wp.max(d, _clip_xz(p, _xhi(_N), _zhi(_N)))
    c = _crease(m)
    ac = _axis_coord(p, m)
    base = wp.max(_sheet(_unfold(p, m)), ac - c)            # half that stays (<= crease)
    base = wp.max(base, _clip_xz(p, _xhi(m), _zhi(m)))      # ...clipped to the already-folded footprint
    q = _hinge_back(p, m, theta)
    acq = _axis_coord(q, m)
    flap = wp.max(_sheet(_unfold(q, m)), c - acq)           # half hinging over (>= crease), connected at c
    flap = wp.max(flap, _clip_xz(q, _xhi(m), _zhi(m)))
    return wp.min(base, flap)


@wp.func
def _shade_q(p: wp.vec3, m: int, theta: float) -> wp.vec3:
    """Board-local coord of the nearer half, so each layer paints its own strip of the real card."""
    if m >= _N:
        return _unfold(p, _N)
    c = _crease(m)
    base = wp.max(_sheet(_unfold(p, m)), _axis_coord(p, m) - c)
    base = wp.max(base, _clip_xz(p, _xhi(m), _zhi(m)))
    q = _hinge_back(p, m, theta)
    flap = wp.max(_sheet(_unfold(q, m)), c - _axis_coord(q, m))
    flap = wp.max(flap, _clip_xz(q, _xhi(m), _zhi(m)))
    if flap < base:
        return _unfold(q, m)
    return _unfold(p, m)


@wp.func
def _fnormal(p: wp.vec3, m: int, theta: float) -> wp.vec3:
    e = 0.0011
    dx = _fmap(p + wp.vec3(e, 0.0, 0.0), m, theta) - _fmap(p - wp.vec3(e, 0.0, 0.0), m, theta)
    dy = _fmap(p + wp.vec3(0.0, e, 0.0), m, theta) - _fmap(p - wp.vec3(0.0, e, 0.0), m, theta)
    dz = _fmap(p + wp.vec3(0.0, 0.0, e), m, theta) - _fmap(p - wp.vec3(0.0, 0.0, e), m, theta)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _fao(p: wp.vec3, n: wp.vec3, m: int, theta: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.008 + 0.04 * float(k)
        d = _fmap(p + n * hr, m, theta)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3, right: wp.vec3,
                   up: wp.vec3, width: int, height: int, time: float, tanfov: float,
                   m: int, theta: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(340):
        p = eye + rd * t
        d = _fmap(p, m, theta)
        if d < 0.0006 * t + 0.0003:
            hit = 1
            break
        t += d * 0.7
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    n = _fnormal(p, m, theta)
    ao = _fao(p, n, m, theta)
    q = _shade_q(p, m, theta)
    col = board_shade(q, n, rd, ao, time)
    # warm rim glow along the folded creases so the fold reads as it stacks
    seam = wp.pow(wp.clamp(1.0 - wp.abs(wp.dot(n, -rd)), 0.0, 1.0), 3.0)
    prog = wp.clamp(float(m) / float(_N), 0.0, 1.0)
    col = col + wp.vec3(1.0, 0.55, 0.2) * (seam * (0.25 + 0.5 * prog))
    img[i, j] = col


def _progress(time):
    u = (float(time) % _CYCLE) / _CYCLE
    return 1.0 - abs(2.0 * u - 1.0)


def _fold_state(prog):
    """(active-fold index m, hinge angle theta) for a compression amount in [0,1]."""
    s = prog * float(_N)
    m = int(math.floor(s))
    if m >= _N:
        return _N, 0.0
    frac = s - float(m)
    theta = (frac * frac * (3.0 - 2.0 * frac)) * math.pi     # smoothstep ease into each hinge, 0..pi
    return m, theta


def _render(width, height, time, mouse, device):
    prog = _progress(time)
    m, theta = _fold_state(prog)
    # frame the kept-footprint centre; zoom in as it condenses to the cube
    az = 0.62 + float(mouse[0]) * 0.006
    el = 0.42 * (1.0 - prog) + 0.92 * prog                   # rise to look down onto the folded cube
    dist = 9.6 * (1.0 - prog) + 3.6 * prog
    cx = -0.05 * (1.0 - prog) + _CX * prog
    cz = -0.05 * (1.0 - prog) + _CZ * prog
    cy = 0.0 * (1.0 - prog) + _CY * prog
    tgt = wp.vec3(cx, cy, cz)
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, float(time), tanfov,
                      int(m), float(theta)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="warp_fold_card",
    description="C2 — the real RTX 6000 Pro board FOLDED (never torn) into a ~20x smaller cube: one "
                "connected sheet creased in half five times (x,x,z,x,z), each half swinging about its "
                "crease and staying joined, layers stacking gaplessly, only self-collision ignored — "
                "the card's own board material folding and squishing into a compact laminated cube.",
    renderer=_render,
)
