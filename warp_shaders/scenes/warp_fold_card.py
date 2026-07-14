"""warp_fold_card — the RTX 6000 Pro board folded into a mini-cube, in time.

This is warp compression applied to the *actual card*: the real `gpu_board` SDF (the RTX 6000 Pro
Blackwell board — exposed GPU die, GDDR7, VRM, PCIe edge) is rendered, then its domain is **folded**
— the long ends of the card fold inward over the middle and the board stacks in layers — condensing
the whole card into a compact glowing **mini-cube** of its own silicon and copper, then unfolding
back flat. The board's real materials (green solder mask, gold fingers, the die floorplan) fold with
it. `time` drives the fold forward (compress) and back (decompress).
"""

import math

import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..scene import Scene
from .gpu_board import board_map, board_shade

_MAXD = 40.0
_CYCLE = 10.0


@wp.func
def _tri(a: float, f: float) -> float:
    """Accordion (mirror) fold: map any coordinate into ``[0, f]`` by reflecting at each ``f``."""
    a = wp.abs(a)
    a = a - 2.0 * f * wp.floor(a / (2.0 * f))
    if a > f:
        a = 2.0 * f - a
    return a


@wp.func
def _warp(p: wp.vec3, fold: float) -> wp.vec3:
    """Fold the board's domain into the cube: the long card accordion-folds in x and z and the thin
    board stacks in y, so the material that fills the shrinking cube is folded board."""
    fxz = 10.0 * (1.0 - fold) + 0.95 * fold
    lh = 100.0 * (1.0 - fold) + 0.44 * fold
    # offsets scale with fold so the warp is the identity (pristine board) at fold = 0
    qx = _tri(p[0] + 3.4 * fold, fxz) - 0.5 * fxz * fold        # centred accordion fold in x
    qz = _tri(p[2] + 1.4 * fold, fxz) - 0.5 * fxz * fold        # and z
    qy = (p[1] - 0.1) - lh * wp.round((p[1] - 0.1) / lh) + 0.1  # stack the board in layers
    return wp.vec3(qx, qy, qz)


@wp.func
def _spin(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.3 * time
    ca = wp.cos(a); sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _cube_h(fold: float) -> wp.vec3:
    return wp.vec3(10.0 * (1.0 - fold) + 0.95 * fold,
                   10.0 * (1.0 - fold) + 0.7 * fold,
                   10.0 * (1.0 - fold) + 0.95 * fold)


@wp.func
def _fmap(p: wp.vec3, time: float, fold: float) -> float:
    sp = _spin(p, time)
    board = board_map(_warp(sp, fold))
    # intersect with a shrinking cube so the folded material is bounded into a mini-cube
    from_center = sp - wp.vec3(0.0, 0.1, 0.0)
    h = _cube_h(fold)
    dq = wp.vec3(wp.abs(from_center[0]) - h[0], wp.abs(from_center[1]) - h[1],
                 wp.abs(from_center[2]) - h[2])
    box = wp.length(wp.vec3(wp.max(dq[0], 0.0), wp.max(dq[1], 0.0), wp.max(dq[2], 0.0))) \
        + wp.min(wp.max(dq[0], wp.max(dq[1], dq[2])), 0.0)
    return wp.max(board, box)


@wp.func
def _fnormal(p: wp.vec3, time: float, fold: float) -> wp.vec3:
    e = 0.0012
    dx = _fmap(p + wp.vec3(e, 0.0, 0.0), time, fold) - _fmap(p - wp.vec3(e, 0.0, 0.0), time, fold)
    dy = _fmap(p + wp.vec3(0.0, e, 0.0), time, fold) - _fmap(p - wp.vec3(0.0, e, 0.0), time, fold)
    dz = _fmap(p + wp.vec3(0.0, 0.0, e), time, fold) - _fmap(p - wp.vec3(0.0, 0.0, e), time, fold)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _fao(p: wp.vec3, n: wp.vec3, time: float, fold: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _fmap(p + n * hr, time, fold)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3, right: wp.vec3,
                   up: wp.vec3, width: int, height: int, time: float, tanfov: float, fold: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(260):
        p = eye + rd * t
        d = _fmap(p, time, fold)
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
    n = _fnormal(p, time, fold)
    ao = _fao(p, n, time, fold)
    q = _warp(_spin(p, time), fold)
    col = board_shade(q, n, rd, ao, time)
    # warm rim glow along the fold seams so the compression reads as energetic
    seam = wp.pow(wp.clamp(1.0 - wp.abs(wp.dot(n, -rd)), 0.0, 1.0), 3.0)
    col = col + wp.vec3(1.0, 0.55, 0.2) * (seam * fold * 0.6)
    img[i, j] = col


def _progress(time):
    u = (float(time) % _CYCLE) / _CYCLE
    return 1.0 - abs(2.0 * u - 1.0)


def _render(width, height, time, mouse, device):
    fold = _progress(time)
    az = 0.6 + float(mouse[0]) * 0.006
    el = 0.5
    dist = 8.6 * (1.0 - fold) + 6.6 * fold          # ease in a little as the card compresses
    eye = wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el) + 0.15,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.0, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, float(time), tanfov, float(fold)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="warp_fold_card",
    description="the real RTX 6000 Pro board (gpu_board) folded into a mini-cube by warping its "
                "domain — the long ends of the card fold inward and the board stacks into a compact "
                "glowing cube of its own silicon and copper, then unfolds flat again.",
    renderer=_render,
)
