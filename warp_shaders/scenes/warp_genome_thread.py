"""ONE-THREAD genome compression — one point cloud, one camera.

Opens on the REAL ray-marched GPU board for a moment, then the board granulates into its own particles and
the compression runs as ONE ordered thread: scanned into tokens, folded into the base-pair ladder, wound
into the double helix (more stages to come). Same ordered matter throughout; each stage folds the previous
end frame tighter (that shrink is the compression). The camera is fixed for playback but can be
rotated/zoomed (``mouse`` = orbit, ``render_view`` adds zoom) to inspect the thread.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..engine import post
from ..genome import thread as TH
from . import gpu_board as GB
from ..scene import Scene

_TH = TH.build(sub=1, block=5)
_N = _TH.n

_T_BOARD = 1.2                     # seconds of the real ray-marched board at the start
_T_XF = 0.5                        # board -> particles granulation
_PARTS = 13.0                      # seconds for the particle compression (card -> ... -> chromosome)
_T_REPL = 3.0                      # replication: the chromatid duplicates into its sister -> the X
TOTAL = _T_BOARD + _PARTS + _T_REPL


def _ss(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _rotz(p, ang, c):
    ca, sa = math.cos(ang), math.sin(ang)
    d = p - c
    x = d[:, 0] * ca - d[:, 1] * sa
    y = d[:, 0] * sa + d[:, 1] * ca
    return np.stack([x + c[0], y + c[1], d[:, 2] + c[2]], 1).astype(np.float32)

_pos = _col = None
_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)


@wp.kernel
def _splat(pos: wp.array(dtype=wp.vec3), zbuf: wp.array2d(dtype=wp.int32),
           width: int, height: int, ro: wp.vec3, uu: wp.vec3, vv: wp.vec3, ww: wp.vec3,
           zoom: float, rad_world: float, dnear: float, dfar: float):
    e = wp.tid()
    rel = pos[e] - ro
    cz = wp.dot(rel, ww)
    if cz < 0.05:
        return
    px = int(wp.round(zoom * wp.dot(rel, uu) / cz * float(height) + 0.5 * float(width) - 0.5))
    py = int(wp.round(0.5 * float(height) - 0.5 - zoom * wp.dot(rel, vv) / cz * float(height)))
    rad = int(wp.clamp(zoom * rad_world / cz * float(height), 1.0, 6.0))
    depthq = int(wp.clamp((cz - dnear) / (dfar - dnear) * 1022.0, 0.0, 1022.0))
    key = (depthq << _IDX_BITS) | e
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            if float(dx * dx + dy * dy) <= float(rad * rad) + 0.5:
                xx = px + dx
                yy = py + dy
                if xx >= 0 and xx < width and yy >= 0 and yy < height:
                    wp.atomic_min(zbuf, yy, xx, key)


@wp.kernel
def _resolve(zbuf: wp.array2d(dtype=wp.int32), col: wp.array(dtype=wp.vec3),
             img: wp.array2d(dtype=wp.vec3), width: int, height: int):
    i, j = wp.tid()
    bg = wp.vec3(0.017, 0.020, 0.030) * (1.0 - 0.5 * float(i) / float(height))
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    shade = 1.2 - 0.5 * (float((key >> _IDX_BITS) & 0x3FF) / 1022.0)
    img[i, j] = col[idx] * shade


@wp.kernel
def _cardcol_k(pos: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3), time: float):
    e = wp.tid()
    out[e] = GB.board_shade(pos[e], wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, -1.0, 0.0), 1.0, time)


_cardcol_done = False


def _ensure_cardcol(device):
    """Colour the card particles with the REAL board's own materials (sampled from gpu_board.board_shade
    at each voxel) so the ray-marched board granulates into particles of the same colour — seamless."""
    global _cardcol_done
    if _cardcol_done:
        return
    p = wp.array(np.ascontiguousarray(_TH.card, np.float32), dtype=wp.vec3, device=device)
    o = wp.zeros(_N, dtype=wp.vec3, device=device)
    wp.launch(_cardcol_k, dim=_N, inputs=[p, o, 0.0], device=device)
    wp.synchronize_device(device)
    _TH.col_card = np.clip(o.numpy() * 1.25 + 0.02, 0.0, 1.0).astype(np.float32)   # lift a touch to read
    _cardcol_done = True


def _cam(mx: float, my: float, zoomf: float):
    az = 0.62 + float(mx) * 0.010
    el = min(max(0.36 + float(my) * 0.006, 0.05), 1.45)
    dist = 9.2 / max(zoomf, 0.15)
    target = np.array([0.0, 0.55, 0.0], np.float32)
    ro = target + dist * np.array([math.cos(el) * math.sin(az), math.sin(el),
                                   math.cos(el) * math.cos(az)], np.float32)
    ww = target - ro; ww /= np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0, 1.0, 0.0], np.float32)); uu /= np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    return ro, uu, vv, ww, float(dist)


def _particles(W, H, pos_np, col_np, cam, device):
    pos_np = np.ascontiguousarray(pos_np, np.float32)
    col_np = np.ascontiguousarray(col_np, np.float32)
    m = pos_np.shape[0]
    pos = wp.array(pos_np, dtype=wp.vec3, device=device)
    col = wp.array(col_np, dtype=wp.vec3, device=device)
    ro, uu, vv, ww, dist = cam
    wcam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
            wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(_splat, dim=m, inputs=[pos, zbuf, W, H, *wcam, 1.7, 0.02, 1.5, dist + 12.0], device=device)
    wp.launch(_resolve, dim=(H, W), inputs=[zbuf, col, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = post.bloom(img.numpy(), threshold=0.9, strength=0.3, radius=3, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.0, preserve_hue=True)


def _board(W, H, time, cam, device):
    ro, uu, vv, ww, dist = cam
    camtuple = (ro, uu, vv, ww, dist)
    return np.clip(GB._render(W, H, time, (0.0, 0.0), device, cam=camtuple), 0.0, 1.0)


def _core(W, H, time, mx, my, zoomf, device):
    _ensure_cardcol(device)
    cam = _cam(mx, my, zoomf)
    t = float(time) % TOTAL
    if t < _T_BOARD:
        return _board(W, H, t, cam, device)                       # the real GPU, held
    if t < _T_BOARD + _T_XF:
        f = (t - _T_BOARD) / _T_XF                                 # board granulates into its own particles
        b = _board(W, H, _T_BOARD, cam, device)
        p0, c0 = TH.frame(_TH, 0.0)
        return np.clip((1.0 - f) * b + f * _particles(W, H, p0, c0, cam, device), 0.0, 1.0)
    pt = t - _T_BOARD
    if pt <= _PARTS:                                               # the compression: card -> chromosome
        pos, col = TH.frame(_TH, min(pt / _PARTS, 1.0))
        return _particles(W, H, pos, col, cam, device)
    # REPLICATION -> the metaphase X: the chromatid duplicates; the sister grows out of the centromere and
    # the two tilt apart, crossing at the centromere into the X.
    r = _ss((pt - _PARTS) / _T_REPL)
    pos, col = TH.frame(_TH, 1.0)                                  # the one chromatid
    c = np.array([0.0, 0.55, 0.0], np.float32)
    tilt = 0.42 * r
    a = _rotz(pos, +tilt, c)                                       # this chromatid leans one way
    b = c + r * (_rotz(pos, -tilt, c) - c)                         # its sister grows from the centromere
    posX = np.concatenate([a, b], 0)
    colX = np.concatenate([col, col], 0)
    return _particles(W, H, posX, colX, cam, device)


def render_view(width, height, time, mx, my, zoomf, device):
    return _core(int(width), int(height), time, mx, my, zoomf, device)


def _render(width, height, time, mouse, device):
    # playback: fixed camera (mouse orbit if a viewer passes it; zoom via render_view)
    return _core(int(width), int(height), time, float(mouse[0]), float(mouse[1]), 1.0, device)


SCENE = Scene(
    name="warp_genome_thread",
    description="The genome compression as ONE thread from one camera: the real GPU board granulates into "
                "its particles, scanned into tokens, folded into the base-pair ladder, wound into the double "
                "helix. Same ordered matter; each stage folds the previous end frame tighter.",
    renderer=_render,
)
