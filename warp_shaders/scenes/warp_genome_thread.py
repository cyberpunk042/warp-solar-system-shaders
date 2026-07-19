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

# We perfect ONE stage at a time. `_END_STAGE` is the stage the take stops at (held, then loops) so the
# later stages don't clutter the view while we work. Set it to the stage we're refining.
# ---- V3 TIMELINE. Each row: (label, engine-progress at the END of this beat, seconds to MOVE there,
# seconds to HOLD, camera azimuth, camera elevation). The camera angle eases from the previous beat's angle
# to this one over the move; the distance always auto-fits the thread's current size, so it dollies in with
# the compression. Holds sit on the money shots (the board, the base-pair forest, the feed, the X). ----
_TL = [
    # label        prog   move  hold   az     el
    ("gpu_board",   0.00,  0.0,  1.8,   0.62,  0.34),   # the real GPU, held
    ("scan",        0.14,  4.2,  0.3,   0.60,  0.27),   # granulation sweep (slow — key)
    ("bind",        0.26,  1.9,  0.4,   0.66,  0.24),   # tokens bind in twos
    ("pair",        0.40,  2.3,  1.2,   0.72,  0.17),   # base-pair forest (grazing), hold
    ("helix",       0.52,  2.5,  0.9,   0.50,  0.16),   # helix forest (low)
    ("nucleo",      0.66,  2.5,  0.9,   0.60,  0.30),   # nucleosome beads
    ("fibre",       0.82,  2.5,  0.9,   0.56,  0.34),   # fibre coil
    ("chromo",      1.00,  6.0,  2.6,   0.05,  0.50),   # feed through the tips into BOTH X strands (front-on)
]


def _build_timeline():
    segs = []
    t, ps, paz, pel = 0.0, 0.0, _TL[0][4], _TL[0][5]
    for label, pe, mv, hd, az, el in _TL:
        segs.append(dict(label=label, t=t, mv=mv, hd=hd, ps=ps, pe=pe, az0=paz, el0=pel, az=az, el=el))
        t += mv + hd
        ps, paz, pel = pe, az, el
    return segs, t


_SEGS_TL, TOTAL = _build_timeline()
_SEG = [(s["label"],) for s in _SEGS_TL]                 # progress-bar segments for the viewer
_START = [s["t"] for s in _SEGS_TL]


def _timeline(time):
    """Map global time -> (label, engine-progress, camera az, el, replication-fraction)."""
    tt = float(time) % TOTAL if TOTAL > 0 else 0.0
    seg = _SEGS_TL[0]
    for s in _SEGS_TL:
        if tt >= s["t"]:
            seg = s
    local = tt - seg["t"]
    f = _ss(local / seg["mv"]) if seg["mv"] > 1e-6 and local < seg["mv"] else 1.0
    prog = seg["ps"] + (seg["pe"] - seg["ps"]) * f
    az = seg["az0"] + (seg["az"] - seg["az0"]) * f
    el = seg["el0"] + (seg["el"] - seg["el0"]) * f
    rr = f if seg["label"] == "replicate_X" else 0.0
    return seg["label"], prog, az, el, rr


def _ss(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _with_rungs(pos, col):
    """Add the base-pair RUNG between each pair's two backbone tokens — two fill points coloured by the
    pair's bases — so the strand reads as base pairs (coloured A-T/G-C rungs), not loose dots."""
    a, b = _TH.a_tok, _TH.b_tok
    pa, pb = pos[a], pos[b]
    m1 = pa * 0.66 + pb * 0.34
    m2 = pa * 0.34 + pb * 0.66
    cm = 0.5 * (col[a] + col[b])
    return (np.concatenate([pos, m1, m2], 0).astype(np.float32),
            np.concatenate([col, cm, cm], 0).astype(np.float32))


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
             pos: wp.array(dtype=wp.vec3), img: wp.array2d(dtype=wp.vec3), width: int, height: int,
             ro: wp.vec3, uu: wp.vec3, vv: wp.vec3, ww: wp.vec3, zoom: float, rad_world: float):
    i, j = wp.tid()
    bg = wp.vec3(0.017, 0.020, 0.030) * (1.0 - 0.5 * float(i) / float(height))
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    # shade the winning particle as a LIT SPHERE: recover its screen centre + radius, take the pixel's
    # offset as the sphere normal, light it. Flat dots -> rounded, lit 3-D beads.
    rel = pos[idx] - ro
    cz = wp.dot(rel, ww)
    sx = zoom * wp.dot(rel, uu) / cz * float(height) + 0.5 * float(width) - 0.5
    sy = 0.5 * float(height) - 0.5 - zoom * wp.dot(rel, vv) / cz * float(height)
    rad = wp.max(zoom * rad_world / cz * float(height), 1.0)
    ox = (float(j) - sx) / rad
    oy = (float(i) - sy) / rad
    nz = wp.sqrt(wp.max(1.0 - ox * ox - oy * oy, 0.0))
    nrm = wp.normalize(wp.vec3(ox, -oy, nz + 0.35))
    diff = wp.max(wp.dot(nrm, wp.normalize(wp.vec3(0.45, 0.65, 0.55))), 0.0)
    rim = wp.pow(1.0 - nz, 2.5) * 0.22
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1022.0
    dsh = 1.12 - 0.42 * depthq
    lit = col[idx] * ((0.32 + 0.82 * diff) * dsh) + wp.vec3(rim, rim, rim) * dsh
    fog = wp.clamp((depthq - 0.35) * 1.7, 0.0, 0.9)           # far strands recede into the dark -> depth
    img[i, j] = wp.lerp(lit, bg, fog)


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


def _cam(pos, az, el, mx: float, my: float, zoomf: float):
    # AUTO-FRAME at the timeline's per-stage angle: aim at the thread's centre, distance from its current
    # extent (so the camera eases inward with the compression), az/el from the timeline (each stage from its
    # best angle). User orbit (mx,my) + wheel zoom ride on top. One continuous motivated move.
    c = pos.mean(0).astype(np.float32)
    r = float(np.percentile(np.linalg.norm(pos - c, axis=1), 99)) + 1e-3   # full extent — don't clip
    dist = max(r * 4.2, 1.2) / max(zoomf, 0.15)                            # zoomed OUT, lots of margin
    azf = float(az) + float(mx) * 0.010
    elf = min(max(float(el) + float(my) * 0.006, 0.05), 1.45)
    ro = c + dist * np.array([math.cos(elf) * math.sin(azf), math.sin(elf),
                              math.cos(elf) * math.cos(azf)], np.float32)
    ww = c - ro; ww /= np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0, 1.0, 0.0], np.float32)); uu /= np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    return ro, uu, vv, ww, float(dist)


def _particles_masked(W, H, pos_np, col_np, cam, device):
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
    wp.launch(_resolve, dim=(H, W), inputs=[zbuf, col, pos, img, W, H, *wcam, 1.7, 0.02], device=device)
    wp.synchronize_device(device)
    cover = zbuf.numpy() != 0x7FFFFFFF                    # which pixels a particle actually covers
    hdr = post.bloom(img.numpy(), threshold=0.9, strength=0.3, radius=3, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.0, preserve_hue=True), cover


def _particles(W, H, pos_np, col_np, cam, device):
    return _particles_masked(W, H, pos_np, col_np, cam, device)[0]


def _board(W, H, time, cam, device, cut_x=-1.0e9):
    return np.clip(GB._render(W, H, time, (0.0, 0.0), device, cam=tuple(cam), cut_x=cut_x), 0.0, 1.0)


def _core(W, H, time, mx, my, zoomf, device):
    _ensure_cardcol(device)
    label, prog, az, el, _rr = _timeline(time)
    if label == "gpu_board":                                     # the real GPU, held; frame the whole card
        pos0, _ = TH.frame(_TH, 0.0)
        return _board(W, H, 0.0, _cam(pos0, az, el, mx, my, zoomf), device)
    pos, col = TH.frame(_TH, prog)
    cam = _cam(pos, az, el, mx, my, zoomf)                        # auto-frame at the timeline angle
    if prog <= TH.scan_end():
        # GRANULATION SCAN: the board erodes behind the wavefront, its matter releasing as token particles.
        front = TH.scan_front(_TH, prog)
        board = _board(W, H, 0.0, cam, device, cut_x=front)
        pos2 = pos.copy()
        pos2[pos2[:, 0] > front] = cam[0]                         # hide still-solid-board particles (cull)
        part, cover = _particles_masked(W, H, pos2, col, cam, device)
        return np.where(cover[..., None], part, board)
    posr, colr = _with_rungs(pos, col)                           # draw the base-pair rungs
    if label == "chromo":
        # feed directly into BOTH final strands of the metaphase X: this chromatid + its mirror sister (the
        # replicated pair). Same feed g -> both build together as the strand reels through the tips.
        posB = posr.copy(); posB[:, 0] = -posB[:, 0]
        posr = np.concatenate([posr, posB], 0)
        colr = np.concatenate([colr, colr], 0)
        cam = _cam(posr, az, el, mx, my, zoomf)                   # reframe for the whole X
    return _particles(W, H, posr, colr, cam, device)


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
