"""Process 4 scene — the double helices bead up into nucleosomes ("beads on a string").

Chains directly from Process 3: it starts from the exact wound double helices Process 3 produced and
wraps them. A nucleosome is ~one double helix's worth of DNA (~150 base pairs) coiled ~1.75 turns around
a histone core, with linker DNA reaching to the next bead — so the 1663 helices become **1663 beads on a
string** (the count barely changes; the win is that each tall helix wraps down into a compact bead).

Conserving and physical: every base pair is reused — the middle of each helix wraps the bead, the two ends
are the linker to its neighbours; no point is created or teleports, each moves continuously from its helix
onto its bead. The camera holds a fixed course (a slow tilt down, no spin) as the tall forest of helices
collapses into the flat carpet of beads-on-a-string — the whole field, the whole wrap, in frame.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..genome import wrap_nucleosomes
from ..scene import Scene

_NC = wrap_nucleosomes(sub=2, block=5)
_P = _NC.n_pairs
_SAMPLES = 4                      # per pair: ribbon edge A, ribbon edge B, 2 interior points
_M = _P * _SAMPLES
_NB = _NC.n_beads                 # one histone core per bead, splatted after the DNA (indices _M.._M+_NB)
_CORE_R = float(_NC.core_radius * 0.82)   # the octamer sits just inside the wrapped DNA ring

_ha = _hb = _na = _nb = _a_col = _b_col = _ctr = None


def _ensure(device):
    global _ha, _hb, _na, _nb, _a_col, _b_col, _ctr
    if _ha is None:
        _ha = wp.array(_NC.helix_a, dtype=wp.vec3, device=device)
        _hb = wp.array(_NC.helix_b, dtype=wp.vec3, device=device)
        _na = wp.array(_NC.nuc_a, dtype=wp.vec3, device=device)
        _nb = wp.array(_NC.nuc_b, dtype=wp.vec3, device=device)
        _a_col = wp.array(_NC.a_col, dtype=wp.vec3, device=device)
        _b_col = wp.array(_NC.b_col, dtype=wp.vec3, device=device)
        _ctr = wp.array(_NC.centers, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)
_BACKBONE = wp.constant(wp.vec3(0.46, 0.53, 0.66))   # sugar-phosphate backbones — muted so they don't blow out
_HISTONE = wp.constant(wp.vec3(0.30, 0.40, 0.56))    # the histone octamer core the DNA wraps — a muted protein blue
_MBASE = wp.constant(int(_P * 4))                     # element index where the core splats begin


@wp.kernel
def _wrap_kernel(
    ha: wp.array(dtype=wp.vec3),
    hb: wp.array(dtype=wp.vec3),
    na: wp.array(dtype=wp.vec3),
    nb: wp.array(dtype=wp.vec3),
    a_col: wp.array(dtype=wp.vec3),
    b_col: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    to_bead: float,         # 0 = Process-3 wound helices, 1 = wrapped into nucleosome beads
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    zoom: float,
    dnear: float,
    dfar: float,
):
    e = wp.tid()
    pr = e / 4
    s = e - pr * 4

    pa = wp.lerp(ha[pr], na[pr], to_bead)              # continuous: helix backbone -> bead ribbon edge
    pb = wp.lerp(hb[pr], nb[pr], to_bead)

    if s == 0:
        pos = pa
        col = _BACKBONE
    elif s == 1:
        pos = pb
        col = _BACKBONE
    elif s == 2:
        pos = wp.lerp(pa, pb, 0.36)
        col = a_col[pr]
    else:
        pos = wp.lerp(pa, pb, 0.64)
        col = b_col[pr]
    elemcol[e] = col

    rel = pos - ro
    cz = wp.dot(rel, ww)
    if cz < 0.05:
        return
    cx = wp.dot(rel, uu)
    cy = wp.dot(rel, vv)
    pfx = zoom * cx / cz * float(height_px) + 0.5 * float(width) - 0.5
    pfy = 0.5 * float(height_px) - 0.5 - zoom * cy / cz * float(height_px)
    px = int(wp.round(pfx))
    py = int(wp.round(pfy))

    base = 0.030
    if s >= 2:
        base = 0.016
    rpx = zoom * base / cz * float(height_px)
    rad = int(wp.clamp(rpx, 1.0, 6.0))

    depthq = int(wp.clamp((cz - dnear) / (dfar - dnear) * 1022.0, 0.0, 1022.0))
    key = (depthq << _IDX_BITS) | e

    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            if float(dx * dx + dy * dy) <= float(rad * rad) + 0.5:
                xx = px + dx
                yy = py + dy
                if xx >= 0 and xx < width and yy >= 0 and yy < height_px:
                    wp.atomic_min(zbuf, yy, xx, key)


@wp.kernel
def _core_kernel(
    ctr: wp.array(dtype=wp.vec3),
    elemcol: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    width: int,
    height_px: int,
    to_bead: float,
    core_r: float,
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    zoom: float,
    dnear: float,
    dfar: float,
):
    bid = wp.tid()
    if to_bead < 0.02:                                    # no core while still an unwrapped helix
        return
    pos = ctr[bid]
    e = _MBASE + bid
    elemcol[e] = _HISTONE

    rel = pos - ro
    cz = wp.dot(rel, ww)
    if cz < 0.05:
        return
    cx = wp.dot(rel, uu)
    cy = wp.dot(rel, vv)
    pfx = zoom * cx / cz * float(height_px) + 0.5 * float(width) - 0.5
    pfy = 0.5 * float(height_px) - 0.5 - zoom * cy / cz * float(height_px)
    px = int(wp.round(pfx))
    py = int(wp.round(pfy))

    rpx = zoom * (core_r * to_bead) / cz * float(height_px)
    rad = int(wp.clamp(rpx, 1.0, 9.0))

    depthq = int(wp.clamp((cz - dnear) / (dfar - dnear) * 1022.0, 0.0, 1022.0))
    key = (depthq << _IDX_BITS) | e
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            if float(dx * dx + dy * dy) <= float(rad * rad) + 0.5:
                xx = px + dx
                yy = py + dy
                if xx >= 0 and xx < width and yy >= 0 and yy < height_px:
                    wp.atomic_min(zbuf, yy, xx, key)


@wp.kernel
def _resolve_kernel(
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height_px: int,
):
    i, j = wp.tid()
    yy = float(i) / float(height_px)
    bg = wp.vec3(0.016, 0.020, 0.030) * (1.0 - 0.45 * yy)
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1022.0
    shade = 1.28 - 1.12 * depthq
    fog = wp.clamp((depthq - 0.20) * 2.3, 0.0, 0.94)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


def _schedule(time: float):
    """Whole process on one timeline: hold the helices, then wrap them into beads-on-a-string."""
    u = min(max((time - 0.5) / 4.0, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _camera(time: float):
    # fixed course, no spin: as the tall helices wrap down into the flat bead carpet, the camera dollies in
    # and tilts down a little to look over the beads-on-a-string.
    u = min(max((time - 0.3) / 4.2, 0.0), 1.0)
    u = u * u * (3.0 - 2.0 * u)
    dist = 44.0 * (1.0 - u) + 26.0 * u
    ty = 0.0
    target = np.array([0.0, ty, 0.0], np.float32)
    direction = np.array([0.10, 0.20 + 0.14 * u, 1.0], np.float32)   # end lower/grazing so beads read as beads
    direction = direction / np.linalg.norm(direction)
    ro = target + dist * direction
    ww = target - ro
    ww = ww / np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0, 1.0, 0.0], np.float32))
    uu = uu / np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    return ro, uu, vv, ww, dist


def _render(width, height, time, mouse, device, cam=None):
    _ensure(device)
    W, H = int(width), int(height)
    to_bead = _schedule(float(time))
    if cam is None:
        ro, uu, vv, ww, dist = _camera(float(time))
    else:
        ro, uu, vv, ww, dist = cam
    dnear = 1.5
    dfar = float(dist) + 34.0

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M + _NB, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _wrap_kernel,
        dim=_M,
        inputs=[_ha, _hb, _na, _nb, _a_col, _b_col, zbuf, elemcol, W, H,
                float(to_bead), *cam, 1.7, dnear, dfar],
        device=device,
    )
    wp.launch(
        _core_kernel,
        dim=_NB,
        inputs=[_ctr, elemcol, zbuf, W, H, float(to_bead), _CORE_R, *cam, 1.7, dnear, dfar],
        device=device,
    )
    wp.launch(_resolve_kernel, dim=(H, W), inputs=[zbuf, elemcol, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=0.9, strength=0.35, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.0, preserve_hue=True)
    ldr = post.vignette(ldr, amount=0.3)
    return ldr


SCENE = Scene(
    name="warp_nucleosome",
    description=(
        "Process 4 — nucleosomes. The wound double helices from Process 3 wrap into beads on a string: "
        "each helix's DNA coils ~1.75 turns around a histone core, its ends the linker to the next bead, "
        "so the 1663 helices become 1663 nucleosome beads. Conserving: chained from Process 3's actual "
        "output, every base pair reused, nothing spawned, fixed camera, the whole field wrapping in frame."
    ),
    renderer=_render,
)
