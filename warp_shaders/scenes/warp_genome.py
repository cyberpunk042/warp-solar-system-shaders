"""The whole genome process — one long, continuous animation from base pairs to the chromosome.

This is the master sequence. It does not re-simulate anything: it takes the **actual output arrays** of
every genome process (all the same 182 872 base pairs, in the same order, because each process chains from
the last) and morphs continuously through them, so you watch the DNA condense the whole way down:

    Process 2  base pairs          — the raw paired field
      → Process 3  double helices  — the pairs wind into 1663 helices
      → Process 4  nucleosomes     — helices wrap into beads on a string
      → Process 5  30 nm fibre     — beads coil into 47 chromatin fibres
      → Process 6  telomeres       — the strand's two ends curl into t-loop caps
      → Process 7  the chromosome  — the fibre coils into one condensed chromatid

Every base pair is conserved and simply *moved* from one process's position to the next (a smooth lerp with
ease), nothing is spawned or destroyed. The camera pulls back and re-frames each stage automatically (the
base-pair field is huge, the chromosome tiny), holding a fixed 3/4 course — a slow reveal, no gratuitous
spin — and settles on the finished, pretty coiled chromosome.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..genome.basepair import bind_pairs
from ..genome.chromosome import fold_chromosome
from ..genome.fibre import coil_fibre
from ..genome.helix import wind_helix, wound_positions
from ..genome.nucleosome import wrap_nucleosomes
from ..genome.telomere import cap_telomeres
from ..scene import Scene

_SUB, _BLOCK = 2, 5

# --- assemble the real keyframes: every process's actual output, same pairs, same order ----------------
_bp = bind_pairs(sub=_SUB, block=_BLOCK)
_hx = wind_helix(sub=_SUB, block=_BLOCK)
_h_a, _h_b = wound_positions(_hx)
_nu = wrap_nucleosomes(sub=_SUB, block=_BLOCK)
_fb = coil_fibre(sub=_SUB, block=_BLOCK)
_tl = cap_telomeres(sub=_SUB, block=_BLOCK)
_cr = fold_chromosome(sub=_SUB, block=_BLOCK)

_KA = np.stack([_bp.a_pos, _h_a, _nu.nuc_a, _fb.bead_a, _tl.tel_a, _cr.chr_a]).astype(np.float32)  # (6,P,3)
_KB = np.stack([_bp.b_pos, _h_b, _nu.nuc_b, _fb.bead_b, _tl.tel_b, _cr.chr_b]).astype(np.float32)
_STAGES = _KA.shape[0]
_P = _KA.shape[1]
_SAMPLES = 4
_M = _P * _SAMPLES
_COLA = _cr.a_col.astype(np.float32)                     # telomere-tinted base colours, consistent through-line
_COLB = _cr.b_col.astype(np.float32)

# per-stage centroid + full extent so the camera can auto-fit each stage
_CENT = np.zeros((_STAGES, 3), np.float32)
_RAD = np.zeros(_STAGES, np.float32)
for _k in range(_STAGES):
    _pts = np.concatenate([_KA[_k], _KB[_k]], 0)
    _c = _pts.mean(0)
    _CENT[_k] = _c
    _RAD[_k] = float(np.percentile(np.linalg.norm(_pts - _c, axis=1), 96.0))

# per-stage CAMERA: for the fine scales (helices, nucleosomes) we fly IN to a small window so the actual
# structure is visible up close; for the big scales we frame the whole thing. Direction reveals each layout.
#          base pairs      helices        nucleosomes    30 nm fibre    telomeres      chromosome
_STAGE_DIR = np.array([
    [0.28, 0.52, 1.0], [0.44, 0.30, 1.0], [0.44, 0.28, 1.0],
    [0.30, 0.34, 1.0], [0.32, 0.20, 1.0], [0.42, 0.16, 1.0],
], np.float32)
_STAGE_FRAME = np.array([7.0, 3.4, 3.6, 11.0, _RAD[4], _RAD[5]], np.float32)  # fly IN for the fine scales

_KAf = _KA.reshape(_STAGES * _P, 3)                      # flat: stage k, pair pr → k*P + pr
_KBf = _KB.reshape(_STAGES * _P, 3)

_ka = _kb = _cola = _colb = None


def _ensure(device):
    global _ka, _kb, _cola, _colb
    if _ka is None:
        _ka = wp.array(_KAf, dtype=wp.vec3, device=device)
        _kb = wp.array(_KBf, dtype=wp.vec3, device=device)
        _cola = wp.array(_COLA, dtype=wp.vec3, device=device)
        _colb = wp.array(_COLB, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)
_BACKBONE = wp.constant(wp.vec3(0.60, 0.66, 0.80))


@wp.kernel
def _morph_kernel(
    ka: wp.array(dtype=wp.vec3),
    kb: wp.array(dtype=wp.vec3),
    cola: wp.array(dtype=wp.vec3),
    colb: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    n_pairs: int,
    seg: int,
    f: float,
    base_pt: float,
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

    ia = seg * n_pairs + pr
    ib = (seg + 1) * n_pairs + pr
    pa = wp.lerp(ka[ia], ka[ib], f)
    pb = wp.lerp(kb[ia], kb[ib], f)

    ac = cola[pr]
    bc = colb[pr]
    tel = ac[1] > 0.9

    if s == 0:
        pos = pa
        col = _BACKBONE
        if tel:
            col = ac
    elif s == 1:
        pos = pb
        col = _BACKBONE
        if tel:
            col = bc
    elif s == 2:
        pos = wp.lerp(pa, pb, 0.36)
        col = ac
    else:
        pos = wp.lerp(pa, pb, 0.64)
        col = bc
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

    rpx = zoom * base_pt / cz * float(height_px)
    rad = int(wp.clamp(rpx, 1.0, 11.0))

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
    bg = wp.vec3(0.015, 0.019, 0.029) * (1.0 - 0.45 * yy)
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1022.0
    shade = 1.32 - 1.04 * depthq
    fog = wp.clamp((depthq - 0.34) * 1.6, 0.0, 0.85)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


# --- timeline: a long hold on base pairs, then each transition eases over TRANS with a hold after -------
_STAGE_DUR = 5.0                                         # seconds each process occupies
_TRANS = 3.4                                             # of which this many are the moving transition


def _progress(time: float):
    """Global progress in [0, STAGES-1]: integer part = stage, fractional = eased transition to the next."""
    t = max(float(time), 0.0)
    seg = int(t / _STAGE_DUR)
    if seg > _STAGES - 2:
        return float(_STAGES - 1)
    f = (t - seg * _STAGE_DUR) / _TRANS
    f = min(max(f, 0.0), 1.0)
    f = f * f * (3.0 - 2.0 * f)
    return seg + f


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


def _camera(time: float, g: float):
    seg = min(int(g), _STAGES - 2)
    f = g - seg
    cent = _lerp(_CENT[seg], _CENT[seg + 1], f)
    frame = float(_lerp(_STAGE_FRAME[seg], _STAGE_FRAME[seg + 1], f))     # framed radius (zoom in on fine stages)
    dist = 2.6 * frame + 5.0                              # fit the framed window, a little breathing room
    direction = _lerp(_STAGE_DIR[seg], _STAGE_DIR[seg + 1], f)
    direction = direction / np.linalg.norm(direction)
    target = cent.astype(np.float32)
    ro = target + dist * direction
    ww = target - ro
    ww = ww / np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0, 1.0, 0.0], np.float32))
    uu = uu / np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    return ro, uu, vv, ww, dist, frame


def _render(width, height, time, mouse, device):
    _ensure(device)
    W, H = int(width), int(height)
    g = _progress(float(time))
    seg = min(int(g), _STAGES - 2)
    f = float(g - seg)
    ro, uu, vv, ww, dist, rad = _camera(float(time), g)
    dnear = float(dist) - float(rad) * 1.25
    dfar = float(dist) + float(rad) * 1.25
    base_pt = 0.006 * rad + 0.010                         # splat scales with the framed size → opaque at every stage

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _morph_kernel,
        dim=_M,
        inputs=[_ka, _kb, _cola, _colb, zbuf, elemcol, W, H, _P, int(seg), f, float(base_pt),
                *cam, 1.7, dnear, dfar],
        device=device,
    )
    wp.launch(_resolve_kernel, dim=(H, W), inputs=[zbuf, elemcol, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=0.9, strength=0.4, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.0, preserve_hue=True)
    ldr = post.vignette(ldr, amount=0.3)
    return ldr


SCENE = Scene(
    name="warp_genome",
    description=(
        "The whole genome process in one long, continuous animation. The real output of every process — base "
        "pairs → double helices → nucleosomes → 30 nm fibre → telomeres → the condensed chromosome — morphs "
        "smoothly one into the next (the same 182 872 base pairs, conserved and only moved, each stage chained "
        "from the last), the camera re-framing each scale, settling on the finished coiled chromatid. No spin, "
        "nothing spawned — the DNA condensing the whole way down."
    ),
    renderer=_render,
)
