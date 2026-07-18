"""The whole genome compression as ONE continuous animation — tokenized base pairs condensing, weave by
weave, all the way down to a single packed chromatid. One thread of the card's 182 872 base pairs, driven
end to end by the engine's genome libs — no shortcut, no morph between incompatible things, no matter
spawned or deleted. Every frame is a real, physically-valid intermediate state of the *same* thread.

It is literally the six library processes run back-to-back on one timeline and one camera, each chaining
from the previous one's actual output (verified: each stage's end state equals the next stage's start
state, the same 182 872 pairs throughout):

  1-2. base pairs        the tokenized rungs Process 2 bound (``bind_pairs`` field)
  3.   double helices    Process 3 winds the flat ladders into real 10.5-bp/turn helices — a true twist
                         (``warp_helix``'s ``_helix_end``, twist 0->1), not a lerp between two clouds
  4.   nucleosomes       Process 4 wraps each helix down onto its histone bead (``wrap_nucleosomes``)
  5.   30 nm fibre       Process 5 coils the beads-on-a-string into solenoid fibres (``coil_fibre``)
  6.   telomeres         Process 6 curls the strand's two ends into protective t-loop caps (``cap_telomeres``)
  7.   chromatid         Process 7 folds the capped fibre onto a coil scaffold (``fold_chromatid``) — the
                         fibre's real fine structure carried rigidly, the axis wound short and dense until
                         the arm is opaque, with a centromere waist and a telomere tip at each end.

Between chained stages the pairs move by the same continuous lerp the individual stage scenes use (the two
end states are geometrically adjacent by construction, so this is a real motion, not a teleport); the helix
stage twists for real; the fold winds the real matter tighter. The camera holds a fixed 3/4 look and only
dollies in — from the wide base-pair field to the tight chromatid — so the whole compression stays in frame.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..genome import wind_helix, wrap_nucleosomes, coil_fibre, cap_telomeres
from ..genome.chromatid import fold_chromatid
from ..scene import Scene

# --- the six real library states, each chaining from the previous (same 182 872 pairs throughout) ---
_HX = wind_helix(sub=2, block=5)                 # Process 3 — field -> helix (twisted in-kernel)
_NC = wrap_nucleosomes(sub=2, block=5)           # Process 4 — helix_a -> nuc_a
_FB = coil_fibre(sub=2, block=5)                 # Process 5 — bead_a(==nuc_a) -> fib_a
_TL = cap_telomeres(sub=2, block=5)              # Process 6 — fib_a -> tel_a
_CH = fold_chromatid(sub=2, block=5)             # Process 7 — tel_a -> chr_a

_P = int(_HX.n_pairs)
_G = int(_HX.bp_per_helix)
_SAMPLES = 4                                     # per pair: backbone A, backbone B, 2 rung points
_M = _P * _SAMPLES

# telomere-tinted base colours, carried through every stage
_A_COL = _CH.a_col.astype(np.float32).copy()
_B_COL = _CH.b_col.astype(np.float32).copy()

_dev = {}


def _ensure(device):
    if device in _dev:
        return _dev[device]
    a = {
        "field_a": wp.array(_HX.field_a, dtype=wp.vec3, device=device),
        "field_b": wp.array(_HX.field_b, dtype=wp.vec3, device=device),
        "centers": wp.array(_HX.centers, dtype=wp.vec3, device=device),
        "helix_a": wp.array(_NC.helix_a, dtype=wp.vec3, device=device),
        "helix_b": wp.array(_NC.helix_b, dtype=wp.vec3, device=device),
        "nuc_a": wp.array(_NC.nuc_a, dtype=wp.vec3, device=device),
        "nuc_b": wp.array(_NC.nuc_b, dtype=wp.vec3, device=device),
        "fib_a": wp.array(_FB.fib_a, dtype=wp.vec3, device=device),
        "fib_b": wp.array(_FB.fib_b, dtype=wp.vec3, device=device),
        "tel_a": wp.array(_TL.tel_a, dtype=wp.vec3, device=device),
        "tel_b": wp.array(_TL.tel_b, dtype=wp.vec3, device=device),
        "chr_a": wp.array(_CH.chr_a, dtype=wp.vec3, device=device),
        "chr_b": wp.array(_CH.chr_b, dtype=wp.vec3, device=device),
        "a_col": wp.array(_A_COL, dtype=wp.vec3, device=device),
        "b_col": wp.array(_B_COL, dtype=wp.vec3, device=device),
    }
    _dev[device] = a
    return a


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)
_BACKBONE = wp.constant(wp.vec3(0.46, 0.53, 0.66))


@wp.func
def _helix_end(c: wp.vec3, l: int, g: int, off: float, twist: float,
               radius: float, height: float, dtheta: float) -> wp.vec3:
    y = (float(l) / float(g) - 0.5) * height
    theta = twist * float(l) * dtheta + off              # twist=0 -> flat ladder; twist=1 -> full helix
    return wp.vec3(c[0] + radius * wp.cos(theta), c[1] + y, c[2] + radius * wp.sin(theta))


@wp.kernel
def _master_kernel(
    field_a: wp.array(dtype=wp.vec3),
    field_b: wp.array(dtype=wp.vec3),
    centers: wp.array(dtype=wp.vec3),
    helix_a: wp.array(dtype=wp.vec3),
    helix_b: wp.array(dtype=wp.vec3),
    nuc_a: wp.array(dtype=wp.vec3),
    nuc_b: wp.array(dtype=wp.vec3),
    fib_a: wp.array(dtype=wp.vec3),
    fib_b: wp.array(dtype=wp.vec3),
    tel_a: wp.array(dtype=wp.vec3),
    tel_b: wp.array(dtype=wp.vec3),
    chr_a: wp.array(dtype=wp.vec3),
    chr_b: wp.array(dtype=wp.vec3),
    a_col: wp.array(dtype=wp.vec3),
    b_col: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    seg: int,
    phase: float,           # lerp phase within a chained segment (seg 1..4)
    to_ladder: float,       # seg 0: base-pair field -> straight ladder
    twist: float,           # seg 0: flat ladder -> full double helix
    g: int,
    radius: float,
    hheight: float,
    dtheta: float,
    groove: float,
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    zoom: float,
    dnear: float,
    dfar: float,
    base_pt: float,
    max_rad: float,
):
    e = wp.tid()
    pr = e / 4
    s = e - pr * 4

    if seg == 0:
        hg = pr / g
        l = pr - hg * g
        c = centers[hg]
        ha = _helix_end(c, l, g, 0.0, twist, radius, hheight, dtheta)
        hb = _helix_end(c, l, g, groove, twist, radius, hheight, dtheta)
        pa = wp.lerp(field_a[pr], ha, to_ladder)
        pb = wp.lerp(field_b[pr], hb, to_ladder)
    elif seg == 1:
        pa = wp.lerp(helix_a[pr], nuc_a[pr], phase)       # Process 4 wrap
        pb = wp.lerp(helix_b[pr], nuc_b[pr], phase)
    elif seg == 2:
        pa = wp.lerp(nuc_a[pr], fib_a[pr], phase)         # Process 5 coil
        pb = wp.lerp(nuc_b[pr], fib_b[pr], phase)
    elif seg == 3:
        pa = wp.lerp(fib_a[pr], tel_a[pr], phase)         # Process 6 cap
        pb = wp.lerp(fib_b[pr], tel_b[pr], phase)
    else:
        pa = wp.lerp(tel_a[pr], chr_a[pr], phase)         # Process 7 fold
        pb = wp.lerp(tel_b[pr], chr_b[pr], phase)

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

    # telomere-green pairs tint their backbone too, so the caps read through the whole ladder
    ac = a_col[pr]
    if ac[1] > 0.9 and (s == 0 or s == 1):
        col = ac
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

    base = base_pt
    if s >= 2:
        base = base_pt * 0.55                              # rungs a touch smaller than backbones
    rpx = zoom * base / cz * float(height_px)
    rad = int(wp.clamp(rpx, 1.0, max_rad))

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
def _resolve_kernel(zbuf: wp.array2d(dtype=wp.int32), elemcol: wp.array(dtype=wp.vec3),
                    img: wp.array2d(dtype=wp.vec3), width: int, height_px: int):
    i, j = wp.tid()
    yy = float(i) / float(height_px)
    bg = wp.vec3(0.016, 0.020, 0.030) * (1.0 - 0.45 * yy)
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1022.0
    shade = 1.30 - 1.06 * depthq
    fog = wp.clamp((depthq - 0.26) * 1.8, 0.0, 0.9)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


# ---- one continuous timeline (seconds): the six processes run back-to-back ----
_SEG1, _SEG2, _SEG3, _SEG4 = 6.7, 10.35, 13.85, 17.35     # segment change-over times
_END = 24.2


def _ss(t, a, b):
    u = min(max((t - a) / (b - a), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _keys(t, ks):
    for i in range(len(ks) - 1):
        (ta, va), (tb, vb) = ks[i], ks[i + 1]
        if t <= tb:
            return va + (vb - va) * _ss(t, ta, tb)
    return ks[-1][1]


def _stage(t):
    """(seg, phase, to_ladder, twist) — which real transition is running at time t."""
    if t < _SEG1:
        return 0, 0.0, _ss(t, 0.6, 3.0), _ss(t, 3.2, 6.4)
    if t < _SEG2:
        return 1, _ss(t, 7.0, 10.2), 1.0, 1.0
    if t < _SEG3:
        return 2, _ss(t, 10.5, 13.7), 1.0, 1.0
    if t < _SEG4:
        return 3, _ss(t, 14.0, 17.2), 1.0, 1.0
    return 4, _ss(t, 17.5, 23.3), 1.0, 1.0


def _camera(t):
    # a single 3/4 look that only dollies in as the thread condenses (radius 41 -> ~10), tilting a little
    # more side-on onto the finished chromatid rod. One continuous path, no cut, no spin.
    rframe = _keys(t, [(0.0, 38.0), (6.6, 42.0), (10.2, 42.0), (13.6, 34.0),
                       (17.2, 33.0), (20.2, 20.0), (23.3, 11.0), (_END, 11.0)])
    dx = _keys(t, [(0.0, 0.10), (13.0, 0.12), (20.0, 0.20), (_END, 0.24)])
    dyv = _keys(t, [(0.0, 0.46), (6.0, 0.34), (13.0, 0.30), (20.0, 0.14), (_END, 0.07)])
    dist = 1.34 * rframe + 7.0
    target = np.array([0.0, 0.0, -0.4], np.float32)
    direction = np.array([dx, dyv, 1.0], np.float32)
    direction = direction / np.linalg.norm(direction)
    ro = target + dist * direction
    ww = target - ro
    ww = ww / np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0, 1.0, 0.0], np.float32))
    uu = uu / np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    return ro, uu, vv, ww, dist, rframe


def _render(width, height, time, mouse, device):
    a = _ensure(device)
    W, H = int(width), int(height)
    t = float(time)
    seg, phase, to_ladder, twist = _stage(t)
    ro, uu, vv, ww, dist, rframe = _camera(t)
    dnear = max(1.5, float(dist) - float(rframe) * 1.15)
    dfar = float(dist) + float(rframe) * 1.15

    # splats fatten as the thread packs, so the finished chromatid reads dense and opaque
    base_pt = _keys(t, [(0.0, 0.017), (10.2, 0.020), (14.0, 0.028),
                        (17.5, 0.040), (23.3, 0.060), (_END, 0.060)])
    max_rad = _keys(t, [(0.0, 6.0), (17.0, 6.0), (20.0, 9.0), (_END, 12.0)])

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _master_kernel,
        dim=_M,
        inputs=[a["field_a"], a["field_b"], a["centers"], a["helix_a"], a["helix_b"],
                a["nuc_a"], a["nuc_b"], a["fib_a"], a["fib_b"], a["tel_a"], a["tel_b"],
                a["chr_a"], a["chr_b"], a["a_col"], a["b_col"], zbuf, elemcol, W, H,
                int(seg), float(phase), float(to_ladder), float(twist),
                _G, _HX.radius, _HX.height, _HX.dtheta, _HX.groove,
                *cam, 1.7, dnear, dfar, float(base_pt), float(max_rad)],
        device=device,
    )
    wp.launch(_resolve_kernel, dim=(H, W), inputs=[zbuf, elemcol, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=0.9, strength=0.35, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.03, preserve_hue=True)
    ldr = post.vignette(ldr, amount=0.3)
    return ldr


SCENE = Scene(
    name="warp_genome",
    description=(
        "The whole genome compression as one continuous animation — the card's 182 872 base pairs condensing "
        "weave by weave down to a single packed chromatid. It is the engine's six genome library processes "
        "run back-to-back on one timeline and one camera, each chaining from the previous one's actual output: "
        "the tokenized base pairs twist into real 10.5-bp/turn double helices (a true winding, not a morph), "
        "wrap onto nucleosome beads, coil into 30 nm fibres, curl their two ends into telomere t-loop caps, "
        "and fold onto a coil scaffold into a dense, opaque chromatid with a centromere waist and a telomere "
        "tip at each end. The same thread throughout — matter conserved, never copied, never spawned, every "
        "frame a real partially-condensed state; the camera only dollies in as it packs."
    ),
    renderer=_render,
)


# expose the conserved thread for tests
_N = _P
