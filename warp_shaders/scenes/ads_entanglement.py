"""Entanglement builds geometry — the Ryu-Takayanagi formula, drawn quantitatively.

``ads_cft`` decorates the disk with RT arcs; this scene *computes* them. Two boundary
intervals A (magenta) and B (green) live on the conformal boundary of a Poincaré-disk
slice of AdS₃, and every quantity on screen comes from the real dictionary
(``engine.adscft``, host-side, test-asserted):

* **Entropy = geodesic length.** ``interval_entropy`` is the regularized length of the
  bulk geodesic hanging from an interval's endpoints — Ryu-Takayanagi in global AdS₃,
  which lands exactly on the Calabrese-Cardy CFT answer ``S = (c/3) ln((2/ε) sin(Δθ/2))``.
  The two computations agree because the duality is true; that agreement is the scene.
* **The mutual-information phase transition.** S(A∪B) is the MINIMUM over the two allowed
  geodesic pairings: each interval capped by its own arc (*disconnected*, I(A:B) = 0) or
  the two gaps capped instead (*connected*, I(A:B) > 0). As the intervals drift together
  the minimal pairing SWAPS — first-order, by saddle competition, exactly like
  Hawking-Page. The winning pairing is drawn bright, the losing one ghost-faint: the
  subleading saddle is still there, it just doesn't dominate the entropy.
* **The entanglement wedge.** In the connected phase the region bounded by the two
  cross-arcs — the bulk that A∪B reconstructs — fills with light scaled by the actual
  I(A:B) value (``mutual_information``): when the intervals share no information the
  wedge is two disjoint slivers; when I jumps on, a connected chunk of spacetime
  *belongs* to them. Entanglement is literally holding that region together (ER=EPR in
  its infant, two-interval form).

The gap sweeps through the transition once per cycle, so the swap and the wedge ignition
repeat. See ``docs/research/46-ads-cft-holography.md`` (Part IV). --frames runs the cycle;
iMouse rotates.
"""

import math

import warp as wp

from ..engine import post
from ..engine.adscft import (
    interval_entropy,
    mutual_information,
    poincare_fold,
    rt_geodesic_glow,
    tile_edge,
)
from ..scene import Scene

_DISK_R = wp.constant(0.43)          # screen radius of the conformal boundary
_EPS_UV = 0.05                       # UV cutoff for the regularized entropies
_SIZE = 1.35                         # angular size of each interval (breathes slightly)


@wp.func
def _in_arc(th: float, t1: float, t2: float) -> float:
    """1 when boundary angle th lies in the ccw arc [t1, t2], else 0 (wrap-safe)."""
    span = t2 - t1 - 6.2831853 * wp.floor((t2 - t1) / 6.2831853)
    d = th - t1 - 6.2831853 * wp.floor((th - t1) / 6.2831853)
    out = float(0.0)
    if d < span:
        out = 1.0
    return out


@wp.func
def _outside_geo(zd: wp.vec2, th1: float, th2: float) -> float:
    """Soft mask: 1 on the far side of the geodesic capping the ccw gap [th1, th2].

    The orthogonal circle |z − c| = rad separates the disk into a gap side and a wedge
    side. Which sign is the gap side flips when the gap subtends more than π, so it is
    fixed by the gap arc's midpoint m (always on the gap side): wedge = the sign of
    (|z − c| − rad) OPPOSITE to (|m − c| − rad).
    """
    u = wp.vec2(wp.cos(th1), wp.sin(th1))
    v = wp.vec2(wp.cos(th2), wp.sin(th2))
    den = 1.0 + u[0] * v[0] + u[1] * v[1] + 1.0e-6
    c = wp.vec2((u[0] + v[0]) / den, (u[1] + v[1]) / den)
    rad = wp.sqrt(wp.max(wp.dot(c, c) - 1.0, 1.0e-8))
    span = th2 - th1 - 6.2831853 * wp.floor((th2 - th1) / 6.2831853)
    mth = th1 + 0.5 * span
    m = wp.vec2(wp.cos(mth), wp.sin(mth))
    flip = 1.0
    if wp.length(m - c) - rad > 0.0:
        flip = -1.0
    return wp.clamp(flip * (wp.length(zd - c) - rad) / 0.02, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, a1: float, a2: float, b1: float, b2: float,
                   conn: float, mi_glow: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    uv = wp.vec2((fx - 0.5 * res[0]) / res[1], (fy - 0.5 * res[1]) / res[1])

    zd = uv / _DISK_R
    r = wp.length(zd)
    th = wp.atan2(zd[1], zd[0])
    px = 1.0 / (_DISK_R * res[1])

    col = wp.vec3(0.0, 0.0, 0.0)

    # ---- faint hyperbolic bulk (the stage, not the star) ----
    if r < 1.0:
        rot = 0.04 * time + float(mouse[0]) * 0.004
        cp = wp.cos(rot)
        sp = wp.sin(rot)
        z = wp.vec2(cp * zd[0] - sp * zd[1], sp * zd[0] + cp * zd[1])
        folded = poincare_fold(z)
        edge = tile_edge(folded, px)
        depth = wp.min(folded[2] / 12.0, 1.0)
        col = wp.vec3(0.012, 0.020, 0.052) * (0.6 + 0.9 * depth)
        col = col + wp.vec3(0.16, 0.34, 0.55) * edge * (0.20 + 0.45 * depth)

        # ---- the entanglement wedge: bulk owned by A∪B in the connected phase ----
        wedge = _outside_geo(zd, a2, b1) * _outside_geo(zd, b2, a1)
        col = col + wp.vec3(0.42, 0.20, 0.62) * (wedge * conn * mi_glow * 0.55)

    # ---- the conformal boundary + the two intervals living on it ----
    bw = wp.max(0.004, 1.8 * px)
    ring = wp.exp(-((r - 1.0) * (r - 1.0)) / (bw * bw)) + 0.25 * wp.exp(-wp.abs(r - 1.0) * 16.0)
    col = col + wp.vec3(0.55, 0.48, 0.36) * ring * 0.7
    band = wp.exp(-((r - 1.0) * (r - 1.0)) / (9.0 * bw * bw))
    col = col + wp.vec3(1.00, 0.30, 0.78) * (band * _in_arc(th, a1, a2) * 1.6)
    col = col + wp.vec3(0.32, 1.00, 0.55) * (band * _in_arc(th, b1, b2) * 1.6)

    # ---- the two competing geodesic pairings: winner bright, loser ghost-faint ----
    # (geodesics live in the bulk: the orthogonal circles are masked to the disk)
    inside = wp.clamp((1.03 - r) / 0.03, 0.0, 1.0)
    g_disc = rt_geodesic_glow(zd, a1, a2, px) + rt_geodesic_glow(zd, b1, b2, px)
    g_conn = rt_geodesic_glow(zd, a2, b1, px) + rt_geodesic_glow(zd, b2, a1, px)
    amp_d = wp.lerp(1.15, 0.16, conn) * inside
    amp_c = wp.lerp(0.16, 1.15, conn) * inside
    col = col + wp.vec3(1.00, 0.62, 0.30) * (g_disc * amp_d)
    col = col + wp.vec3(0.95, 0.90, 1.00) * (g_conn * amp_c)

    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    t = float(time)
    # the two intervals, symmetric about the x-axis; the gap g1 sweeps through the
    # mutual-information transition (at g1 ~ 0.83 for these sizes) once per cycle (~13 s)
    size = _SIZE + 0.05 * math.sin(0.23 * t)
    g1 = 1.40 + 1.20 * math.sin(0.47 * t)
    a1 = 0.5 * g1
    a2 = a1 + size
    b2 = -0.5 * g1
    b1 = b2 - size

    mi, connected = mutual_information(g1, size, size, eps=_EPS_UV)
    # normalize the wedge glow against the maximal I in this sweep (g1 at its minimum)
    mi_max, _ = mutual_information(0.20, _SIZE + 0.05, _SIZE + 0.05, eps=_EPS_UV)
    mi_glow = min(mi / max(mi_max, 1e-9), 1.0)
    _ = interval_entropy(size, _EPS_UV)      # the per-interval S the pairing competes over

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, t, wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(a1), float(a2), float(b1), float(b2),
                      1.0 if connected else 0.0, float(mi_glow)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.45, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.12, preserve_hue=True)


SCENE = Scene(
    name="ads_entanglement",
    description="the Ryu-Takayanagi formula drawn quantitatively — two boundary intervals "
                "on a Poincare-disk slice of AdS3, entropy = regularized geodesic length "
                "(Calabrese-Cardy exactly), the mutual-information phase transition as a "
                "first-order swap between competing geodesic pairings (winner bright, loser "
                "ghost-faint), and the entanglement wedge filling with light scaled by the "
                "actual I(A:B) — entanglement literally holding a region of spacetime "
                "together. --frames sweeps through the transition.",
    renderer=_render,
)
