"""MERA — the hyperbolic tiling IS a renormalization network.

Swingle's observation (2009): draw the MERA tensor network that renormalizes a critical
spin chain — one layer of disentanglers and coarse-grainers per RG step — and you have
drawn a discretized time-slice of AdS. Radial direction = RG scale; each tiling
generation inward = one coarse-graining layer; the UV lattice lives on the boundary and
the deep IR sits at the centre. The `{7,3}` tiling the whole holography set is built on
was a MERA all along.

The dictionary entry is a counting fact (``engine.holoinfo``, test-asserted): an
interval of ℓ sites is coarse-grained away in ``mera_layers(ℓ) = ceil(log₂ ℓ)`` steps,
and the minimal cut through the network that isolates it severs
``mera_cut_bonds(ℓ) = 2·ceil(log₂ ℓ) + O(1)`` bonds. With bond dimension χ each severed
bond carries at most ln χ of entanglement, so ``S(ℓ) ≲ (2·log₂ ℓ)·ln χ`` — **the CFT
log law, produced by a network you can count on your fingers**. RT's geodesic is the
continuum limit of that minimal cut.

The scene stages the count:

* the tiling glows in **banded layers** — each generation one RG step, UV gold at the
  boundary shading to IR blue at the centre;
* a boundary interval sweeps in size; its **causal cone** (the wedge of tensors that
  feel it) tints violet down to the depth ``mera_layers(ℓ)`` where it closes;
* the **minimal cut** — the RT geodesic — is drawn as a chain of **beads at equal
  hyperbolic spacing**: each bead one severed bond. The bead count grows one PAIR at a
  time as the interval doubles: the log law, literally countable on screen.

Bead positions are computed host-side by integrating hyperbolic arclength along the
geodesic (with the same UV cutoff that regularizes the entropy — the bond count is
cutoff-dependent exactly as S is). --frames runs one sweep; iMouse rotates. See
``docs/research/49-holographic-quantum-information.md`` (Part II).
"""

import math

import warp as wp

from ..engine import post
from ..engine.adscft import (
    geodesic_far_side,
    in_boundary_arc,
    poincare_fold,
    rt_geodesic_glow,
    tile_edge,
)
from ..engine.holoinfo import mera_cut_bonds, mera_layers
from ..scene import Scene

_DISK_R = wp.constant(0.43)
_T_CYCLE = 14.0
_R_CUT = 0.985                            # UV cutoff radius for arclength integration
_DS_BOND = 0.85                           # hyperbolic spacing between severed bonds
_MAX_BEADS = 24


def _geodesic_beads(th1, th2):
    """Bead positions at equal hyperbolic arclength along the geodesic between boundary
    angles th1, th2 (host-side; clipped at the UV cutoff radius)."""
    u = (math.cos(th1), math.sin(th1))
    v = (math.cos(th2), math.sin(th2))
    den = 1.0 + u[0] * v[0] + u[1] * v[1] + 1e-9
    cx, cy = (u[0] + v[0]) / den, (u[1] + v[1]) / den
    rad = math.sqrt(max(cx * cx + cy * cy - 1.0, 1e-9))
    a1 = math.atan2(u[1] - cy, u[0] - cx)
    a2 = math.atan2(v[1] - cy, v[0] - cx)
    dphi = (a2 - a1 + math.pi) % (2.0 * math.pi) - math.pi     # short way around c
    n_s = 600
    beads, s_acc, s_next = [], 0.0, 0.5 * _DS_BOND
    for k in range(n_s):
        ph = a1 + dphi * (k + 0.5) / n_s
        x, y = cx + rad * math.cos(ph), cy + rad * math.sin(ph)
        r2 = x * x + y * y
        if r2 > _R_CUT * _R_CUT:
            continue
        ds = 2.0 * rad * abs(dphi) / n_s / max(1.0 - r2, 1e-6)
        s_acc += ds
        if s_acc >= s_next and len(beads) < _MAX_BEADS:
            beads.append((x, y))
            s_next += _DS_BOND
    return beads


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, a1: float, a2: float, cone_depth: float,
                   bead_x: wp.array(dtype=float), bead_y: wp.array(dtype=float),
                   n_beads: int):
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

    if r < 1.0:
        # ---- the network: tiling generations as RG layers, UV gold -> IR blue ----
        rot = float(mouse[0]) * 0.004
        cp = wp.cos(rot)
        sp = wp.sin(rot)
        z = wp.vec2(cp * zd[0] - sp * zd[1], sp * zd[0] + cp * zd[1])
        folded = poincare_fold(z)
        edge = tile_edge(folded, px)
        gen = folded[2]
        band = 0.5 + 0.5 * wp.sin(gen * 1.1)
        f_uv = wp.clamp(gen / 9.0, 0.0, 1.0)          # 0 deep IR .. 1 UV rim
        layer_col = wp.lerp(wp.vec3(0.10, 0.22, 0.55), wp.vec3(0.95, 0.70, 0.25), f_uv)
        col = layer_col * (0.045 + 0.035 * band)
        col = col + layer_col * edge * (0.30 + 0.35 * f_uv)

        # ---- the causal cone: the tensors that feel the interval, down to its depth ----
        wedge = 1.0 - geodesic_far_side(zd, a1, a2)
        feel = wedge * wp.clamp((gen - (9.0 - cone_depth)) / 2.0 + 1.0, 0.0, 1.0)
        col = col + wp.vec3(0.42, 0.20, 0.60) * (feel * 0.40)

    # ---- the minimal cut: the RT geodesic, one bead per severed bond ----
    inside = wp.clamp((1.03 - r) / 0.03, 0.0, 1.0)
    g_rt = rt_geodesic_glow(zd, a1, a2, px)
    col = col + wp.vec3(0.85, 0.78, 1.00) * (g_rt * 0.55 * inside)
    for k in range(n_beads):
        d2 = (zd[0] - bead_x[k]) * (zd[0] - bead_x[k]) + (zd[1] - bead_y[k]) * (zd[1] - bead_y[k])
        w = wp.max(0.010, 3.5 * px)
        col = col + wp.vec3(1.00, 0.92, 0.55) * (1.6 * wp.exp(-d2 / (w * w)) * inside)

    # ---- the UV lattice: the boundary + the interval ----
    bw = wp.max(0.004, 1.8 * px)
    ring = wp.exp(-((r - 1.0) * (r - 1.0)) / (bw * bw)) + 0.25 * wp.exp(-wp.abs(r - 1.0) * 16.0)
    col = col + wp.vec3(0.55, 0.48, 0.36) * ring * 0.7
    band2 = wp.exp(-((r - 1.0) * (r - 1.0)) / (9.0 * bw * bw))
    col = col + wp.vec3(1.00, 0.30, 0.78) * (band2 * in_boundary_arc(th, a1, a2) * 1.6)

    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)

    theta = 0.35 + 2.45 * 0.5 * (1.0 - math.cos(2.0 * math.pi * tau / _T_CYCLE))
    a1 = 0.5 * math.pi - 0.5 * theta
    a2 = 0.5 * math.pi + 0.5 * theta

    # the dictionary: sites ~ theta / lattice spacing; depth + bond count are the counts
    l_sites = max(2, round(theta / 0.06))
    depth = min(mera_layers(l_sites), 9)
    _ = mera_cut_bonds(l_sites)              # the count the beads realize geometrically

    beads = _geodesic_beads(a1, a2)
    bx = wp.array([b[0] for b in beads] or [9.0], dtype=float, device=device)
    by = wp.array([b[1] for b in beads] or [9.0], dtype=float, device=device)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(a1), float(a2), float(depth), bx, by, int(len(beads))],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.45, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.12, preserve_hue=True)


SCENE = Scene(
    name="holo_mera",
    description="the hyperbolic tiling IS a tensor network — Swingle's MERA/AdS map: "
                "tiling generations as RG layers (UV gold rim to IR blue core), a "
                "boundary interval's causal cone tinting the tensors that feel it, and "
                "the minimal cut drawn as beads at equal hyperbolic spacing along the "
                "RT geodesic — one bead per severed bond, their count growing "
                "logarithmically as the interval sweeps: the CFT log law S ~ (c/3) ln l, "
                "counted on screen. --frames runs one sweep.",
    renderer=_render,
)
