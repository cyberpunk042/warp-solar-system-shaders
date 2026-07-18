"""AdS/CFT — holographic duality on the Poincaré disk, tiling, boundary and RT geodesics.

**Anti-de Sitter / Conformal Field Theory correspondence** (Maldacena 1997): a theory of
gravity in a (d+1)-dimensional negatively curved *bulk* is exactly equivalent — dual — to a
quantum field theory with no gravity living on its d-dimensional *conformal boundary*. This
scene draws a constant-time slice of AdS₃, which is the **Poincaré disk** — the conformal map
of an infinite hyperbolic plane into a finite circle:

* **The bulk** — a `{7,3}` hyperbolic tiling built by reflection-group folding (rotate into a
  wedge of angle π/7, reflect, invert in the edge-mirror circle, repeat). Every heptagon is the
  *same hyperbolic size*; the crowding toward the rim is pure metric distortion, exactly
  Escher's *Circle Limit* prints. The Euclidean distance from the origin to the edge-mirror
  follows from hyperbolic trigonometry: ``cosh m = cos(π/q) / sin(π/p)``.
* **The conformal boundary** at r = 1 — infinitely far away in hyperbolic distance yet a finite
  glowing ring on screen. This is where the CFT lives.
* **The hologram** — outside the disk the *same* tiling reappears through the inversion
  ``z → z/|z|²``, which maps the exterior conformally onto the interior: the boundary theory
  encoding the bulk, drawn literally.
* **Ryu–Takayanagi geodesics** — circular arcs orthogonal to the boundary, anchored on
  boundary intervals. RT (2006): the entanglement entropy of a boundary interval equals the
  length of the bulk geodesic hanging from its endpoints, S = Length/4G. Endpoint dots are the
  interval endpoints ("operator insertions").
* **The dictionary in motion** — the whole tiling flows under a time-dependent Möbius isometry
  ``z → (z − a)/(1 − ā z)``: an isometry of the bulk is a conformal transformation of the
  boundary, the entry-one of the AdS/CFT dictionary.

See ``docs/research/46-ads-cft-holography.md``. --frames flows the isometry; iMouse rotates.
"""

import math

import warp as wp

from ..engine import post
from ..scene import Scene

# ---- {p,q} = {7,3} tiling constants (derived, not tuned) ----------------------------------
# Right hyperbolic triangle O-M-V (polygon centre, edge midpoint, vertex) with angles
# pi/p at O and pi/q at V:  cosh(OM) = cos(pi/q) / sin(pi/p).
_P = 7.0
_Q = 3.0
_A = math.pi / _P                              # half-wedge angle at the origin
_COSH_M = math.cos(math.pi / _Q) / math.sin(math.pi / _P)
_M = math.acosh(_COSH_M)
_X0 = math.tanh(0.5 * _M)                      # Euclidean distance to the edge midpoint
_DC = (1.0 + _X0 * _X0) / (2.0 * _X0)          # mirror-circle centre (orthogonal: dc^2 = 1 + rr^2)
_RC = _DC - _X0                                # mirror-circle radius

_WEDGE = wp.constant(_A)
_WEDGE2 = wp.constant(2.0 * _A)
_MIR_D = wp.constant(_DC)
_MIR_R2 = wp.constant(_RC * _RC)
_FOLDS = 48                                    # reflection-group folding depth
_DISK_R = wp.constant(0.43)                    # screen radius of the conformal boundary (uv half-height is 0.5)
_N_GEO = 3                                     # Ryu-Takayanagi geodesics


@wp.func
def _mobius(z: wp.vec2, a: wp.vec2) -> wp.vec2:
    """Disk isometry z -> (z - a) / (1 - conj(a) z), complex arithmetic on vec2."""
    n = wp.vec2(z[0] - a[0], z[1] - a[1])
    d = wp.vec2(1.0 - (a[0] * z[0] + a[1] * z[1]), a[1] * z[0] - a[0] * z[1])
    s = d[0] * d[0] + d[1] * d[1] + 1.0e-12
    return wp.vec2((n[0] * d[0] + n[1] * d[1]) / s, (n[1] * d[0] - n[0] * d[1]) / s)


@wp.func
def _mobius_jac(z: wp.vec2, a: wp.vec2) -> float:
    """|f'(z)| = (1 - |a|^2) / |1 - conj(a) z|^2 — local magnification of the isometry."""
    d = wp.vec2(1.0 - (a[0] * z[0] + a[1] * z[1]), a[1] * z[0] - a[0] * z[1])
    return (1.0 - (a[0] * a[0] + a[1] * a[1])) / (d[0] * d[0] + d[1] * d[1] + 1.0e-12)


@wp.func
def _rt_geodesic(zd: wp.vec2, th1: float, th2: float, px: float) -> float:
    """Glow of the bulk geodesic anchored at boundary angles th1, th2.

    The unique circle through both endpoints orthogonal to the unit circle has centre
    c = (u + v)/(1 + u.v) and radius^2 = |c|^2 - 1; its arc inside the disk IS the
    hyperbolic geodesic (the RT minimal surface for the boundary interval).
    """
    u = wp.vec2(wp.cos(th1), wp.sin(th1))
    v = wp.vec2(wp.cos(th2), wp.sin(th2))
    den = 1.0 + u[0] * v[0] + u[1] * v[1]
    c = wp.vec2((u[0] + v[0]) / den, (u[1] + v[1]) / den)
    rad = wp.sqrt(wp.max(wp.dot(c, c) - 1.0, 1.0e-8))
    darc = wp.abs(wp.length(zd - c) - rad)
    w = wp.max(0.0035, 1.5 * px)
    glow = wp.exp(-(darc * darc) / (w * w)) + 0.35 * wp.exp(-darc * 26.0)
    # endpoint dots — the boundary interval's operator insertions
    de = wp.min(wp.length(zd - u), wp.length(zd - v))
    glow += 2.2 * wp.exp(-(de * de) / (9.0 * w * w))
    return glow


@wp.kernel
def _render_kernel(
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int,
    time: float,
    mouse: wp.vec2,
):
    i, j = wp.tid()

    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    uv = wp.vec2((fx - 0.5 * res[0]) / res[1], (fy - 0.5 * res[1]) / res[1])

    # display coordinates on the Poincare disk (boundary at |zd| = 1)
    zd = uv / _DISK_R
    r = wp.length(zd)
    px = 1.0 / (_DISK_R * res[1])              # one pixel, in disk units

    # ---- bulk sample point: interior directly, exterior through the inversion hologram ----
    ext = float(0.0)
    z = zd
    foot = px                                   # pixel footprint carried through every map
    ext_fade = float(1.0)
    if r >= 1.0:
        ext = 1.0
        z = zd / (r * r)                        # z -> z / |z|^2 maps exterior onto interior
        foot = px / (r * r)
        ext_fade = wp.exp(-(r - 1.0) * 2.4)

    # ---- the AdS isometry flow (= boundary conformal transformation) ----
    ph = 0.06 * time + float(mouse[0]) * 0.004
    cp = wp.cos(ph)
    sp = wp.sin(ph)
    z = wp.vec2(cp * z[0] - sp * z[1], sp * z[0] + cp * z[1])
    a = wp.vec2(0.26 * wp.cos(0.23 * time + 1.2), 0.26 * wp.sin(0.31 * time))
    foot = foot * _mobius_jac(z, a)
    z = _mobius(z, a)

    # ---- {7,3} reflection-group fold ----
    depth = int(0)
    for _f in range(_FOLDS):
        ang = wp.atan2(z[1], z[0])
        k = wp.floor((ang + _WEDGE) / _WEDGE2)
        if k != 0.0:
            ca = wp.cos(-k * _WEDGE2)
            sa = wp.sin(-k * _WEDGE2)
            z = wp.vec2(ca * z[0] - sa * z[1], sa * z[0] + ca * z[1])
        if z[1] < 0.0:
            z = wp.vec2(z[0], -z[1])
        w = wp.vec2(z[0] - _MIR_D, z[1])
        r2 = wp.dot(w, w)
        if r2 < _MIR_R2:
            kinv = _MIR_R2 / r2
            z = wp.vec2(_MIR_D + w[0] * kinv, w[1] * kinv)
            foot = foot * kinv
            depth += 1
        else:
            break

    # distance to the mirror geodesic (the tile edge), converted back to screen pixels
    w = wp.vec2(z[0] - _MIR_D, z[1])
    e = wp.abs(wp.length(w) - wp.sqrt(_MIR_R2))
    npix = e / wp.max(foot, 1.0e-12)
    edge = wp.exp(-0.5 * (npix / 1.5) * (npix / 1.5))

    parity = float(depth % 2)                   # chequer parity of the reflection count
    t_depth = wp.min(float(depth) / 14.0, 1.0)
    crowd = t_depth * t_depth                   # cells crowd (UV-diverge) toward the boundary

    # ---- palettes: cool geometric bulk, warm field-theory hologram ----
    cell_a = wp.vec3(0.030, 0.052, 0.150)
    cell_b = wp.vec3(0.012, 0.110, 0.165)
    edge_c = wp.vec3(0.30, 0.85, 1.00)
    if ext > 0.5:
        cell_a = wp.vec3(0.130, 0.045, 0.014)
        cell_b = wp.vec3(0.165, 0.085, 0.020)
        edge_c = wp.vec3(1.00, 0.55, 0.18)

    col = wp.lerp(cell_a, cell_b, parity)
    col = col * (0.75 + 0.6 * t_depth)
    col = col + edge_c * edge * (0.55 + 1.1 * t_depth)
    col = wp.lerp(col, edge_c * 0.85, 0.55 * crowd)   # unresolvable rim shimmer
    if ext > 0.5:
        col = col * ext_fade

    # ---- the conformal boundary: infinitely far, finitely drawn ----
    bw = wp.max(0.004, 1.8 * px)
    ring = wp.exp(-((r - 1.0) * (r - 1.0)) / (bw * bw)) + 0.30 * wp.exp(-wp.abs(r - 1.0) * 16.0)
    col = col + wp.vec3(1.00, 0.86, 0.58) * ring * 1.25

    # ---- Ryu-Takayanagi geodesics, anchored on drifting boundary intervals ----
    for g in range(_N_GEO):
        gf = float(g)
        thc = gf * 2.0943951 + 0.21 * time * (1.0 - 0.24 * gf)
        half = 0.62 + 0.30 * wp.sin(0.37 * time + gf * 2.1)
        glow = _rt_geodesic(zd, thc - half, thc + half, px)
        gc = wp.vec3(1.0, 0.42, 0.85)
        if g == 1:
            gc = wp.vec3(0.45, 1.0, 0.70)
        if g == 2:
            gc = wp.vec3(1.0, 0.80, 0.35)
        amp = 1.15
        if ext > 0.5:
            amp = 0.20 * ext_fade               # faint mirrored arc in the hologram
        col = col + gc * glow * amp

    # gentle vignette so the disk floats in darkness
    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))

    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(
        _render_kernel,
        dim=(height, width),
        inputs=[img, width, height, float(time), wp.vec2(float(mouse[0]), float(mouse[1]))],
        device=device,
    )
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.45, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.12, preserve_hue=True)


SCENE = Scene(
    name="ads_cft",
    description="the AdS/CFT correspondence on a Poincare-disk slice of AdS3 — a {7,3} "
                "hyperbolic tiling bulk crowding toward the glowing conformal boundary, the "
                "same tiling holographically mirrored outside through z -> z/|z|^2, "
                "Ryu-Takayanagi entanglement geodesics hanging from boundary intervals, all "
                "flowing under a Mobius isometry (= a boundary conformal transformation).",
    renderer=_render,
)
