"""The HaPPY code — spacetime as a quantum error-correcting code.

The deepest entry in the dictionary. Almheiri, Dong and Harlow (2014) noticed that bulk
reconstruction has the *structure* of quantum error correction: a bulk operator deep in
AdS can be represented on many different boundary regions — no single boundary qubit
matters, exactly like a logical qubit in a code. Pastawski, Yoshida, Harlow and Preskill
(2015) then built the toy model: tile the Poincaré disk with pentagons `{5,4}`, put one
**[[5,1,3]] perfect-code tensor** on each pentagon — one logical (bulk) qubit in five
physical legs — and contract. The result is a holographic code whose entanglement wedges,
RT surfaces and reconstruction properties all emerge from the tiling.

The engine carries the actual code (``engine.holoinfo``, all test-asserted):
``five_qubit_stabilizers`` (cyclic XZZXI over GF(2)) and ``erasure_correctable`` — the
exact criterion (no logical operator supported on the erasure), brute-forced: **any two
erased legs are correctable, any three are fatal** (quantum MDS / no-cloning). The
geometric rule ``happy_central_recoverable`` — the central bulk qubit survives iff ≥3 of
its 5 legs reach intact boundary — is cross-checked in the suite against the algebraic
criterion on the same erased sets: *the wedge rule and the code rule are the same fact.*

The scene erases the boundary and watches the code fight:

* the `{5,4}` pentagon tiling glows gold — every tile one perfect-code tensor;
* an **erased arc** (dark crimson) sweeps across the boundary and back once per cycle;
  the bulk on the erased side desaturates as it falls out of the intact wedge;
* the **intact region's entanglement wedge** (violet, bounded by the geodesic capping
  the erasure) shrinks as the erasure grows;
* the **central logical qubit** — a white star on five spokes — stays lit while ≥3
  spokes reach intact boundary, and DIES the moment the third leg is erased: at the
  exact step the [[5,1,3]] brute force says recovery becomes impossible. Heal the
  boundary and it returns.

--frames runs one erase-and-heal cycle; iMouse rotates. See
``docs/research/49-holographic-quantum-information.md`` (Part III).
"""

import math

import warp as wp

from ..engine import post
from ..engine.adscft import geodesic_far_side, in_boundary_arc
from ..engine.holoinfo import happy_central_recoverable, happy_erased_legs
from ..scene import Scene

_DISK_R = wp.constant(0.43)
_T_CYCLE = 16.0
_LEG_OFF = 0.30                            # leg-0 angular offset

# ---- {5,4} tiling constants (same construction as the {7,3} fold, p=5, q=4) ----
_A5 = math.pi / 5.0
_X05 = math.tanh(0.5 * math.acosh(math.cos(math.pi / 4.0) / math.sin(math.pi / 5.0)))
WEDGE5 = wp.constant(_A5)
WEDGE5x2 = wp.constant(2.0 * _A5)
MIR_D5 = wp.constant((1.0 + _X05 * _X05) / (2.0 * _X05))
MIR_R25 = wp.constant(((1.0 + _X05 * _X05) / (2.0 * _X05) - _X05) ** 2)


@wp.func
def _fold54(z: wp.vec2) -> wp.vec4:
    """Fold z into the {5,4} fundamental domain (pentagons, four at a vertex — the
    HaPPY tiling). Returns vec4(z'.x, z'.y, generation, conformal scale)."""
    depth = float(0.0)
    scale = float(1.0)
    for _f in range(48):
        ang = wp.atan2(z[1], z[0])
        k = wp.floor((ang + WEDGE5) / WEDGE5x2)
        if k != 0.0:
            ca = wp.cos(-k * WEDGE5x2)
            sa = wp.sin(-k * WEDGE5x2)
            z = wp.vec2(ca * z[0] - sa * z[1], sa * z[0] + ca * z[1])
        if z[1] < 0.0:
            z = wp.vec2(z[0], -z[1])
        w = wp.vec2(z[0] - MIR_D5, z[1])
        r2 = wp.dot(w, w)
        if r2 < MIR_R25:
            kinv = MIR_R25 / r2
            z = wp.vec2(MIR_D5 + w[0] * kinv, w[1] * kinv)
            scale = scale * kinv
            depth += 1.0
        else:
            break
    return wp.vec4(z[0], z[1], depth, scale)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, e1: float, e2: float, has_erase: float,
                   legs_alive: wp.array(dtype=float), recoverable: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    uv = wp.vec2((fx - 0.5 * res[0]) / res[1], (fy - 0.5 * res[1]) / res[1])

    zd = uv / _DISK_R
    r = wp.length(zd)
    th = wp.atan2(zd[1], zd[0])
    if th < 0.0:
        th += 6.2831853
    px = 1.0 / (_DISK_R * res[1])

    col = wp.vec3(0.0, 0.0, 0.0)

    if r < 1.0:
        # ---- the code: {5,4} pentagons, one [[5,1,3]] tensor each ----
        folded = _fold54(zd)
        w = wp.vec2(folded[0] - MIR_D5, folded[1])
        e = wp.abs(wp.length(w) - wp.sqrt(MIR_R25))
        npix = e / wp.max(folded[3] * px, 1.0e-12)
        edge = wp.exp(-0.5 * (npix / 1.5) * (npix / 1.5))
        depth = wp.min(folded[2] / 8.0, 1.0)

        # the intact wedge: the far side of the geodesic capping the erased arc
        wedge = float(1.0)
        if has_erase > 0.5:
            wedge = geodesic_far_side(zd, e1, e2)
        alive = 0.18 + 0.82 * wedge

        col = wp.vec3(0.050, 0.034, 0.012) * (0.6 + 1.1 * depth) * alive
        col = col + wp.vec3(1.00, 0.74, 0.28) * edge * (0.45 + 0.75 * depth) * alive
        col = col + wp.vec3(0.36, 0.18, 0.55) * (wedge * has_erase * 0.38)

        # ---- the five legs of the central tensor ----
        for k in range(5):
            la = 2.0 * math.pi * float(k) / 5.0 + _LEG_OFF
            d = wp.vec2(wp.cos(la), wp.sin(la))
            t_par = zd[0] * d[0] + zd[1] * d[1]
            perp = wp.abs(zd[0] * d[1] - zd[1] * d[0])
            if t_par > 0.05:
                lw = wp.max(0.006, 2.0 * px)
                lg = wp.exp(-(perp * perp) / (lw * lw)) * wp.exp(-t_par * 1.2)
                lcol = wp.lerp(wp.vec3(0.45, 0.10, 0.10), wp.vec3(0.55, 0.95, 0.75),
                               legs_alive[k])
                col = col + lcol * (lg * (0.25 + 0.75 * legs_alive[k]))

        # ---- the central logical qubit: lit while the code can recover it ----
        d2c = wp.dot(zd, zd)
        wstar = wp.max(0.016, 5.0 * px)
        star = wp.exp(-d2c / (wstar * wstar))
        lit = wp.vec3(1.00, 0.98, 0.90) * (2.4 * recoverable)
        dead = wp.vec3(0.45, 0.10, 0.08) * (0.7 * (1.0 - recoverable))
        col = col + (lit + dead) * star

    # ---- the boundary: intact gold, erased crimson (halo clipped outside the disk,
    # so the two-colour split doesn't cast a sector into the void) ----
    bw = wp.max(0.004, 1.8 * px)
    out = wp.max(r - 1.0, 0.0)
    hmask = wp.exp(-out * out * 90.0)
    ring = (wp.exp(-((r - 1.0) * (r - 1.0)) / (bw * bw))
            + 0.25 * wp.exp(-wp.abs(r - 1.0) * 16.0)) * hmask
    erased = in_boundary_arc(th, e1, e2) * has_erase
    col = col + wp.vec3(0.55, 0.48, 0.36) * (ring * 0.7 * (1.0 - erased))
    col = col + wp.vec3(0.55, 0.06, 0.06) * (ring * 1.1 * erased)

    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)

    # the erased arc grows from 0 to ~5.5 rad and heals back, once per cycle
    arc = 5.5 * 0.5 * (1.0 - math.cos(2.0 * math.pi * tau / _T_CYCLE))
    e1, e2 = 0.0, arc                          # erase ccw from angle 0

    rec, _n = happy_central_recoverable(arc, _LEG_OFF)
    erased_legs = happy_erased_legs(arc, _LEG_OFF)
    alive = [0.0 if k in erased_legs else 1.0 for k in range(5)]

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(e1), float(e2), 1.0 if arc > 0.02 else 0.0,
                      wp.array(alive, dtype=float, device=device),
                      1.0 if rec else 0.0],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.45, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.12, preserve_hue=True)


SCENE = Scene(
    name="holo_code",
    description="spacetime as a quantum error-correcting code — the HaPPY pentagon "
                "tiling ({5,4}, one [[5,1,3]] perfect-code tensor per tile) fights a "
                "growing boundary erasure: the intact wedge shrinks, legs of the "
                "central tensor die crimson as the erased arc swallows them, and the "
                "central logical qubit — the bulk — goes dark at EXACTLY the third "
                "lost leg, where the GF(2) brute force proves recovery impossible; "
                "heal the boundary and the bulk returns. --frames runs one "
                "erase-and-heal cycle.",
    renderer=_render,
)
