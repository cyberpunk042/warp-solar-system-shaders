"""The thermal universe — Gibbons-Hawking radiation and the entropy of everything.

Gibbons & Hawking 1977: the de Sitter horizon is a thermodynamic object *pointed at
you*. An inertial observer in empty Λ-space is bathed in thermal radiation at

    T = H/2π          (``gibbons_hawking_temperature``)

and the horizon carries entropy

    S = A/4 = π/H²    (``ds_entropy``, S = A/4 asserted)

— for our universe ~10¹²², the largest entropy any bounded system we can observe will
ever have (Bousso's bound saturated). The strangest clause: unlike a black hole, a
HOTTER de Sitter horizon is a SMALLER one carrying LESS entropy — feed Λ and you
*shrink* your world while warming it.

One cycle sweeps the cosmological constant up and back (H: 0.35 → 0.9 → 0.35), with
every element driven by the exact dictionary:

* the horizon ring sits at ``r_H = 1/H``, glowing with warmth ∝ ``T = H/2π``;
* **thermal quanta** stream inward from the horizon toward the observer — spawn rate
  and speed rise with T (the bath the far-future astronomer will actually measure,
  at 10⁻³⁰ K);
* the ring is subdivided into **entropy tiles** — one tick per quantum of horizon
  area, their count ∝ ``S = π/H²`` recomputed live: watch the ledger DROP as H rises
  (hotter → smaller → less room for information) and refill as Λ relaxes.

--frames runs one sweep of Λ; iMouse rotates. See
``docs/research/51-desitter-cosmic-horizon.md`` (Part III).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.desitter import ds_entropy, gibbons_hawking_temperature, horizon_radius
from ..scene import Scene

_DISK_R = wp.constant(0.44)
_T_CYCLE = 16.0
_N_Q = 42
_S_TILE = 0.4                                # horizon area per drawn entropy tile


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, r_h: float, warm: float, n_tiles: float,
                   quanta: wp.array(dtype=wp.vec4)):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    uv = wp.vec2((fx - 0.5 * res[0]) / res[1], (fy - 0.5 * res[1]) / res[1])

    zd = uv / _DISK_R
    r = wp.length(zd)
    th = wp.atan2(zd[1], zd[0])
    px = 1.0 / (_DISK_R * res[1])

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- interior: the observer's patch, faintly warm with the bath ----
    if r < r_h:
        col = col + wp.vec3(0.030, 0.020, 0.014) * ((0.25 + 0.75 * warm) * (r / r_h))

    # ---- the horizon ring at 1/H, glowing at T = H/2pi ----
    d_h = wp.abs(r - r_h)
    wh = wp.max(0.010, 2.2 * px)
    hot = wp.vec3(1.00, 0.50 + 0.35 * warm, 0.22 + 0.55 * warm)
    col = col + hot * (wp.exp(-(d_h * d_h) / (wh * wh)) * (0.7 + 0.8 * warm))

    # ---- the entropy ledger: S/S_tile ticks around the ring, count = pi/H^2 live ----
    if d_h < 0.045 and n_tiles > 0.0:
        segf = (th + 3.14159265) * n_tiles / 6.2831853
        seg = segf - wp.floor(segf)
        tick = wp.exp(-seg * seg / 0.006) + wp.exp(-(1.0 - seg) * (1.0 - seg) / 0.006)
        col = col + wp.vec3(0.35, 0.85, 1.00) * (tick * 0.65)

    # ---- the Gibbons-Hawking quanta: thermal dots streaming inward ----
    for q in range(_N_Q):
        qx = quanta[q][0]
        qy = quanta[q][1]
        qb = quanta[q][2]
        if qb > 0.0:
            wq = wp.max(0.009, 2.4 * px)
            d2 = (zd[0] - qx) * (zd[0] - qx) + (zd[1] - qy) * (zd[1] - qy)
            col = col + hot * (qb * 1.6 * wp.exp(-d2 / (wq * wq)))

    # ---- the observer ----
    wob = wp.max(0.012, 3.0 * px)
    col = col + wp.vec3(0.45, 0.75, 1.00) * (1.4 * wp.exp(-(r * r) / (wob * wob)))

    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)

    # Lambda sweeps up and back: H 0.35 -> 0.9 -> 0.35
    h = 0.35 + 0.55 * 0.5 * (1.0 - math.cos(2.0 * math.pi * tau / _T_CYCLE))
    scale = 0.34                                      # world -> disk units (coldest ring fits)
    r_h = horizon_radius(h) * scale
    t_gh = gibbons_hawking_temperature(h)
    warm = (h - 0.35) / 0.55
    n_tiles = ds_entropy(h) / _S_TILE                 # the ledger: fewer tiles when hotter

    # thermal quanta: spawned at the horizon, streaming inward; rate and speed ~ T
    quanta = []
    rate = 0.35 + 2.2 * warm
    speed = (0.16 + 0.30 * warm) * scale
    for q in range(_N_Q):
        birth = (q / rate + 3.0 * math.sin(q * 12.9898)) % 30.0
        age = (tau + 30.0 - birth) % (30.0 / rate) if rate > 0 else 0.0
        rq = r_h - age * speed
        if 0.03 < rq < r_h * 0.985:
            angq = 2.399963 * q + 0.3 * math.sin(q * 3.7)
            fade = min(1.0, (r_h - rq) / 0.08) * min(1.0, rq / 0.15)
            quanta.append((rq * math.cos(angq), rq * math.sin(angq), 0.9 * fade, 0.0))
    while len(quanta) < _N_Q:
        quanta.append((9.0, 9.0, 0.0, 0.0))
    _ = t_gh                                           # warmth IS this number

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(r_h), float(warm), float(n_tiles),
                      wp.array(np.asarray(quanta, np.float32), dtype=wp.vec4, device=device)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="ds_thermal",
    description="the Gibbons-Hawking bath — the de Sitter horizon as a thermodynamic "
                "object pointed at the observer: the ring at 1/H glows at T = H/2pi "
                "and streams thermal quanta inward, while cyan entropy tiles around it "
                "count S = pi/H^2 = A/4 live — and as Lambda sweeps up, the horizon "
                "SHRINKS, warms, and the ledger DROPS: a hotter universe holds less. "
                "--frames runs one sweep of the cosmological constant.",
    renderer=_render,
)
