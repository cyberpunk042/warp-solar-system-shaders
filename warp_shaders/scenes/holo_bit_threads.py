"""Bit threads — entanglement entropy as a maximum flow.

Ryu-Takayanagi says S(A) is the area of a minimal surface: a *cut*. Freedman and
Headrick (2016) recast it by strong duality: S(A) is equally the **maximum flux** of a
divergence-free flow of "bit threads" — unbreakable, unoriented threads of cross-section
1/4G leaving A — and **max flow = min cut** is exactly the LP-duality theorem of network
theory. The threads are the entanglement: each thread leaving A is one Bell pair's worth
of correlation between A and its complement, and the RT geodesic is simply the
bottleneck where the thread bundle saturates.

The engine backs the picture with an actual computation (``engine.holoinfo``): a polar
grid on the Poincaré disk whose edge capacities are the hyperbolic lengths of the dual
segments they cross, plus Dinic's algorithm. The suite asserts flow == cut to 1e-12
(MFMC, exactly) and that the flow tracks the analytic geodesic length
``2·ln(sin(Δθ/2)) + const`` across interval sizes — the discrete threads really do
compute the RT entropy.

The scene draws the dual picture live:

* a magenta boundary interval A, sweeping in size once per cycle;
* the **thread bundle**: nested geodesic arcs pairing points of A with points of the
  complement — non-crossing, as an optimal flow must be — with the number of drawn
  threads proportional to the analytic entropy S(A): watch threads switch on as the
  interval grows and its capacity to carry correlation rises;
* the **bottleneck**: the RT geodesic glowing white-violet where the bundle squeezes
  through at maximum density — the min cut that the max flow saturates.

--frames runs one sweep; iMouse rotates. See
``docs/research/49-holographic-quantum-information.md`` (Part I).
"""

import math

import warp as wp

from ..engine import post
from ..engine.adscft import (
    in_boundary_arc,
    interval_entropy,
    poincare_fold,
    rt_geodesic_glow,
    tile_edge,
)
from ..scene import Scene

_DISK_R = wp.constant(0.43)
_T_CYCLE = 14.0
_N_MAX = 26                              # thread slots
_EPS_UV = 0.05


@wp.func
def _thread_glow(zd: wp.vec2, th1: float, th2: float, px: float) -> float:
    """A single bit thread: the geodesic arc between boundary angles th1, th2 (thinner
    and softer than an RT-surface glow — a strand, not a cut)."""
    u = wp.vec2(wp.cos(th1), wp.sin(th1))
    v = wp.vec2(wp.cos(th2), wp.sin(th2))
    den = 1.0 + u[0] * v[0] + u[1] * v[1] + 1.0e-6
    c = wp.vec2((u[0] + v[0]) / den, (u[1] + v[1]) / den)
    rad = wp.sqrt(wp.max(wp.dot(c, c) - 1.0, 1.0e-8))
    darc = wp.abs(wp.length(zd - c) - rad)
    w = wp.max(0.0022, 1.1 * px)
    return wp.exp(-(darc * darc) / (w * w))


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, a1: float, a2: float,
                   tha: wp.array(dtype=float), thb: wp.array(dtype=float), n_thr: int):
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
        # ---- the faint hyperbolic stage ----
        rot = 0.03 * time + float(mouse[0]) * 0.004
        cp = wp.cos(rot)
        sp = wp.sin(rot)
        z = wp.vec2(cp * zd[0] - sp * zd[1], sp * zd[0] + cp * zd[1])
        folded = poincare_fold(z)
        edge = tile_edge(folded, px)
        depth = wp.min(folded[2] / 12.0, 1.0)
        col = wp.vec3(0.010, 0.016, 0.042) * (0.6 + 0.9 * depth)
        col = col + wp.vec3(0.12, 0.26, 0.45) * edge * (0.14 + 0.30 * depth)

        # ---- the thread bundle: nested, non-crossing, cyan strands ----
        tsum = float(0.0)
        for k in range(n_thr):
            tsum += _thread_glow(zd, tha[k], thb[k], px)
        col = col + wp.vec3(0.30, 0.85, 0.95) * wp.min(tsum, 2.2) * 0.55

    # ---- the bottleneck: the RT geodesic the flow saturates ----
    inside = wp.clamp((1.03 - r) / 0.03, 0.0, 1.0)
    g_rt = rt_geodesic_glow(zd, a1, a2, px)
    col = col + wp.vec3(0.90, 0.80, 1.00) * (g_rt * 0.9 * inside)

    # ---- the conformal boundary + the interval A ----
    bw = wp.max(0.004, 1.8 * px)
    ring = wp.exp(-((r - 1.0) * (r - 1.0)) / (bw * bw)) + 0.25 * wp.exp(-wp.abs(r - 1.0) * 16.0)
    col = col + wp.vec3(0.55, 0.48, 0.36) * ring * 0.7
    band = wp.exp(-((r - 1.0) * (r - 1.0)) / (9.0 * bw * bw))
    col = col + wp.vec3(1.00, 0.30, 0.78) * (band * in_boundary_arc(th, a1, a2) * 1.6)

    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)

    # the interval sweeps in size; centred on the top of the disk
    theta = 0.45 + 2.30 * 0.5 * (1.0 - math.cos(2.0 * math.pi * tau / _T_CYCLE))
    a1 = 0.5 * math.pi - 0.5 * theta
    a2 = 0.5 * math.pi + 0.5 * theta

    # thread count tracks the entropy: the max flow the interval can carry
    s_now = interval_entropy(theta, _EPS_UV)
    s_max = interval_entropy(0.45 + 2.30, _EPS_UV)
    n_thr = max(4, min(_N_MAX, round(_N_MAX * s_now / s_max)))

    # nested pairing (non-crossing, as an optimal flow is): the k-th thread leaves A at
    # a point nested inward from the ends and lands symmetrically in the complement
    tha, thb = [], []
    for k in range(n_thr):
        f = (k + 0.5) / n_thr
        tha.append(a1 + f * theta)                                  # across A
        thb.append(a2 + (2.0 * math.pi - theta) * (1.0 - f))        # nested landing
    arr_a = wp.array(tha, dtype=float, device=device)
    arr_b = wp.array(thb, dtype=float, device=device)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(a1), float(a2), arr_a, arr_b, int(n_thr)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.45, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.12, preserve_hue=True)


SCENE = Scene(
    name="holo_bit_threads",
    description="entanglement entropy as a maximum flow — Freedman-Headrick bit "
                "threads: nested cyan strands pair every point of a boundary interval "
                "with the complement, their count tracking S(A), and the RT geodesic "
                "glows as the bottleneck the bundle saturates: max flow = min cut by "
                "exact LP duality, verified in the suite by Dinic's algorithm on a "
                "hyperbolic grid (flow == cut to 1e-12, tracking 2 ln sin(theta/2)). "
                "--frames sweeps the interval once.",
    renderer=_render,
)
