"""Electricity toolkit — glowing conductors, arcs, coronas, and fractal bolts.

Shared helpers for the *electricity in motion* strand: current pulses travelling a
wire, the soft corona around a charged conductor, and — the star — a **fractal
lightning bolt** generated on the host by recursive midpoint-displacement with
branching (the standard model of stepped-leader dielectric breakdown), returned as a
dense list of points that a kernel lights up by closest-approach glow (the same
ray-vs-point trick as ``particles.emitter``). See ``docs/research/38-electricity.md``.
"""

import numpy as np
import warp as wp


@wp.func
def pt_glow(ro: wp.vec3, rd: wp.vec3, p: wp.vec3, width: float) -> float:
    """Glow of a point light seen along a ray — Gaussian in the ray's closest
    approach distance. Dense points along a path make a glowing filament."""
    v = p - ro
    t = wp.max(wp.dot(v, rd), 0.0)
    d = wp.length(ro + rd * t - p)
    return wp.exp(-(d * d) / (width * width))


@wp.func
def seg_glow(ro: wp.vec3, rd: wp.vec3, a: wp.vec3, b: wp.vec3, width: float) -> float:
    """Glow of a glowing segment a->b, sampled at a few points along it."""
    g = float(0.0)
    for s in range(5):
        u = (float(s) + 0.5) / 5.0
        g += pt_glow(ro, rd, a + (b - a) * u, width)
    return g / 5.0


@wp.func
def current_pulse(u: float, time: float, speed: float, level: float) -> float:
    """Brightness of a current pulse at path-fraction u in [0,1] — a steady thread
    plus bright travelling pulses, fading up with `level` (0..1) as power is drawn."""
    p = 0.5 + 0.5 * wp.sin(u * 26.0 - time * speed)
    p = p * p * p
    return (0.35 + 0.65 * level) * (0.4 + 1.8 * p * level)


@wp.func
def corona(ro: wp.vec3, rd: wp.vec3, c: wp.vec3, r: float) -> float:
    """Soft radial glow (the corona around a charged conductor / an arc terminus)."""
    v = c - ro
    t = wp.max(wp.dot(v, rd), 0.0)
    d = wp.length(ro + rd * t - c)
    return wp.exp(-(d * d) / (r * r))


def generate_bolt(a, b, seed, gens=6, jitter=0.85, branch_prob=0.42, pts_per_seg=3):
    """Fractal lightning a->b by recursive midpoint displacement + branching.

    Each segment is split at a perpendicular-displaced midpoint (dielectric
    breakdown wanders); with some probability a side branch forks off and dies
    quickly. Returns an ``(M, 3)`` float32 array of dense points along every
    segment (main channel + branches) for a kernel to light up. Deterministic in
    ``seed`` so a strike is stable within its flash and re-rolls on the next.
    """
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    segs = []

    def sub(p0, p1, depth, amp):
        if depth == 0:
            segs.append((p0, p1))
            return
        d = p1 - p0
        perp = np.cross(d, rng.randn(3))
        n = np.linalg.norm(perp)
        if n > 1e-6:
            perp = perp / n
        mid = (p0 + p1) * 0.5 + perp * amp * rng.uniform(-1.0, 1.0)
        sub(p0, mid, depth - 1, amp * 0.62)
        sub(mid, p1, depth - 1, amp * 0.62)
        if depth >= 2 and rng.uniform() < branch_prob:
            bdir = p1 - p0
            L = np.linalg.norm(bdir)
            if L > 1e-6:
                bdir = bdir / L
            end = mid + bdir * amp * rng.uniform(1.6, 3.4) + perp * amp * rng.uniform(-2.0, 2.0)
            sub(mid, end, max(depth - 2, 1), amp * 0.5)

    sub(a, b, gens, jitter)
    pts = []
    for (p0, p1) in segs:
        for k in range(pts_per_seg):
            u = (k + 0.5) / pts_per_seg
            pts.append(p0 + (p1 - p0) * u)
    if not pts:
        pts = [a, b]
    return np.asarray(pts, dtype=np.float32)


def upload_points(pts, device):
    """Upload an (M,3) point array as a Warp vec3 array (padded to >=1 point)."""
    if pts.shape[0] == 0:
        pts = np.zeros((1, 3), dtype=np.float32)
    return wp.array(pts.astype(np.float32), dtype=wp.vec3, device=device), pts.shape[0]
