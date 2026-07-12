"""Buildings as signed distance fields — a parametric architecture kit.

Original Warp reimplementation of standard SDF-architecture techniques (Quilez
distance functions + domain repetition): a body box with a **window-grid facade**
(carved by domain repetition + subtraction), parapet / pitched roofs, and a whole
**city** from one function via per-lot domain repetition + hashed variation. Built
to sphere-trace like any SDF — and to drop into `blast.render`'s per-cell
instancing later, so a nuke can be tested on a city. See
``docs/research/17-buildings.md``. Inspired by (not copied from) ShaderToy
studies by kishimisu and dr2.
"""

from __future__ import annotations

import warp as wp

from ..procedural.hash import hash21
from ..procedural.sdf import op_subtract, op_union, sd_box


@wp.func
def _rep(x: float, c: float) -> float:
    """Domain repetition: fold `x` into one period-`c` cell centred on 0."""
    return x - c * wp.floor(x / c + 0.5)


@wp.func
def _repid(x: float, c: float) -> float:
    """The integer id of the period-`c` cell `x` falls in."""
    return wp.floor(x / c + 0.5)


@wp.func
def sd_triprism(p: wp.vec3, hx: float, hz: float) -> float:
    """Triangular prism (a gable roof): triangle sized by `hx`, length `hz`."""
    q = wp.vec3(wp.abs(p[0]), wp.abs(p[1]), wp.abs(p[2]))
    return wp.max(q[2] - hz, wp.max(q[0] * 0.866025 + p[1] * 0.5, -p[1]) - hx * 0.5)


@wp.func
def sd_tower(p: wp.vec3, half: wp.vec3, floor_h: float, win_w: float) -> float:
    """A modern high-rise: body + protruding **floor bands** + corner pilasters +
    parapet + base plinth (windows are a shading detail, `window_mask`, so the SDF
    stays a clean solid). `half` = half-extents about the body centre."""
    body = sd_box(p, half)
    # floor-band ledges: a thin slab poking out at every floor -> facade relief
    band = sd_box(wp.vec3(p[0], _rep(p[1], floor_h), p[2]),
                  wp.vec3(half[0] + 0.12, floor_h * 0.09, half[2] + 0.12))
    band = wp.max(band, wp.abs(p[1]) - half[1])      # clamp to the building height
    d = op_union(body, band)
    # corner pilasters: vertical ribs at the four edges
    pil = sd_box(wp.vec3(wp.abs(p[0]) - half[0], p[1], wp.abs(p[2]) - half[2]),
                 wp.vec3(0.18, half[1], 0.18))
    d = op_union(d, pil)
    # parapet: a hollow rim at the roofline (a thin ledge that reads as a cap)
    top = p - wp.vec3(0.0, half[1], 0.0)
    ledge = sd_box(top, wp.vec3(half[0] + 0.15, 0.14, half[2] + 0.15))
    ledge = op_subtract(ledge, sd_box(top, wp.vec3(half[0] - 0.12, 0.5, half[2] - 0.12)))
    d = op_union(body, ledge)
    # base plinth
    base = sd_box(p - wp.vec3(0.0, -half[1], 0.0), wp.vec3(half[0] + 0.2, 0.5, half[2] + 0.2))
    return op_union(d, base)


@wp.func
def sd_house(p: wp.vec3, half: wp.vec3, roof_h: float) -> float:
    """A house: box body + pitched (gable) roof + a carved door."""
    body = sd_box(p, half)
    rp = p - wp.vec3(0.0, half[1], 0.0)
    roof = sd_triprism(rp, roof_h, half[2] * 1.02)
    d = op_union(body, roof)
    door = sd_box(p - wp.vec3(0.0, -half[1] + 0.9, -half[2]), wp.vec3(0.5, 0.9, 0.35))
    d = op_subtract(d, door)                          # carve the door out
    # a couple of windows on the front
    winb = sd_box(wp.vec3(_rep(p[0], 1.7), p[1] - half[1] * 0.2, p[2] - half[2]),
                  wp.vec3(0.4, 0.4, 0.3))
    return op_subtract(d, winb)


@wp.func
def sd_block(p: wp.vec3, half: wp.vec3, floor_h: float) -> float:
    """A low, wide office block (solid) with a roof cap; banded windows come
    from `window_mask` in shading."""
    body = sd_box(p, half)
    cap = sd_box(p - wp.vec3(0.0, half[1], 0.0), wp.vec3(half[0] + 0.12, 0.16, half[2] + 0.12))
    return op_union(body, cap)


@wp.func
def city_de(p: wp.vec3, lot: float, seed: float) -> wp.vec4:
    """A city block by domain repetition: each lot grows a different building
    (height / footprint / variant hashed from the lot id). Returns
    ``(dist, height, variant, lot_rand)``. Streets are the gaps between lots."""
    idx = _repid(p[0], lot)
    idz = _repid(p[2], lot)
    rnd = hash21(wp.vec2(idx + seed, idz - seed))
    rnd2 = hash21(wp.vec2(idx * 1.7 + 5.3, idz * 2.3 + 9.1))
    q = wp.vec3(_rep(p[0], lot), p[1], _rep(p[2], lot))

    h = 4.0 + 26.0 * rnd * rnd                       # tall towers are rarer
    w = lot * 0.5 * (0.32 + 0.2 * rnd2)              # footprint (rest is street)
    half = wp.vec3(w, h, w)
    qb = wp.vec3(q[0], q[1] - h, q[2])               # base sits on the ground
    if rnd2 < 0.28:
        d = sd_block(qb, wp.vec3(w, h * 0.45, w), 1.5)
    else:
        d = sd_tower(qb, half, 1.6, 0.5)
    return wp.vec4(d, h, rnd2, rnd)


@wp.func
def suburb_de(p: wp.vec3, lot: float, seed: float) -> wp.vec4:
    """A suburb by domain repetition: each lot grows a different **house**
    (pitched roof). Returns ``(dist, body_half_h, variant, lot_rand)``."""
    idx = _repid(p[0], lot)
    idz = _repid(p[2], lot)
    rnd = hash21(wp.vec2(idx + seed, idz - seed))
    rnd2 = hash21(wp.vec2(idx * 1.3 + 2.1, idz * 1.9 + 4.7))
    q = wp.vec3(_rep(p[0], lot), p[1], _rep(p[2], lot))
    half = wp.vec3(2.2 + 0.9 * rnd, 1.7 + 0.7 * rnd2, 2.8 + 1.0 * rnd)
    qb = wp.vec3(q[0], q[1] - half[1], q[2])
    d = sd_house(qb, half, 2.0 + 0.9 * rnd2)
    return wp.vec4(d, half[1], rnd2, rnd)


@wp.func
def window_mask(p: wp.vec3, cell_xy: float, floor_h: float) -> float:
    """1 inside a window pane, 0 on the mullions — for glass vs concrete
    material and lit-window selection at a surface point."""
    fx = wp.abs(_rep(p[0] + p[2], cell_xy))          # column phase (both facades)
    fy = wp.abs(_rep(p[1], floor_h))
    return wp.step(cell_xy * 0.34 - fx) * wp.step(floor_h * 0.34 - fy)
