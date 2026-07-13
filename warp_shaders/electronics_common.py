"""Shared material palette + lighting + finish for the electronics component scenes.

Every part of a computer is one of a small set of real materials — silicon,
copper, gold, solder, fibreglass PCB, moulded plastic, ceramic, aluminium — so
they are defined once here (albedo + roughness + metalness) and shaded with a
common two-light GGX studio setup + ambient occlusion, then run through the engine
bloom + tonemap. Individual component scenes supply their own SDF ``_map`` and
material ids and call :func:`lit`. See ``docs/research/35-electronics-components.md``.
"""

import numpy as np
import warp as wp

from .engine import post
from .engine.pbr import shade_pbr

# material id -> (albedo, roughness, metalness)
SILICON = 0      # dark blue-grey semiconductor
COPPER = 1
GOLD = 2
SOLDER = 3       # tin-silver, dull metal
PCB = 4          # fibreglass green
PLASTIC = 5      # black epoxy moulding
CERAMIC = 6      # tan
ALU = 7          # aluminium / steel can
GLASS = 8
TINT_RED = 9     # painted band / marking
TINT_BLUE = 10


@wp.func
def mat_albedo(m: int) -> wp.vec3:
    if m == 0:
        return wp.vec3(0.28, 0.30, 0.36)
    if m == 1:
        return wp.vec3(0.72, 0.44, 0.22)
    if m == 2:
        return wp.vec3(0.92, 0.72, 0.30)
    if m == 3:
        return wp.vec3(0.68, 0.70, 0.74)
    if m == 4:
        return wp.vec3(0.05, 0.32, 0.14)
    if m == 5:
        return wp.vec3(0.045, 0.045, 0.055)
    if m == 6:
        return wp.vec3(0.80, 0.71, 0.52)
    if m == 7:
        return wp.vec3(0.80, 0.82, 0.86)
    if m == 8:
        return wp.vec3(0.10, 0.12, 0.14)
    if m == 9:
        return wp.vec3(0.75, 0.12, 0.10)
    return wp.vec3(0.12, 0.22, 0.7)


@wp.func
def mat_rough(m: int) -> float:
    if m == 0:
        return 0.30
    if m == 1:
        return 0.32
    if m == 2:
        return 0.28
    if m == 3:
        return 0.42
    if m == 4:
        return 0.55
    if m == 5:
        return 0.6
    if m == 6:
        return 0.7
    if m == 7:
        return 0.35
    if m == 8:
        return 0.12
    return 0.55


@wp.func
def mat_metal(m: int) -> float:
    if m == 1 or m == 2 or m == 3 or m == 7:
        return 1.0                       # copper / gold / solder / aluminium
    if m == 0:
        return 0.6                       # polished silicon reads half-metallic
    return 0.0


@wp.func
def studio_sky(rd: wp.vec3) -> wp.vec3:
    """A soft dark-to-light studio gradient (product-shot backdrop)."""
    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    return wp.vec3(0.02, 0.025, 0.035) * (1.0 - up) + wp.vec3(0.10, 0.12, 0.16) * up


@wp.func
def lit(n: wp.vec3, rd: wp.vec3, m: int, ao: float, emit: wp.vec3) -> wp.vec3:
    """Two-light GGX + sky-ambient shading for a hit with material ``m``."""
    alb = mat_albedo(m)
    rg = mat_rough(m)
    mt = mat_metal(m)
    v = -rd
    key = wp.normalize(wp.vec3(0.5, 0.85, 0.42))
    fill = wp.normalize(wp.vec3(-0.6, 0.35, -0.55))
    c = shade_pbr(n, v, key, alb, rg, mt, wp.vec3(1.0, 0.98, 0.94)) * 2.4
    c = c + shade_pbr(n, v, fill, alb, rg, mt, wp.vec3(0.45, 0.55, 0.8)) * 0.7
    amb = wp.cw_mul(alb, studio_sky(n)) * (1.6 * ao)
    rim = wp.pow(1.0 - wp.max(wp.dot(n, v), 0.0), 3.0)
    return c * ao + amb + alb * (rim * 0.15) + emit


def finish(hdr, width, height, exposure=1.05, threshold=1.3, strength=0.35):
    r = max(2, int(min(width, height) * 0.014))
    hdr = post.bloom(np.asarray(hdr, np.float32), threshold=threshold,
                     strength=strength, radius=r, passes=3, octaves=3)
    return post.tonemap(hdr, mode="aces", exposure=exposure, preserve_hue=True)
