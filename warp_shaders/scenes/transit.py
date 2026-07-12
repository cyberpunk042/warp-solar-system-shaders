"""Transit / eclipse — an exoplanet crossing the face of its star.

A dark planet drifts across a limb-darkened star: the star's disk dims where the
planet occults it (the transit light-curve dip), and the planet's thin atmosphere
lights up as a **backlit ring** where starlight refracts through it — the same
geometry that lets us detect atmospheres in real exoplanet transits. Animate with
``--frames``. See ``docs/research/20-more-cosmos-worlds-crossstrand.md``.
"""

import math

import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.color import kelvin_to_rgb
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3
from ..scene import Scene

_STAR_R = 3.0                                    # star at origin
_PLANET_R = 0.62


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, planet_c: wp.vec3,
                  star_k: float, star_glow: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    col = stars(rd)

    star_rgb = kelvin_to_rgb(star_k)

    # the star (emissive sphere, limb-darkened) — behind the planet
    gs = _rs(ro, rd, _STAR_R)
    star_hit = gs[0] > 0.0 and gs[0] < 1.0e29
    if star_hit:
        ns = wp.normalize(ro + rd * gs[0])
        mu = wp.max(wp.dot(ns, -rd), 0.0)               # cos(angle) to line of sight
        limb = 0.28 + 0.72 * wp.pow(mu, 0.7)            # limb darkening (edge dims + reddens)
        gran = 0.88 + 0.18 * fbm3(ns * 9.0, 4)          # faint granulation
        edge_warm = wp.vec3(1.0, 0.72, 0.42)            # cooler, redder limb
        tint = star_rgb * mu + edge_warm * (1.0 - mu)
        col = tint * (star_glow * limb * gran)
    # soft chromospheric glow just outside the limb
    bimp = wp.length(wp.cross(ro, rd))
    if not star_hit:
        halo = wp.exp(-(bimp - _STAR_R) * 2.2) * wp.step(bimp - _STAR_R)
        col = col + star_rgb * (halo * star_glow * 0.25)

    # the planet (dark silhouette) in front, with a backlit atmosphere ring
    gp = _rs(ro - planet_c, rd, _PLANET_R)
    if gp[0] > 0.0 and gp[0] < 1.0e29:
        npl = wp.normalize((ro - planet_c) + rd * gp[0])
        # backlit atmosphere: bright where the limb is thin (grazing) and the
        # star is behind it — a refracted orange ring
        rim = wp.pow(1.0 - wp.max(wp.dot(npl, -rd), 0.0), 4.0)
        behind_star = float(0.0)
        if star_hit and gp[0] < gs[0]:
            behind_star = 1.0
        atmo = wp.vec3(1.0, 0.5, 0.22) * (rim * (2.6 + 2.2 * behind_star))
        night = wp.vec3(0.015, 0.015, 0.025)            # near-black planet body
        col = night + atmo

    img[i, j] = col


def _render(width, height, time, mouse, device, period=12.0):
    prog = min(time / period, 1.0)
    # planet tracks left→right across the star's face, slightly below centre
    px = -5.2 + 10.4 * prog
    planet_c = wp.vec3(px, -0.35, 1.6)                  # in front of the star

    az = 0.0 + float(mouse[0]) * 0.01
    el = 0.02 + float(mouse[1]) * 0.004
    dist = 11.5
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=42.0, aspect=width / height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, planet_c, 5700.0, 1.55, int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5,
                     radius=max(3, int(min(width, height) * 0.02)), passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="transit",
    description="An exoplanet transiting its star — a limb-darkened stellar disk "
                "occulted by a dark planet whose thin atmosphere lights up as a "
                "backlit refracted ring. Animate with --frames.",
    renderer=_render,
)
