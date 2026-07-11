"""Analytic soft-shadow + ambient-occlusion showcase (engine/shadow.py).

Three spheres floating over a checkered plane, ray-traced with *no* SDF march:
primary hits come from :mod:`engine.intersect` (``ray_sphere`` / ``ray_plane``),
and all the light transport is the closed-form :mod:`engine.shadow` primitives —

- **soft shadows** the spheres cast onto the plane and onto each other come from
  :func:`soft_shadow_sphere` (a real penumbra that widens with distance);
- **ambient occlusion** — the contact darkening where a sphere nears the plane,
  and where spheres crowd each other — comes from :func:`sphere_ao` (Quilez's
  exact analytic sphere occlusion).

Because both are analytic, the shadows are noise-free at any resolution and cost
three sphere tests per pixel — the point of the module. `--quality` only changes
the supersampling the launcher applies.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.intersect import ray_plane, ray_sphere
from ..engine.pbr import shade_pbr
from ..engine.shadow import soft_shadow_sphere, sphere_ao
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..scene import Scene

_C0 = wp.constant(wp.vec3(-1.7, 0.0, 0.0))
_C1 = wp.constant(wp.vec3(0.0, 0.0, 0.0))
_C2 = wp.constant(wp.vec3(1.7, 0.0, 0.0))
_RA = wp.constant(0.7)
_PLANE_Y = wp.constant(-0.75)
_K = wp.constant(12.0)                       # penumbra hardness


@wp.func
def _centers(idx: int, time: float) -> wp.vec3:
    bob = 0.18 * wp.sin(time * 1.3)
    if idx == 0:
        return _C0 + wp.vec3(0.0, bob, 0.0)
    if idx == 1:
        return _C1 + wp.vec3(0.0, -bob, 0.0)
    return _C2 + wp.vec3(0.0, bob, 0.0)


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    base = wp.vec3(0.52, 0.60, 0.76) * (1.0 - up) + wp.vec3(0.12, 0.28, 0.60) * up
    s = wp.pow(wp.max(wp.dot(rd, sun), 0.0), 64.0)
    return base + wp.vec3(1.0, 0.9, 0.7) * (s * 4.0)


@wp.func
def _shadow_all(p: wp.vec3, sun: wp.vec3, skip: int, time: float) -> float:
    """Product of the soft shadows cast by every sphere except `skip`."""
    sh = float(1.0)
    for k in range(3):
        if k != skip:
            sh = sh * soft_shadow_sphere(p, sun, _centers(k, time), _RA, _K)
    return sh


@wp.func
def _ao_all(p: wp.vec3, n: wp.vec3, skip: int, time: float) -> float:
    """Product of the analytic ambient-occlusion visibility of every sphere
    except `skip` (contact darkening on the plane + between spheres)."""
    ao = float(1.0)
    for k in range(3):
        if k != skip:
            ao = ao * sphere_ao(p, n, _centers(k, time), _RA)
    return ao


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera,
                  width: int, height: int, time: float, sun: wp.vec3):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    # nearest of {3 spheres, ground plane}
    best_t = float(1.0e30)
    which = int(-1)                              # -1 sky, 0..2 sphere, 3 plane
    for k in range(3):
        h = ray_sphere(ro, rd, _centers(k, time), _RA)
        if h[1] > h[0] and h[0] > 0.001 and h[0] < best_t:
            best_t = h[0]
            which = k
    tp = ray_plane(ro, rd, wp.vec3(0.0, _PLANE_Y, 0.0), wp.vec3(0.0, 1.0, 0.0))
    if tp > 0.001 and tp < best_t:
        best_t = tp
        which = 3

    if which < 0:
        img[i, j] = _sky(rd, sun)
        return

    p = ro + rd * best_t
    v_dir = -rd
    lcol = wp.vec3(1.0, 0.96, 0.9)

    if which == 3:                               # ground plane
        n = wp.vec3(0.0, 1.0, 0.0)
        chk = wp.floor(p[0] * 0.7) + wp.floor(p[2] * 0.7)
        g = 0.18 + 0.12 * wp.abs(chk - 2.0 * wp.floor(chk * 0.5))
        albedo = wp.vec3(g, g, g)
        sh = _shadow_all(p + n * 0.001, sun, -1, time)
        ao = _ao_all(p, n, -1, time)
        direct = shade_pbr(n, v_dir, sun, albedo, 0.9, 0.0, lcol * (3.0 * sh))
        amb = wp.cw_mul(_sky(n, sun), albedo) * (0.30 * ao)
        img[i, j] = direct + amb
        return

    # sphere hit
    n = wp.normalize(p - _centers(which, time))
    albedo = wp.vec3(0.85, 0.22, 0.18)
    rough = float(0.35)
    if which == 1:
        albedo = wp.vec3(0.95, 0.78, 0.35)
        rough = 0.25
    elif which == 2:
        albedo = wp.vec3(0.22, 0.48, 0.85)
        rough = 0.5
    sh = _shadow_all(p + n * 0.001, sun, which, time)
    ao = _ao_all(p, n, which, time)
    direct = shade_pbr(n, v_dir, sun, albedo, rough, 0.0, lcol * (3.0 * sh))
    amb = wp.cw_mul(_sky(n, sun), albedo) * (0.28 * ao)
    img[i, j] = direct + amb


def _render(width, height, time, mouse, device):
    az = 0.5 + 0.2 * math.sin(time * 0.25) + float(mouse[0]) * 0.01
    el = 0.32 + float(mouse[1]) * 0.005
    dist = 5.2
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el) + 0.6,
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, -0.15, 0.0), fov_deg=42.0, aspect=width / height)

    sa = 0.5 + 0.15 * math.sin(time * 0.3)
    sun = wp.vec3(math.cos(sa) * 0.55, 0.72, math.sin(sa) * 0.42)
    sun = wp.normalize(sun)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, width, height, float(time), sun], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.4, radius=r, passes=2)
    out = post.tonemap(hdr, mode="aces", exposure=1.1)
    return post.vignette(out, 0.28)


SCENE = Scene(
    name="shadow_demo",
    description="Analytic soft shadows + ambient occlusion (engine/shadow.py) — 3 spheres over a plane, no SDF march.",
    renderer=_render,
)
