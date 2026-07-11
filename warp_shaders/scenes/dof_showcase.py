"""Depth of field — a focus-pull showcase for the thin-lens camera.

A row of glossy PBR spheres recedes into the distance over a checker floor; the
camera focuses on the middle sphere, so the near and far spheres (and their
specular highlights) melt into bokeh. Each pixel accumulates K lens samples —
`eye + lens_offset` aimed at `focus_point` — with K scaling by quality tier
(`low` is nearly pinhole for speed; `ultra` is creamy). iMouse pans.

This is the copy-me pattern for adding depth of field to any raymarch scene.
"""

import math

import warp as wp

from ..engine import post
from ..engine.material import make_mat, shade_material
from ..engine.shading import sky_gradient, sun_disk
from ..engine.uniforms import (
    Camera, Light, camera_ray_dir, focus_point, lens_offset, make_camera,
    make_light,
)
from ..lod import active_tier
from ..procedural.hash import hash31
from ..procedural.sdf import sd_sphere
from ..scene import Scene

_MAXD = 40.0
_NZ = wp.constant(6)


@wp.func
def _sphere_z(i: int) -> float:
    return -1.5 + float(i) * 2.6


@wp.func
def _map(p: wp.vec3) -> float:
    d = p[1]                                   # floor plane at y=0
    for i in range(_NZ):
        d = wp.min(d, sd_sphere(p - wp.vec3(0.0, 0.7, _sphere_z(i)), 0.7))
    return d


@wp.func
def _hit_id(p: wp.vec3) -> int:
    best = p[1]
    which = int(-1)                            # -1 = floor
    for i in range(_NZ):
        di = sd_sphere(p - wp.vec3(0.0, 0.7, _sphere_z(i)), 0.7)
        if di < best:
            best = di
            which = i
    return which


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.0015
    dx = _map(p + wp.vec3(e, 0.0, 0.0)) - _map(p - wp.vec3(e, 0.0, 0.0))
    dy = _map(p + wp.vec3(0.0, e, 0.0)) - _map(p - wp.vec3(0.0, e, 0.0))
    dz = _map(p + wp.vec3(0.0, 0.0, e)) - _map(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    base = sky_gradient(rd, wp.vec3(0.60, 0.66, 0.78), wp.vec3(0.15, 0.28, 0.60))
    return base + sun_disk(rd, sun, wp.vec3(1.0, 0.95, 0.85), 0.9993, 0.5)


@wp.func
def _shade(ro: wp.vec3, rd: wp.vec3, light: Light) -> wp.vec3:
    t = float(0.0)
    hit = int(0)
    for _ in range(140):
        p = ro + rd * t
        d = _map(p)
        if d < 0.0006 * t + 0.0003:
            hit = 1
            break
        t += d * 0.9
        if t > _MAXD:
            break
    if hit == 0:
        return _sky(rd, light.dir)

    p = ro + rd * t
    n = _normal(p)
    which = _hit_id(p)
    if which < 0:
        chk = wp.floor(p[0] * 0.7) + wp.floor(p[2] * 0.7)
        g = 0.18 + 0.14 * wp.abs(chk - 2.0 * wp.floor(chk * 0.5))
        mat = make_mat(wp.vec3(g, g, g), 0.6, 0.0)
    else:
        # each sphere a different hue; middle ones metallic for punchy bokeh
        h = float(which) / float(_NZ)
        col = wp.vec3(0.5 + 0.5 * wp.sin(h * 6.28),
                      0.5 + 0.5 * wp.sin(h * 6.28 + 2.1),
                      0.5 + 0.5 * wp.sin(h * 6.28 + 4.2))
        mat = make_mat(col * 0.7 + wp.vec3(0.2, 0.2, 0.2), 0.22, 0.6)
    direct = shade_material(mat, n, -rd, light.dir, light.color, light.intensity)
    amb = wp.cw_mul(_sky(n, light.dir), mat.albedo) * 0.25
    return direct + amb


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, light: Light,
                  dof_samples: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0

    fp = focus_point(cam, u, v)
    acc = wp.vec3(0.0, 0.0, 0.0)
    for k in range(dof_samples):
        # per-sample jitter on the aperture disk (decorrelated per pixel + k)
        seed = wp.vec3(float(j) * 0.017, float(i) * 0.017, float(k) * 1.37)
        s1 = hash31(seed)
        s2 = hash31(seed + wp.vec3(5.2, 1.3, 9.1))
        ro = cam.eye + lens_offset(cam, s1, s2)
        rd = wp.normalize(fp - ro)
        acc += _shade(ro, rd, light)
    img[i, j] = acc / float(dof_samples)


def _samples(name):
    return {"low": 6, "medium": 14, "high": 28, "ultra": 52}.get(name, 14)


def _render(width, height, time, mouse, device):
    tier = active_tier()
    k = _samples(tier.name)

    az = 0.35 + float(mouse[0]) * 0.006 + time * 0.05
    eye = (2.1 * math.cos(az), 1.15, -3.6 + 0.6 * math.sin(az))
    mid = (0.0, 0.7, _z_mid())            # focus on the middle sphere
    cam = make_camera(eye, mid, fov_deg=40.0, aspect=width / height,
                      aperture=0.085, focus_dist=None)
    light = make_light((0.5, 0.8, -0.35), (1.0, 0.96, 0.9), 3.4)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, light, int(k), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5, radius=r, passes=2)
    out = post.tonemap(hdr, mode="aces", exposure=1.1)
    return post.vignette(out, 0.25)


def _z_mid():
    # host mirror of _sphere_z for the middle sphere (index 2 of 6)
    return -1.5 + 2.0 * 2.6


SCENE = Scene(
    name="dof_showcase",
    description="Depth-of-field focus pull: a row of PBR spheres, middle in focus, near/far bokeh. --quality low..ultra.",
    renderer=_render,
)
