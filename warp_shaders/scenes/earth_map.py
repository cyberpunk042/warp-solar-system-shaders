"""Earth from a baked equirectangular map — demonstrates the texture system.

A detailed Earth albedo map is baked ONCE with the procedural toolkit (high
octave count, crisper coastlines than real-time), cached, and sampled per frame
via `textures.sample_equirect`. A real NASA Blue-Marble JPG can replace the bake
(`textures.load_equirect`) with no other change. Globe is lit with a PBR ocean
sun-glint + lambert land + night lights, wrapped in the physically based
atmosphere. iMouse orbits / moves the sun.
"""

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.atmosphere import atmosphere, sample_counts
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.pbr import shade_pbr
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import domain_warp3, fbm3
from ..scene import Scene
from ..textures import sample_equirect, to_texture

_RG = 6360000.0
_map_cache = {}


@wp.func
def _dir_from_uv(u: float, v: float) -> wp.vec3:
    lon = (u - 0.5) * 6.28318530718
    lat = (0.5 - v) * 3.14159265
    cl = wp.cos(lat)
    return wp.vec3(cl * wp.cos(lon), wp.sin(lat), cl * wp.sin(lon))


@wp.kernel
def bake_earth(tex: wp.array2d(dtype=wp.vec3), width: int, height: int):
    i, j = wp.tid()
    u = (float(j) + 0.5) / float(width)
    v = (float(i) + 0.5) / float(height)
    n = _dir_from_uv(u, v)

    c = domain_warp3(n * 1.6, 5, 0.6)
    coast = fbm3(n * 8.0, 4)
    land = wp.smoothstep(0.49, 0.53, c + 0.06 * (coast - 0.5))
    lat = wp.abs(n[1])
    elev = fbm3(n * 5.0, 6)
    dry = fbm3(n * 3.0 + wp.vec3(5.0, 5.0, 5.0), 4)

    ocean = wp.vec3(0.012, 0.045, 0.14) * (1.0 - wp.smoothstep(0.4, 0.5, c)) \
        + wp.vec3(0.02, 0.16, 0.30) * wp.smoothstep(0.4, 0.5, c)
    land_c = wp.vec3(0.09, 0.22, 0.07) * (1.0 - wp.smoothstep(0.4, 0.7, dry)) \
        + wp.vec3(0.5, 0.42, 0.24) * wp.smoothstep(0.4, 0.7, dry)
    snow = wp.smoothstep(0.6, 0.78, elev * 0.5 + wp.smoothstep(0.64, 0.9, lat))
    land_c = land_c * (1.0 - snow) + wp.vec3(0.92, 0.94, 0.97) * snow
    tex[i, j] = ocean * (1.0 - land) + land_c * land


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), tex: wp.array2d(dtype=wp.vec3),
                  cam: Camera, sun: wp.vec3, vs: int, ls: int,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)

    g = _rs(ro, rd, _RG)
    col = stars(rd)
    if g[0] > 0.0:
        n = wp.normalize(ro + rd * g[0])
        albedo = sample_equirect(tex, n)
        is_ocean = wp.smoothstep(0.04, 0.10, albedo[2] - albedo[0])
        v_dir = -rd
        ndl = wp.dot(n, sun)
        day = wp.smoothstep(-0.1, 0.15, ndl)
        sun_col = wp.vec3(1.0, 0.97, 0.92)
        ocean_lit = shade_pbr(n, v_dir, sun, albedo, 0.08, 0.0, sun_col) * 4.5 \
            + wp.cw_mul(albedo, wp.vec3(0.06, 0.11, 0.17))
        land_lit = albedo * (0.06 + 1.25 * wp.max(ndl, 0.0))
        col = land_lit * (1.0 - is_ocean) + ocean_lit * is_ocean
        # night lights on land
        night = 1.0 - day
        city = (1.0 - is_ocean) * wp.smoothstep(0.58, 0.72, fbm3(n * 9.0, 3))
        col = col + wp.vec3(1.0, 0.72, 0.32) * (city * night * 1.6)

    col = col + atmosphere(ro, rd, sun, vs, ls) * 0.32
    img[i, j] = col


def _get_map(device):
    key = (device, 1024)
    if key not in _map_cache:
        w, h = 1024, 512
        tex = wp.zeros((h, w), dtype=wp.vec3, device=device)
        wp.launch(bake_earth, dim=(h, w), inputs=[tex, w, h], device=device)
        wp.synchronize_device(device)
        _map_cache[key] = tex
    return _map_cache[key]


def _render(width, height, time, mouse, device):
    tier = active_tier()
    vs, ls = sample_counts(tier.name)
    tex = _get_map(device)

    az = 0.6 + float(mouse[0]) * 0.01 + time * 0.05
    el = 0.32
    dist = 4.2 * _RG
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=40.0, aspect=width / height)

    sel = 0.25 + 0.45 * math.sin(time * 0.1) + float(mouse[1]) * 0.004
    saz = az + 0.6
    sun = wp.vec3(math.cos(sel) * math.sin(saz), math.sin(sel), math.cos(sel) * math.cos(saz))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, tex, cam, sun, int(vs), int(ls), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=2.0, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.1)


SCENE = Scene(
    name="earth_map",
    description="Globe from a baked equirectangular map (drop-in NASA texture) + atmosphere. --quality low..ultra.",
    renderer=_render,
)
