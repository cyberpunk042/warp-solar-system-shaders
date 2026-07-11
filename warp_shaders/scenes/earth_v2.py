"""Earth v2 — the engine flagship.

A planet-scale globe that composes the whole engine: procedural continents,
a PBR (GGX) ocean with a real sun-glint, night-side city lights and a day/night
terminator, the physically based atmosphere (P3) for the limb halo + aerial
perspective, and a volumetric cloud shell (P4 density model, spherical) raymarched
around the globe. Everything scales with `--quality`. iMouse orbits / moves the sun.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.atmosphere import atmosphere, sample_counts
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.pbr import shade_pbr
from ..earthgfx import stars
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import domain_warp3, fbm3, worley3
from ..scene import Scene

_RG = 6360000.0            # ground radius (matches engine.atmosphere)
_CB = _RG + 1500.0         # cloud shell base
_CT = _RG + 7000.0         # cloud shell top


@wp.func
def _surface(n: wp.vec3, rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    # continents (domain-warped) + ocean/land/ice
    c = domain_warp3(n * 1.6, 4, 0.5)
    land = wp.smoothstep(0.50, 0.55, c)
    lat = wp.abs(n[1])
    elev = fbm3(n * 4.0, 5)
    dry = fbm3(n * 3.0 + wp.vec3(5.0, 5.0, 5.0), 4)

    ocean = wp.vec3(0.015, 0.05, 0.16) * (1.0 - wp.smoothstep(0.4, 0.55, c)) \
        + wp.vec3(0.03, 0.19, 0.34) * wp.smoothstep(0.4, 0.55, c)
    land_c = wp.vec3(0.08, 0.22, 0.06) * (1.0 - wp.smoothstep(0.4, 0.7, dry)) \
        + wp.vec3(0.46, 0.38, 0.22) * wp.smoothstep(0.4, 0.7, dry)
    snow = wp.smoothstep(0.60, 0.78, elev * 0.5 + wp.smoothstep(0.66, 0.9, lat))
    land_c = land_c * (1.0 - snow) + wp.vec3(0.9, 0.92, 0.96) * snow
    albedo = ocean * (1.0 - land) + land_c * land
    is_ocean = 1.0 - land

    v = -rd
    ndl = wp.dot(n, sun)
    day = wp.smoothstep(-0.1, 0.15, ndl)
    sun_col = wp.vec3(1.0, 0.97, 0.92)

    if is_ocean > 0.5:
        # PBR ocean: GGX specular sun-glint + Fresnel, over a dim sky-ambient
        lit = shade_pbr(n, v, sun, albedo, 0.08, 0.0, sun_col) * 4.5
        col = lit + wp.cw_mul(albedo, wp.vec3(0.06, 0.11, 0.17))
    else:
        col = albedo * (0.06 + 1.25 * wp.max(ndl, 0.0))
        lit = shade_pbr(n, v, sun, albedo, 0.7, 0.0, sun_col) * 1.4
        col = col * 0.7 + lit

    # night-side city lights
    night = 1.0 - day
    city = land * wp.smoothstep(0.58, 0.72, fbm3(n * 9.0, 3))
    col = col + wp.vec3(1.0, 0.72, 0.32) * (city * night * 1.6)
    return col


@wp.func
def _cloud_density_shell(p: wp.vec3, time: float, cov: float) -> float:
    r = wp.length(p)
    hf = wp.clamp((r - _CB) / (_CT - _CB), 0.0, 1.0)
    d = p / r
    w = time * 0.0004
    shape = fbm3(d * 7.0 + wp.vec3(w, 0.0, 0.0), 5)
    dens = wp.clamp((shape - (1.0 - cov)) / wp.max(cov, 1e-3), 0.0, 1.0)
    gh = wp.smoothstep(0.0, 0.2, hf) * wp.smoothstep(1.0, 0.5, hf)
    dens = dens * gh
    det = worley3(d * 26.0 + wp.vec3(w * 3.0, 0.0, 0.0))
    return wp.clamp(dens - (1.0 - dens) * det * 0.28, 0.0, 1.0)


@wp.func
def _march_clouds_shell(ro: wp.vec3, rd: wp.vec3, sun: wp.vec3, time: float,
                        globe_t: float, cov: float, steps: int, lsteps: int,
                        sun_col: wp.vec3, amb: wp.vec3) -> wp.vec4:
    top = _rs(ro, rd, _CT)
    if top[1] < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, 1.0)
    t0 = wp.max(top[0], 0.0)
    t1 = top[1]
    if globe_t > 0.0:
        t1 = wp.min(t1, globe_t)
    if t1 <= t0:
        return wp.vec4(0.0, 0.0, 0.0, 1.0)

    seg = (t1 - t0) / float(steps)
    sigma = 6.0e-4
    trans = float(1.0)
    scat = wp.vec3(0.0, 0.0, 0.0)
    g_ph = 0.6 * (1.0 - 0.55 * 0.55) / wp.pow(1.0 + 0.55 * 0.55 - 2.0 * 0.55 * wp.dot(rd, sun), 1.5)
    seg_l = (_CT - _CB) / float(lsteps)

    t = t0 + 0.5 * seg
    for _ in range(steps):
        p = ro + rd * t
        dens = _cloud_density_shell(p, time, cov)
        if dens > 0.002:
            up = p / wp.length(p)
            od_l = float(0.0)
            tl = seg_l * 0.5
            for _ in range(lsteps):
                od_l += _cloud_density_shell(p + up * tl, time, cov) * seg_l
                tl += seg_l
            t_light = wp.exp(-sigma * od_l)
            powder = 1.0 - wp.exp(-2.0 * dens)
            sample = sun_col * (t_light * g_ph * powder * 5.0) + amb
            d_tr = wp.exp(-sigma * dens * seg)
            scat += sample * (trans * (1.0 - d_tr))
            trans *= d_tr
            if trans < 0.02:
                break
        t += seg
    return wp.vec4(scat[0], scat[1], scat[2], trans)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  time: float, vs: int, ls: int, csteps: int, clsteps: int,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    g = _rs(ro, rd, _RG)
    globe_t = float(-1.0)
    if g[0] > 0.0:
        globe_t = g[0]

    col = stars(rd)
    if globe_t > 0.0:
        p = ro + rd * globe_t
        n = wp.normalize(p)
        col = _surface(n, rd, sun)

    # physically based atmosphere (limb halo + aerial perspective / day haze).
    # Scaled: full-strength in-scatter would flood the disk; this reads as haze
    # over the surface and a bright limb against space.
    atm = atmosphere(ro, rd, sun, vs, ls)
    col = col + atm * 0.32

    # volumetric cloud shell over the surface
    cl = _march_clouds_shell(ro, rd, sun, time, globe_t, 0.55, csteps, clsteps,
                             wp.vec3(1.0, 0.97, 0.92), wp.vec3(0.35, 0.45, 0.62) * 0.4)
    col = col * cl[3] + wp.vec3(cl[0], cl[1], cl[2])

    img[i, j] = col


def _cloud_counts(name):
    return {"low": (28, 4), "medium": (48, 6), "high": (72, 8),
            "ultra": (112, 12)}.get(name, (48, 6))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    vs, ls = sample_counts(tier.name)
    cs, cls = _cloud_counts(tier.name)

    az = 0.6 + float(mouse[0]) * 0.01 + time * 0.05
    el = 0.32
    dist = 4.2 * _RG
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=40.0, aspect=width / height)

    sel = 0.2 + 0.5 * math.sin(time * 0.1) + float(mouse[1]) * 0.004
    saz = az + 0.7
    sun = wp.vec3(math.cos(sel) * math.sin(saz), math.sin(sel), math.cos(sel) * math.cos(saz))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, float(time), int(vs), int(ls), int(cs), int(cls),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=2.0, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.1)


SCENE = Scene(
    name="earth_v2",
    description="Flagship Earth: PBR ocean + real atmosphere + volumetric clouds. --quality low..ultra.",
    renderer=_render,
)
