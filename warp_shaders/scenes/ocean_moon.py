"""Ocean moon — a global-ocean world with its gas-giant parent looming behind.

A tidally-warmed Europa/exo-moon: a sphere sheathed in a deep **ocean** (specular
sun-glint, wind-ruffled highlights), **ice caps** at the poles, thin clouds and a
blue atmosphere rim — with a large banded **gas-giant parent** hanging in the sky
behind it, over a starfield. See ``docs/research/20-more-cosmos-worlds-crossstrand.md``.
"""

import math

import warp as wp

from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3, value3
from ..scene import Scene

_R = 1.0
_PARENT = wp.constant(wp.vec3(3.4, 0.9, -7.0))
_PARENT_R = 3.0


@wp.func
def _bg(rd: wp.vec3) -> wp.vec3:
    st = wp.pow(value3(rd * 320.0), 42.0) * 4.0
    return wp.vec3(0.75, 0.82, 1.0) * st + wp.vec3(0.006, 0.008, 0.016)


@wp.func
def _ocean(n: wp.vec3, rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    ndl = wp.max(wp.dot(n, sun), 0.0)
    lat = wp.abs(n[1])
    # deep ocean + wind-ruffled tone
    ripple = fbm3(n * 22.0, 4)
    deep = wp.vec3(0.02, 0.11, 0.26)
    shallow = wp.vec3(0.04, 0.28, 0.42)
    water = deep * (1.0 - 0.4 * ripple) + shallow * (0.4 * ripple)
    # thin clouds
    cl = wp.smoothstep(0.55, 0.75, fbm3(n * 3.2 + wp.vec3(3.0, 1.0, 7.0), 5))
    # ice caps
    ice = wp.smoothstep(0.72, 0.86, lat + 0.06 * fbm3(n * 6.0, 3))
    base = water * (1.0 - cl) + wp.vec3(0.9, 0.93, 0.98) * cl
    base = base * (1.0 - ice) + wp.vec3(0.85, 0.9, 0.97) * ice
    col = base * (0.05 + 1.05 * ndl)
    # specular sun glint off the water (not the ice)
    refl = rd - n * (2.0 * wp.dot(rd, n))
    spec = wp.pow(wp.max(wp.dot(refl, sun), 0.0), 60.0) * (1.0 - ice) * (1.0 - cl)
    col = col + wp.vec3(1.0, 0.95, 0.8) * (spec * 2.0)
    # atmosphere rim
    rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
    col = col + wp.vec3(0.3, 0.5, 0.9) * (rim * (0.3 + 0.7 * ndl))
    return col


@wp.func
def _parent(n: wp.vec3, rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    # zonal bands: sharp latitude striping + turbulent curl
    turb = fbm3(n * 1.6, 4)
    band = 0.5 + 0.5 * wp.sin(n[1] * 11.0 + 1.4 * turb)
    amber = wp.vec3(0.72, 0.52, 0.32)
    cream = wp.vec3(0.9, 0.83, 0.66)
    rust = wp.vec3(0.55, 0.32, 0.2)
    col = amber * (1.0 - band) + cream * band
    col = col * (1.0 - wp.smoothstep(0.25, 0.05, band)) + rust * wp.smoothstep(0.25, 0.05, band)
    ndl = wp.max(wp.dot(n, sun), 0.0)
    limb = wp.pow(wp.max(wp.dot(n, -rd), 0.0), 0.4)
    return col * (0.06 + 0.95 * ndl) * limb


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    col = _bg(rd)

    # gas-giant parent (far) first
    gp = _rs(ro - _PARENT, rd, _PARENT_R)
    if gp[0] > 0.0 and gp[0] < 1.0e29:
        npg = wp.normalize((ro - _PARENT) + rd * gp[0])
        col = _parent(npg, rd, sun)

    # ocean moon (near) over it
    g = _rs(ro, rd, _R)
    if g[0] > 0.0 and g[0] < 1.0e29:
        n = wp.normalize(ro + rd * g[0])
        col = _ocean(n, rd, sun)

    img[i, j] = col


def _render(width, height, time, mouse, device):
    az = 0.4 + float(mouse[0]) * 0.01 + time * 0.03
    el = 0.16
    dist = 4.2
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=42.0, aspect=width / height)
    sel = 0.25 + float(mouse[1]) * 0.004
    saz = az - 1.0
    sun = wp.vec3(math.cos(sel) * math.sin(saz), math.sin(sel), math.cos(sel) * math.cos(saz))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=1.3, strength=0.3, radius=max(2, int(width * 0.01)), passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.08)


SCENE = Scene(
    name="ocean_moon",
    description="A global-ocean moon — sun-glinted water, polar ice caps, thin "
                "clouds, atmosphere rim — with its banded gas-giant parent looming "
                "behind it over a starfield.",
    renderer=_render,
)
