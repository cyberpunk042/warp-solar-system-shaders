"""Nebula — emissive volumetric raymarch of a baked 3D noise volume.

Showcases the 3D texture system: a detailed density volume (fbm + ridged
filaments) is baked ONCE into a wp.array3d, then sampled per raymarch step with
trilinear `sample3d` (cheap) instead of recomputing noise. Emissive colour ramp +
Beer-Lambert extinction over a starfield. `--quality` scales the march steps.
iMouse orbits.
"""

import math

import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import fbm3, ridged3
from ..scene import Scene
from ..textures import sample3d

_R = 1.3
_vol_cache = {}


@wp.kernel
def bake_nebula(vol: wp.array3d(dtype=float), size: int):
    iz, iy, ix = wp.tid()
    p = wp.vec3(float(ix), float(iy), float(iz)) / float(size)
    base = fbm3(p * 3.0, 5)
    fil = ridged3(p * 2.2 + wp.vec3(4.0, 4.0, 4.0), 5)
    vol[iz, iy, ix] = base * 0.55 + fil * 0.55


@wp.func
def _box(ro: wp.vec3, rd: wp.vec3, r: float) -> wp.vec2:
    m = wp.vec3(1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2])
    n = wp.cw_mul(m, ro)
    k = wp.vec3(wp.abs(m[0]) * r, wp.abs(m[1]) * r, wp.abs(m[2]) * r)
    t1 = -n - k
    t2 = -n + k
    tn = wp.max(wp.max(t1[0], t1[1]), t1[2])
    tf = wp.min(wp.min(t2[0], t2[1]), t2[2])
    return wp.vec2(tn, tf)


@wp.func
def _palette(d: float, p: wp.vec3) -> wp.vec3:
    # teal/blue <-> magenta/pink, varied by position; warm cores where densest
    hue = 0.35 + 0.45 * wp.sin(p[0] * 2.3 + p[2] * 1.5 + p[1] * 0.8)
    c_cool = wp.vec3(0.10, 0.45, 0.95)
    c_warm = wp.vec3(1.0, 0.20, 0.5)
    base = c_cool * (1.0 - hue) + c_warm * hue
    hot = wp.vec3(1.0, 0.65, 0.3) * wp.pow(d, 4.0)
    return base * (d * d) + hot


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), vol: wp.array3d(dtype=float),
                  cam: Camera, steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    col = stars(rd)
    bb = _box(ro, rd, _R)
    if bb[1] > wp.max(bb[0], 0.0):
        t0 = wp.max(bb[0], 0.0)
        seg = (bb[1] - t0) / float(steps)
        trans = float(1.0)
        acc = wp.vec3(0.0, 0.0, 0.0)
        sigma = 2.4
        t = t0 + 0.5 * seg
        for _ in range(steps):
            p = ro + rd * t
            uvw = (p + wp.vec3(_R, _R, _R)) / (2.0 * _R)
            s = sample3d(vol, uvw[0], uvw[1], uvw[2], 0)
            fall = wp.smoothstep(_R, 0.2, wp.length(p))
            d = wp.clamp((s - 0.55) * 3.2, 0.0, 1.0) * fall
            if d > 0.01:
                emis = _palette(d, p)
                d_tr = wp.exp(-d * sigma * seg)
                acc += emis * (trans * (1.0 - d_tr))
                trans *= d_tr
                if trans < 0.02:
                    break
            t += seg
        col = col * trans + acc
    img[i, j] = col


def _steps(name):
    return {"low": 40, "medium": 64, "high": 96, "ultra": 160}.get(name, 64)


def _get_vol(device):
    key = (device, 80)
    if key not in _vol_cache:
        s = 80
        vol = wp.zeros((s, s, s), dtype=float, device=device)
        wp.launch(bake_nebula, dim=(s, s, s), inputs=[vol, s], device=device)
        wp.synchronize_device(device)
        _vol_cache[key] = vol
    return _vol_cache[key]


def _render(width, height, time, mouse, device):
    tier = active_tier()
    steps = _steps(tier.name)
    vol = _get_vol(device)

    az = 0.5 + float(mouse[0]) * 0.01 + time * 0.06
    el = 0.25 + float(mouse[1]) * 0.004
    dist = 3.4
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=44.0, aspect=width / height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, vol, cam, int(steps), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=0.8, strength=0.7, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.2)


SCENE = Scene(
    name="nebula",
    description="Emissive volumetric nebula from a baked 3D noise volume (sample3d). --quality low..ultra.",
    renderer=_render,
)
