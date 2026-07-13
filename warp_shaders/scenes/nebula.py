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

# Hot young stars embedded in the cloud — they ionize the gas around them
# (that is *why* an emission nebula glows). Positions + radiant colours.
_S0 = wp.constant(wp.vec3(0.34, 0.10, -0.22))
_S1 = wp.constant(wp.vec3(-0.42, -0.16, 0.26))
_S2 = wp.constant(wp.vec3(0.02, 0.34, 0.12))
_C0 = wp.constant(wp.vec3(0.55, 0.72, 1.0))     # blue O-star
_C1 = wp.constant(wp.vec3(1.0, 0.72, 0.55))     # warmer B-star
_C2 = wp.constant(wp.vec3(0.72, 0.86, 1.0))     # blue-white


@wp.kernel
def bake_nebula(vol: wp.array3d(dtype=float), size: int):
    iz, iy, ix = wp.tid()
    p = wp.vec3(float(ix), float(iy), float(iz)) / float(size)
    # filaments dominate (ridged), warped by a low-frequency flow → tendrils
    warp = fbm3(p * 1.5 + wp.vec3(9.0, 2.0, 5.0), 4)
    fil = ridged3(p * 3.0 + wp.vec3(4.0 + warp * 0.6, 4.0 - warp * 0.5, 4.0 + warp * 0.4), 6)
    base = fbm3(p * 3.6, 6)
    vol[iz, iy, ix] = fil * 0.7 + base * 0.34


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
def _starlight(p: wp.vec3) -> wp.vec3:
    """Radiance reaching a gas point from the embedded ionizing stars."""
    d0 = p - _S0
    d1 = p - _S1
    d2 = p - _S2
    l0 = 1.0 / (wp.dot(d0, d0) + 0.03)
    l1 = 1.0 / (wp.dot(d1, d1) + 0.03)
    l2 = 1.0 / (wp.dot(d2, d2) + 0.03)
    return _C0 * (l0 * 0.05) + _C1 * (l1 * 0.045) + _C2 * (l2 * 0.04)


@wp.func
def _star_core(p: wp.vec3) -> wp.vec3:
    """The bright stars themselves (sharp cores)."""
    d0 = p - _S0
    d1 = p - _S1
    d2 = p - _S2
    c0 = wp.exp(-wp.dot(d0, d0) / 0.0007)
    c1 = wp.exp(-wp.dot(d1, d1) / 0.0007)
    c2 = wp.exp(-wp.dot(d2, d2) / 0.0007)
    return _C0 * (c0 * 3.0) + _C1 * (c1 * 3.0) + _C2 * (c2 * 3.0)


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
        t = t0 + 0.5 * seg
        for _ in range(steps):
            p = ro + rd * t
            uvw = (p + wp.vec3(_R, _R, _R)) / (2.0 * _R)
            s = sample3d(vol, uvw[0], uvw[1], uvw[2], 0)
            env = wp.exp(-wp.dot(p, p) * 0.85)               # soft envelope, fills the box
            d = wp.clamp((s - 0.5) * 2.3, 0.0, 1.0) * env
            # bright stars are always visible (attenuated by foreground dust)
            acc += _star_core(p) * (trans * seg * 6.0)
            if d > 0.01:
                light = _starlight(p)
                lum = light[0] + light[1] + light[2]
                # ionization colour: H-alpha pink near hot stars, OIII teal in
                # the diffuse outskirts; densest cores redden (self-absorbed)
                ion = wp.clamp(lum * 0.7, 0.0, 1.0)
                gas = wp.vec3(0.22, 0.52, 0.55) * (1.0 - ion) \
                    + wp.vec3(0.95, 0.28, 0.42) * ion
                emis = gas * (d * (0.25 + 1.6 * lum))
                # dense filaments are dusty: they absorb strongly → dark pillars
                sigma = 2.0 + 6.0 * d
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

    ss = 2
    W, H = int(width) * ss, int(height) * ss
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=44.0, aspect=W / H)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(H, W),
              inputs=[img, vol, cam, int(steps), int(W), int(H)],
              device=device)
    wp.synchronize_device(device)
    hdr = post.downsample(img.numpy(), ss)
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.55, radius=r, passes=3, octaves=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)


SCENE = Scene(
    name="nebula",
    description="Emissive volumetric nebula from a baked 3D noise volume (sample3d). --quality low..ultra.",
    renderer=_render,
)
