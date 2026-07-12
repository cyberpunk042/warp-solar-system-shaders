"""City on a planet — an ecumenopolis curving to a planetary horizon.

Drops the buildings-city SDF (``buildings.city_de``, a whole skyline from one
domain-repeated function) onto the surface of a **large sphere** instead of a flat
plane: the ground now curves away to a real planetary **horizon**, with the thin
band of **atmosphere** and open **space** above it. A low sun rakes long shadows
down the avenues and half the windows glow. Cross-strand: the buildings strand
meets the cosmos strand. See ``docs/research/20-more-cosmos-worlds-crossstrand.md``.
"""

import math

import warp as wp

from ..buildings.sdf import city_de, window_mask
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.hash import hash21
from ..scene import Scene

_LOT = wp.constant(15.0)
_SEED = wp.constant(4.0)
_RP = wp.constant(520.0)                      # planet radius (centre at (0,-RP,0))


@wp.func
def _surf_h(p: wp.vec3) -> float:
    # height of p above the planet surface (sphere centred below the origin)
    c = wp.vec3(0.0, -_RP, 0.0)
    return wp.length(p - c) - _RP


@wp.func
def _local(p: wp.vec3) -> wp.vec3:
    # map a world point onto the sphere's tangent frame so buildings extrude
    # ALONG THE RADIAL (upright on the curved surface): arc-length x/z, radial y
    c = wp.vec3(0.0, -_RP, 0.0)
    r = p - c
    rad = wp.length(r)
    axis = r[1]                                # component along the pole (y+RP)
    lx = _RP * wp.atan2(r[0], axis)            # arc-length east/west
    lz = _RP * wp.atan2(r[2], axis)            # arc-length north/south
    return wp.vec3(lx, rad - _RP, lz)


@wp.func
def _map(p: wp.vec3) -> float:
    q = _local(p)
    b = city_de(q, _LOT, _SEED)[0]
    return wp.min(b, q[1])                     # union skyline with the curved ground


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.02
    dx = _map(p + wp.vec3(e, 0.0, 0.0)) - _map(p - wp.vec3(e, 0.0, 0.0))
    dy = _map(p + wp.vec3(0.0, e, 0.0)) - _map(p - wp.vec3(0.0, e, 0.0))
    dz = _map(p + wp.vec3(0.0, 0.0, e)) - _map(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _sky(rd: wp.vec3) -> wp.vec3:
    up = rd[1]
    # deep space above, dark
    space = wp.vec3(0.004, 0.006, 0.016)
    # a thin, bright atmospheric limb hugging the horizon (glows blue→warm)
    atmo = wp.exp(-wp.max(up, 0.0) * 16.0)
    band = wp.vec3(0.35, 0.55, 0.95) * atmo + wp.vec3(1.0, 0.6, 0.32) * (atmo * atmo * 0.5)
    s = hash21(wp.vec2(wp.floor(rd[0] * 200.0), wp.floor(rd[2] * 200.0)))
    star = wp.step(0.9955 - s) * wp.clamp(up * 4.0, 0.0, 1.0)   # wp.step(x)=1 when x<0
    return space + band * wp.clamp(up + 0.05, 0.0, 1.0) + wp.vec3(star, star, star)


@wp.func
def _shadow(p: wp.vec3, sun: wp.vec3, steps: int) -> float:
    res = float(1.0)
    t = float(0.3)
    for _ in range(steps):
        h = _map(p + sun * t)
        if h < 0.002:
            return 0.0
        res = wp.min(res, 12.0 * h / t)
        t += wp.clamp(h, 0.06, 3.5)
        if t > 140.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  march_steps: int, shadow_steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    t = float(0.0)
    hit = int(0)
    for _ in range(march_steps):
        p = ro + rd * t
        d = _map(p)
        if d < 0.002 * t + 0.004:
            hit = 1
            break
        t += d * 0.6                                  # small step: domain-rep field
        if t > 900.0:
            break

    col = _sky(rd)
    if hit == 1:
        p = ro + rd * t
        n = _normal(p)
        q = _local(p)
        v4 = city_de(q, _LOT, _SEED)
        is_ground = wp.step(v4[0] - 0.06)
        lr = v4[3]
        win = window_mask(q, 1.1, 1.6) * (1.0 - is_ground)
        fi = wp.floor(q[1] / 1.6)
        ci = wp.floor((q[0] + q[2]) / 1.1)
        lh = hash21(wp.vec2(ci + lr * 61.0, fi - lr * 37.0))
        lit = win * wp.step(lh - 0.5)
        # daytime materials (so the 3-D skyline + curvature read), warm sun
        concrete = wp.vec3(0.34, 0.34, 0.38)
        glassd = wp.vec3(0.18, 0.26, 0.34)
        street = wp.vec3(0.12, 0.12, 0.14)
        mat = concrete * (1.0 - win) + glassd * win
        mat = mat * (1.0 - is_ground) + street * is_ground
        ndl = wp.max(wp.dot(n, sun), 0.0)
        sh = _shadow(p + n * 0.05, sun, shadow_steps)
        sky_amb = wp.vec3(0.30, 0.40, 0.62)
        col_s = wp.cw_mul(mat, wp.vec3(1.0, 0.9, 0.72)) * (ndl * sh * 1.5)
        col_s = col_s + wp.cw_mul(mat, sky_amb) * 0.35
        # lit windows still glow on the shadowed faces
        wc = wp.vec3(1.0, 0.8, 0.46)
        emit = wc * (lit * (1.0 - ndl * sh) * 1.4)
        col = col_s + emit
        # aerial haze toward the horizon hides far warp/flicker + reads as air
        haze = wp.smoothstep(90.0, 300.0, t)
        atmo_haze = wp.vec3(0.14, 0.2, 0.34)
        col = col * (1.0 - haze) + atmo_haze * haze

    img[i, j] = col


def _counts(name):
    return {"low": (180, 18), "medium": (280, 26), "high": (420, 40),
            "ultra": (600, 60)}.get(name, (280, 26))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss = _counts(tier.name)
    az = 0.5 + time * 0.02 + float(mouse[0]) * 0.01
    # camera high above the rooftops (clear of the ~30-tall towers), looking
    # down toward the curved horizon — an aerial ecumenopolis
    h_cam = 42.0 + float(mouse[1]) * 0.2
    eye = (math.sin(az) * 8.0, h_cam, math.cos(az) * 8.0)
    tgt = (math.sin(az) * 130.0, 20.0, math.cos(az) * 130.0 + 210.0)
    cam = make_camera(eye, tgt, fov_deg=60.0, aspect=width / height)
    sun = wp.normalize(wp.vec3(math.cos(az + 2.0) * 0.7, 0.26, math.sin(az + 2.0) * 0.7))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(ms), int(ss), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.15, strength=0.3, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.02)


SCENE = Scene(
    name="city_planet",
    description="An ecumenopolis curving to a planetary horizon — the buildings "
                "city SDF wrapped onto a large sphere, with atmosphere and space "
                "above and a low sun raking long shadows.",
    renderer=_render,
)
