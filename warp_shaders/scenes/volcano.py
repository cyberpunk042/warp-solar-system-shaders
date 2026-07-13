"""A volcano erupting at night — lava fountain, glowing flows, ash plume.

A ground-level night view of a stratovolcano in eruption: a dark basalt cone
whose flanks are cut by **incandescent lava flows** (white-hot at the vents,
cooling to deep red downslope), a **lava fountain** bursting from the summit
crater with a shower of glowing bombs and embers, and a towering **ash plume**
lit orange from beneath by the eruption. Stars behind, the plume's underside
glowing. --frames animates the churn.

Composes the engine's heightfield raymarcher with emissive lava + a volumetric
plume — a new subject, no existing scene touched.
"""

import math

import warp as wp

from ..engine import post
from ..engine.sky import starfield
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3, ridged3
from ..scene import Scene

_FAR = 120.0
_SUMMIT_F = 9.0                     # cone peak height (host)
_SUMMIT = wp.constant(_SUMMIT_F)    # same, for kernels


@wp.func
def _height(x: float, z: float) -> float:
    r = wp.sqrt(x * x + z * z)
    cone = _SUMMIT - 0.62 * r                        # conical flanks
    crater = -2.2 * wp.exp(-(r / 1.6) * (r / 1.6))   # summit crater dip
    rough = ridged3(wp.vec3(x * 0.09, 0.0, z * 0.09), 5) * 2.4 * wp.smoothstep(0.0, 6.0, r)
    return cone + crater + rough - 3.5


@wp.func
def _lava(p: wp.vec3, time: float) -> wp.vec3:
    """Emissive lava on the cone surface at hit point p."""
    r = wp.length(wp.vec2(p[0], p[2]))
    ang = wp.atan2(p[2], p[0])
    below = wp.max(_SUMMIT - 3.5 - p[1], 0.0)         # how far below the summit
    flow_n = fbm3(wp.vec3(ang * 3.0, p[1] * 0.35 - time * 0.6, 0.0), 4)
    # narrow incandescent channels running downslope
    chan = wp.pow(0.5 + 0.5 * wp.sin(ang * 9.0 + flow_n * 4.0), 10.0)
    hot = wp.exp(-below * 0.5)                         # white-hot high, cooling low
    lava = chan * (0.35 + 0.65 * flow_n) * wp.smoothstep(0.0, 0.4, hot + 0.15)
    # crater pool glow
    pool = wp.exp(-(r / 1.7) * (r / 1.7)) * wp.smoothstep(-1.0, 1.5, p[1] - (_SUMMIT - 5.5))
    glow = wp.clamp(lava + pool * 1.2, 0.0, 3.0)
    cool = wp.vec3(0.7, 0.06, 0.02)
    warm = wp.vec3(1.0, 0.85, 0.35)
    col = cool * (1.0 - hot) + warm * hot
    return col * glow * 2.2


@wp.func
def _plume(ro: wp.vec3, rd: wp.vec3, time: float) -> wp.vec4:
    """Ash plume + fountain glow above the crater. Returns (rgb, alpha)."""
    # march a vertical cylinder region above the summit
    top = 60.0
    t0 = float(2.0)
    seg = 1.1
    acc = wp.vec3(0.0, 0.0, 0.0)
    trans = float(1.0)
    t = t0
    for _ in range(48):
        p = ro + rd * t
        if p[1] > _SUMMIT - 5.0 and p[1] < top:
            hpl = (p[1] - (_SUMMIT - 5.0)) / (top - (_SUMMIT - 5.0))
            drift = 7.0 * hpl * hpl                     # leans downwind as it rises
            rr = wp.length(wp.vec2(p[0] - drift, p[2] + drift * 0.4))
            width = 2.6 + 8.5 * hpl                     # plume widens with height
            billow = fbm3(wp.vec3(p[0] * 0.18, p[1] * 0.12 - time * 0.5, p[2] * 0.18), 6)
            d = wp.smoothstep(width, width * 0.55, rr) * wp.clamp(billow + 0.35, 0.0, 1.0)
            d = d * (1.0 - wp.smoothstep(0.75, 1.0, hpl))
            if d > 0.01:
                underlit = wp.exp(-hpl * 3.0)          # orange glow at the base only
                smoke = wp.vec3(0.09, 0.08, 0.09) + wp.vec3(1.0, 0.42, 0.14) * underlit * 1.5
                dd = d * 0.22
                acc = acc + smoke * (dd * trans)
                trans = trans * (1.0 - wp.clamp(dd, 0.0, 1.0))
        t = t + seg
    return wp.vec4(acc[0], acc[1], acc[2], 1.0 - trans)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                  march_steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    t = float(0.5)
    hit = int(0)
    for _ in range(march_steps):
        p = ro + rd * t
        d = p[1] - _height(p[0], p[2])
        if d < 0.0:
            hit = 1
            break
        t += wp.max(d * 0.4, 0.01 * t)
        if t > _FAR:
            break

    if hit == 1:
        a = t - 0.5
        b = t
        for _ in range(6):
            m = 0.5 * (a + b)
            pm = ro + rd * m
            if pm[1] - _height(pm[0], pm[2]) < 0.0:
                b = m
            else:
                a = m
        t = 0.5 * (a + b)
        p = ro + rd * t
        lav = _lava(p, time)
        # dark basalt, faintly lit by the eruption glow
        basalt = wp.vec3(0.03, 0.025, 0.03)
        col = basalt + lav
    else:
        up = wp.clamp(rd[1] * 0.9 + 0.1, 0.0, 1.0)
        col = wp.vec3(0.02, 0.03, 0.06) * (1.0 - up) + wp.vec3(0.01, 0.01, 0.03) * up
        col = col + starfield(rd)
        # lava fountain: a bright emissive burst just above the crater axis
        # (project the summit point; brighten rays passing near it)
        toS = wp.vec3(0.0, _SUMMIT - 2.5, 0.0) - ro
        proj = wp.dot(toS, rd)
        if proj > 0.0:
            closest = wp.length(wp.cross(toS, rd))     # dist from summit to ray
            fount = wp.exp(-(closest * closest) / 3.0)
            spark = wp.pow(wp.max(fbm3(wp.vec3(rd[0] * 40.0, rd[1] * 40.0, time), 2), 0.0), 3.0)
            col = col + wp.vec3(1.0, 0.6, 0.2) * (fount * 1.6) + wp.vec3(1.0, 0.8, 0.4) * (fount * spark * 3.0)

    pl = _plume(ro, rd, time)
    col = col * (1.0 - pl[3]) + wp.vec3(pl[0], pl[1], pl[2])
    img[i, j] = col


def _render(width, height, time, mouse, device):
    az = 0.5 + float(mouse[0]) * 0.006 + time * 0.01
    pitch = 0.16 + float(mouse[1]) * 0.003
    dist = 34.0
    eye = (math.sin(az) * dist, 6.0, math.cos(az) * dist)
    target = (0.0, _SUMMIT_F - 1.0, 0.0)
    ss = 2
    W, H = int(width) * ss, int(height) * ss
    cam = make_camera(eye, target, fov_deg=52.0, aspect=W / H)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(H, W),
              inputs=[img, cam, float(time), int(200), int(W), int(H)], device=device)
    wp.synchronize_device(device)
    hdr = post.downsample(img.numpy(), ss)
    r = max(3, int(min(width, height) * 0.016))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.55, radius=r, passes=3, octaves=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)


SCENE = Scene(
    name="volcano",
    description="A stratovolcano erupting at night — a dark cone cut by incandescent "
                "lava flows, a lava fountain of glowing bombs bursting from the crater, "
                "and an ash plume lit orange from below, over a starfield. --frames "
                "animates the churn.",
    renderer=_render,
)
