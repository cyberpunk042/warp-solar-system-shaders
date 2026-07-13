"""Galaxy — a volumetric emissive spiral disk.

Raymarches a thin galactic disk: log-spiral arms (blue-white), a warm core bulge,
and pink star-forming knots, with a vertical gaussian profile and radial falloff.
Emissive accumulation with mild dust extinction, over a starfield, viewed at an
inclination. `--quality` scales the march steps. iMouse orbits.
"""

import math

import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import fbm3
from ..scene import Scene

_RD = 2.6
_TH = 0.16


@wp.func
def _slab(ro: wp.vec3, rd: wp.vec3) -> wp.vec2:
    if wp.abs(rd[1]) < 1e-5:
        return wp.vec2(1.0, -1.0)
    t0 = (-_TH - ro[1]) / rd[1]
    t1 = (_TH - ro[1]) / rd[1]
    return wp.vec2(wp.min(t0, t1), wp.max(t0, t1))


@wp.func
def _field(p: wp.vec3, time: float) -> wp.vec4:
    """Return (emission.rgb, dust_density) at a disk point."""
    r = wp.length(wp.vec2(p[0], p[2]))
    if r > _RD:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)
    th = wp.atan2(p[2], p[0]) + time * 0.05
    lr = wp.log(r + 0.22)
    turb = fbm3(p * 2.6, 4)
    fine = fbm3(p * 7.5 + wp.vec3(11.0, 3.0, 7.0), 3)
    spark = fbm3(p * 22.0 + wp.vec3(4.0, 9.0, 2.0), 2)   # star-cluster granularity

    vert = wp.exp(-(p[1] * p[1]) / 0.008)
    rad = wp.exp(-r * 1.1)

    # two-arm log spiral, tightened into sharp density-wave ridges
    s = 2.0 * th + 6.5 * lr
    arms = 0.5 + 0.5 * wp.sin(s + turb * 1.7)
    arm = wp.pow(arms, 3.4) * rad * vert

    # young blue-white stars, resolved into clumps + bright sparkle knots
    clump = 0.4 + 1.1 * wp.smoothstep(0.35, 0.9, fine)
    sparkle = wp.smoothstep(0.72, 0.95, spark)
    arm_col = wp.vec3(0.55, 0.68, 1.0) * (arm * 2.2 * clump) \
        + wp.vec3(0.8, 0.9, 1.0) * (arm * sparkle * 3.0)
    # discrete pink/red HII star-forming knots strung along the arms
    hii = wp.smoothstep(0.68, 0.9, fine) * wp.smoothstep(0.55, 0.85, spark) * arm
    hii_col = wp.vec3(1.0, 0.28, 0.42) * (hii * 12.0)

    # bulge: a broad warm glow plus a sharp brilliant nucleus
    bulge = wp.exp(-r * r * 6.5)
    nucleus = wp.exp(-r * r * 90.0)
    core_col = wp.vec3(1.0, 0.85, 0.55) * (bulge * 2.4) \
        + wp.vec3(1.0, 0.96, 0.82) * (nucleus * 8.0)

    emit = arm_col + hii_col + core_col

    # dust lanes: an offset spiral that ABSORBS (no emission) — the dark
    # threads on the leading edge of each arm that make a spiral read
    dusts = 0.5 + 0.5 * wp.sin(s - 0.6 + turb * 1.7)
    dust = wp.pow(dusts, 4.0) * rad * vert * (0.6 + 0.7 * turb) \
        * wp.smoothstep(0.12, 0.5, r)
    return wp.vec4(emit[0], emit[1], emit[2], dust)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                  steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    col = stars(rd)
    # a faint spherical halo of old stars around the bulge
    d_core = wp.length(ro + rd * wp.max(wp.dot(-ro, rd), 0.0))
    col = col + wp.vec3(0.9, 0.82, 0.7) * (0.06 * wp.exp(-d_core * d_core * 0.5))
    sb = _slab(ro, rd)
    t0 = wp.max(sb[0], 0.0)
    if sb[1] > t0:
        seg = (sb[1] - t0) / float(steps)
        acc = wp.vec3(0.0, 0.0, 0.0)
        trans = float(1.0)
        t = t0 + 0.5 * seg
        for _ in range(steps):
            p = ro + rd * t
            f = _field(p, time)
            e = wp.vec3(f[0], f[1], f[2])
            edens = (e[0] + e[1] + e[2]) * 0.15
            acc += e * (seg * 1.5 * trans)
            # dust lanes absorb the light from stars + gas behind them
            trans *= wp.exp(-(f[3] * 3.2 + edens * 0.4) * seg)
            if trans < 0.04:
                break
            t += seg
        col = col * trans + acc
    img[i, j] = col


def _steps(name):
    return {"low": 40, "medium": 64, "high": 100, "ultra": 160}.get(name, 64)


def _render(width, height, time, mouse, device):
    tier = active_tier()
    steps = _steps(tier.name)

    az = 0.4 + float(mouse[0]) * 0.01 + time * 0.03
    el = 0.5 + float(mouse[1]) * 0.004      # inclination
    dist = 6.5
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    ss = 2                                             # SSAA for crisp arm stars
    W, H = int(width) * ss, int(height) * ss
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=40.0, aspect=W / H)

    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(H, W),
              inputs=[img, cam, float(time), int(steps), int(W), int(H)],
              device=device)
    wp.synchronize_device(device)
    hdr = post.downsample(img.numpy(), ss)
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.5, radius=r, passes=3, octaves=3)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="galaxy",
    description="Volumetric spiral galaxy: log-spiral arms, core bulge, star-forming knots. --quality low..ultra.",
    renderer=_render,
)
