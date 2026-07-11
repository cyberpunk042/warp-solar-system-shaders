"""Volcanic world — a basalt heightfield flooded with molten lava.

The surface is `max(rock, lava_level)`: sharp ridged-fbm basalt plates rise out
of a near-flat molten sheet, so lava pools in every valley. Lava is emissive —
a deep orange incandescence broken by drifting cooled-crust noise (domain-warped
flow), brightest where it's deepest — and the basalt shorelines glow with
conducted heat. Dark ashen sky, dim red sun, warm heat-haze. Emission drives the
bloom. Heightfield raymarch with crossing-detection + bisection; `--quality`
scales march/shadow steps. iMouse pans.
"""

import math

import warp as wp

from ..engine import post
from ..engine.pbr import shade_pbr
from ..engine.shading import apply_fog, sky_gradient, sun_disk
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import domain_warp3, fbm3, ridged3
from ..scene import Scene

_FAR = 300.0
_LAVA = 0.0           # molten sheet level (the dominant surface)


@wp.func
def _rock(x: float, z: float) -> float:
    # basalt rises ABOVE the molten plain only at ridge crests -> sparse islands,
    # so most of the frame is lava (below this the surface is the molten sheet).
    p = wp.vec3(x * 0.02, 0.0, z * 0.02)
    plates = (ridged3(p, 6) - 0.52) * 17.0
    detail = fbm3(p * 2.3, 5) * 1.2
    return plates + detail


@wp.func
def _lava_level(x: float, z: float, time: float) -> float:
    # near-flat molten surface with slow advecting ripples
    q = wp.vec3(x * 0.05 + time * 0.15, time * 0.2, z * 0.05)
    return _LAVA + fbm3(q, 3) * 0.45


@wp.func
def _height(x: float, z: float, time: float) -> float:
    return wp.max(_rock(x, z), _lava_level(x, z, time))


@wp.func
def _normal(x: float, z: float, time: float) -> wp.vec3:
    e = 0.09
    nx = _height(x - e, z, time) - _height(x + e, z, time)
    nz = _height(x, z - e, time) - _height(x, z + e, time)
    return wp.normalize(wp.vec3(nx, 2.0 * e, nz))


@wp.func
def _lava_emission(x: float, z: float, time: float, shore: float) -> wp.vec3:
    # crust: domain-warped flow breaks the molten surface into cooled black rafts
    flow = domain_warp3(wp.vec3(x * 0.06 + time * 0.25, z * 0.06, time * 0.1), 4, 1.0)
    crust = wp.smoothstep(0.32, 0.52, flow)          # 1 = cooled black raft
    # finer skin wrinkles break up the sheet even inside the molten channels
    skin = fbm3(wp.vec3(x * 0.22, z * 0.22, time * 0.3), 4)
    crust = wp.clamp(crust + wp.smoothstep(0.55, 0.72, skin) * 0.7, 0.0, 1.0)
    # cooled crust rafts thin near the shorelines (churned hot lava next to rock)
    crust = wp.clamp(crust - shore * 0.5, 0.0, 1.0)
    # incandescence: mixed-scale temperature field -> hot rivers + cooler expanses
    temp = (fbm3(wp.vec3(x * 0.016 + time * 0.05, z * 0.016, time * 0.03), 4) * 0.6
            + fbm3(wp.vec3(x * 0.06, z * 0.06, time * 0.08), 3) * 0.5)
    heat = wp.clamp(temp * 1.5 - 0.25, 0.0, 1.0)
    hot = wp.vec3(1.0, 0.35, 0.05) * (1.0 - heat) + wp.vec3(1.0, 0.82, 0.34) * heat
    glow = hot * (0.9 + 1.7 * heat)
    raft = wp.vec3(0.028, 0.012, 0.008)              # cooled crust, near-black
    # glowing cracks still show hot lava between the rafts
    cracks = wp.smoothstep(0.55, 0.67, flow)
    return glow * (1.0 - crust) + (raft + hot * (0.8 * cracks)) * crust


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    base = sky_gradient(rd, wp.vec3(0.16, 0.09, 0.08), wp.vec3(0.05, 0.04, 0.06))
    return base + sun_disk(rd, sun, wp.vec3(1.0, 0.4, 0.2), 0.9992, 0.5)


@wp.func
def _shadow(p: wp.vec3, sun: wp.vec3, time: float, steps: int) -> float:
    res = float(1.0)
    t = float(0.5)
    for _ in range(steps):
        q = p + sun * t
        h = q[1] - _height(q[0], q[2], time)
        if h < 0.001:
            return 0.0
        res = wp.min(res, 11.0 * h / t)
        t += wp.clamp(h, 0.4, 7.0)
        if t > 110.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  time: float, march_steps: int, shadow_steps: int,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    t = float(1.0)
    prev_t = t
    hit = int(0)
    for _ in range(march_steps):
        p = ro + rd * t
        d = p[1] - _height(p[0], p[2], time)
        if d < 0.0:
            hit = 1
            break
        prev_t = t
        t += wp.max(d * 0.4, 0.01 * t)
        if t > _FAR:
            break

    if hit == 0:
        img[i, j] = _sky(rd, sun)
        return

    a = prev_t
    b = t
    for _ in range(6):
        m = 0.5 * (a + b)
        pm = ro + rd * m
        if pm[1] - _height(pm[0], pm[2], time) < 0.0:
            b = m
        else:
            a = m
    t = 0.5 * (a + b)
    p = ro + rd * t

    rockh = _rock(p[0], p[2])
    lavah = _lava_level(p[0], p[2], time)

    if lavah >= rockh:
        # molten surface — emissive, barely lit. `shore` ~1 right next to rock.
        shore = wp.exp(-(lavah - rockh) * 0.7)
        col = _lava_emission(p[0], p[2], time, shore)
    else:
        n = _normal(p[0], p[2], time)
        v_dir = -rd
        sh = _shadow(p + n * 0.05, sun, time, shadow_steps)
        # dark basalt, slightly rough; cooler up high
        basalt = wp.vec3(0.05, 0.045, 0.05)
        direct = shade_pbr(n, v_dir, sun, basalt, 0.8, 0.0,
                           wp.vec3(1.0, 0.6, 0.4)) * (1.8 * sh)
        amb = wp.cw_mul(_sky(n, sun), basalt) * 0.6
        # conducted heat: basalt near the lava shoreline glows
        shore = wp.exp(-(rockh - lavah) * 1.6)
        heat = wp.vec3(1.0, 0.4, 0.12) * (shore * 1.8)
        col = direct + amb + heat

    col = apply_fog(col, t, wp.vec3(0.16, 0.06, 0.05), 0.0045)
    img[i, j] = col


def _counts(name):
    return {"low": (120, 14), "medium": (200, 22), "high": (300, 30),
            "ultra": (440, 44)}.get(name, (200, 22))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss = _counts(tier.name)

    az = 0.7 + float(mouse[0]) * 0.008 + time * 0.02
    eye = (math.sin(az) * 7.0, 6.0, math.cos(az) * 7.0)
    target = (eye[0] + math.sin(az) * 10.0, 2.8 + float(mouse[1]) * 0.02,
              eye[2] + math.cos(az) * 10.0)
    cam = make_camera(eye, target, fov_deg=60.0, aspect=width / height)

    el = 0.18
    sun = wp.vec3(math.sin(az + 1.6) * math.cos(el), math.sin(el),
                  math.cos(az + 1.6) * math.cos(el))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, float(time), int(ms), int(ss),
                      int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.3, strength=0.35, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=0.9)


SCENE = Scene(
    name="lava_planet",
    description="Volcanic world: basalt heightfield flooded with emissive molten lava. --quality low..ultra.",
    renderer=_render,
)
