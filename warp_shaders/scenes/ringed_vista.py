"""Ringed giant over an alien desert — a ground-level planetary vista.

A cinematic *surface* shot, not another view from space: the camera stands on a
dune sea of a moon and looks toward the horizon, where a huge **ringed gas giant**
hangs low in a dusk-violet sky. The giant is a real shaded sphere (latitude
banding + a day/night terminator lit by a small hard sun) wearing a tilted
Saturn-like **ring system** that passes *behind* its upper limb and *in front of*
its lower limb, casting the classic ellipse. The dunes are a ridged+fbm
heightfield with IQ soft shadows, lit warm by the sun and filled cool by the
giant, fading to the horizon through aerial perspective.

Composes the engine's heightfield raymarcher (cf. ``terrain``) with a procedural
sky body — no globe-from-space, no existing scene touched. iMouse pans the view.
"""

import math

import warp as wp

from ..engine import post
from ..engine.pbr import shade_pbr
from ..engine.shading import apply_fog
from ..engine.sky import starfield
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import fbm_perlin3, ridged3
from ..scene import Scene

_FAR = 340.0

# Giant + ring geometry (directions/angles are constants the kernel reads).
_PLANET_ANG = wp.constant(0.30)      # angular radius of the giant (radians) — big
_RING_IN = wp.constant(1.32)         # ring inner edge (× planet angular radius)
_RING_OUT = wp.constant(2.30)        # ring outer edge
_RING_SQUASH = wp.constant(0.34)     # vertical compression → tilted ellipse


@wp.func
def _height(x: float, z: float) -> float:
    p = wp.vec3(x * 0.019, 0.0, z * 0.019)
    dunes = fbm_perlin3(p, 6) * 5.0
    mesas = ridged3(p * 0.42, 6) * 15.0
    return dunes + mesas - 5.0


@wp.func
def _normal(x: float, z: float) -> wp.vec3:
    e = 0.08
    nx = _height(x - e, z) - _height(x + e, z)
    nz = _height(x, z - e) - _height(x, z + e)
    return wp.normalize(wp.vec3(nx, 2.0 * e, nz))


@wp.func
def _giant(rd: wp.vec3, pdir: wp.vec3, sun: wp.vec3) -> wp.vec4:
    """Shade the ringed giant along ray ``rd``. Returns (rgb, coverage[0..1])."""
    fwd = pdir
    right = wp.normalize(wp.cross(wp.vec3(0.0, 1.0, 0.0), fwd))
    upv = wp.cross(fwd, right)

    x = wp.dot(rd, right)
    y = wp.dot(rd, upv)
    front = wp.dot(rd, fwd)
    if front < 0.0:                                  # body is behind the camera
        return wp.vec4(0.0, 0.0, 0.0, 0.0)

    pr = wp.sin(_PLANET_ANG)                          # disk radius in tangent space
    disk = wp.sqrt(x * x + y * y)

    col = wp.vec3(0.0, 0.0, 0.0)
    cov = float(0.0)

    # --- ring system (tilted ellipse): draw the far (upper) half first so the
    #     planet can occlude it, then the planet, then the near (lower) half. ---
    ry = y / _RING_SQUASH
    rr = wp.sqrt(x * x + ry * ry) / pr               # ring radius in planet-radii
    in_ring = float(0.0)
    if rr > _RING_IN and rr < _RING_OUT:
        band = wp.smoothstep(_RING_IN, _RING_IN + 0.05, rr) \
            * wp.smoothstep(_RING_OUT, _RING_OUT - 0.05, rr)
        # Cassini-style gap + fine ringlets
        t = (rr - _RING_IN) / (_RING_OUT - _RING_IN)
        gap = wp.smoothstep(0.46, 0.5, wp.abs(t - 0.5)) * 0.0 + 1.0
        gap = 1.0 - 0.9 * wp.exp(-((t - 0.45) * (t - 0.45)) / 0.0006)   # Cassini division
        ringlets = 0.68 + 0.32 * wp.sin(rr * 120.0)
        in_ring = band * gap * ringlets
    ring_col = wp.vec3(0.66, 0.58, 0.45)

    # far ring half (behind planet's upper limb): y >= 0
    if in_ring > 0.0 and y >= 0.0:
        occl = wp.smoothstep(pr, pr * 1.02, disk)     # hidden where it's behind disk
        a = in_ring * occl * 0.9
        col = col * (1.0 - a) + ring_col * a
        cov = wp.max(cov, a)

    # --- the planet disk ---
    if disk < pr:
        # local sphere normal at this screen point
        zc = wp.sqrt(wp.max(pr * pr - disk * disk, 0.0)) / pr
        n = wp.normalize(right * (x / pr) + upv * (y / pr) + fwd * zc)
        lat = y / pr
        # banded gas-giant albedo (ammonia creams + ochre belts)
        b = wp.sin(lat * 9.0 + wp.sin(lat * 3.0) * 1.5)
        cream = wp.vec3(0.86, 0.80, 0.66)
        belt = wp.vec3(0.62, 0.44, 0.28)
        base = cream * (0.5 + 0.5 * b) + belt * (0.5 - 0.5 * b)
        # a great-spot storm
        sx = x / pr - 0.30
        sy = lat + 0.18
        spot = wp.exp(-((sx * sx) / 0.016 + (sy * sy) / 0.006))
        base = base * (1.0 - spot) + wp.vec3(0.74, 0.34, 0.24) * spot
        ndl = wp.max(wp.dot(n, sun), 0.0)
        lit = base * (0.05 + 0.95 * ndl)
        # thin bright limb (forward-scattered haze at the terminator edge)
        limb = wp.pow(1.0 - zc, 3.0) * ndl
        lit = lit + wp.vec3(0.9, 0.8, 0.6) * limb * 0.4
        col = lit
        cov = 1.0

    # near ring half (in front of the planet's lower limb): y < 0
    if in_ring > 0.0 and y < 0.0:
        # ring casts a soft shadow band on the planet where it crosses the disk
        a = in_ring * 0.9
        col = col * (1.0 - a) + ring_col * a
        cov = wp.max(cov, a)

    return wp.vec4(col[0], col[1], col[2], cov)


@wp.func
def _sky(rd: wp.vec3, pdir: wp.vec3, sun: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.8 + 0.2, 0.0, 1.0)
    # dusk-violet zenith fading to a warm dusty horizon
    base = wp.vec3(0.42, 0.30, 0.34) * (1.0 - up) + wp.vec3(0.10, 0.09, 0.22) * up
    # small hard sun
    s = wp.max(wp.dot(rd, sun), 0.0)
    base = base + wp.vec3(1.0, 0.7, 0.4) * (wp.pow(s, 40.0) * 0.6 + wp.pow(s, 4000.0) * 40.0)
    # faint stars up high
    base = base + starfield(rd) * up * 0.6
    # the ringed giant
    g = _giant(rd, pdir, sun)
    base = base * (1.0 - g[3]) + wp.vec3(g[0], g[1], g[2])
    return base


@wp.func
def _shadow(p: wp.vec3, sun: wp.vec3, steps: int) -> float:
    res = float(1.0)
    t = float(0.6)
    for _ in range(steps):
        q = p + sun * t
        h = q[1] - _height(q[0], q[2])
        if h < 0.001:
            return 0.0
        res = wp.min(res, 11.0 * h / t)
        t += wp.clamp(h, 0.5, 9.0)
        if t > 130.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  pdir: wp.vec3, march_steps: int, shadow_steps: int,
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
        d = p[1] - _height(p[0], p[2])
        if d < 0.0:
            hit = 1
            break
        prev_t = t
        t += wp.max(d * 0.4, 0.01 * t)
        if t > _FAR:
            break

    if hit == 0:
        img[i, j] = _sky(rd, pdir, sun)
        return

    a = prev_t
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
    n = _normal(p[0], p[2])
    slope = n[1]
    hnorm = wp.clamp((p[1] + 5.0) / 26.0, 0.0, 1.0)

    # alien desert: rust sand, dark rock on steep faces, pale crest dust
    sand = wp.vec3(0.42, 0.22, 0.12)
    rock = wp.vec3(0.20, 0.13, 0.11)
    dust = wp.vec3(0.55, 0.42, 0.34)
    rocky = wp.smoothstep(0.72, 0.5, slope)
    albedo = sand * (1.0 - rocky) + rock * rocky
    crest = wp.smoothstep(0.6, 0.85, hnorm) * wp.smoothstep(0.4, 0.7, slope)
    albedo = albedo * (1.0 - crest) + dust * crest

    v_dir = -rd
    sh = _shadow(p + n * 0.05, sun, shadow_steps)
    direct = shade_pbr(n, v_dir, sun, albedo, 0.9, 0.0, wp.vec3(1.0, 0.82, 0.6)) * (2.7 * sh)
    # cool planet-lit fill from the giant's direction + sky ambient
    pfill = wp.max(wp.dot(n, pdir), 0.0)
    fill = wp.cw_mul(albedo, wp.vec3(0.16, 0.13, 0.22)) * (0.6 + 0.9 * pfill)
    col = direct + fill

    col = apply_fog(col, t, _sky(rd, pdir, sun), 0.007)
    img[i, j] = col


def _counts(name):
    return {"low": (110, 14), "medium": (190, 22), "high": (300, 32),
            "ultra": (440, 48)}.get(name, (190, 22))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss = _counts(tier.name)

    az = 0.2 + float(mouse[0]) * 0.006 + time * 0.015
    eye = (math.sin(az) * 4.0, 7.5, math.cos(az) * 4.0)
    target = (eye[0] + math.sin(az) * 10.0, 8.4 + float(mouse[1]) * 0.02,
              eye[2] + math.cos(az) * 10.0)
    cam = make_camera(eye, target, fov_deg=64.0, aspect=width / height)

    # sun low and to the side (rim-lights the dunes + gives the giant a terminator);
    # same side as the giant's lit limb so the light reads consistently
    sel = 0.12
    saz = az + 1.15
    sun = wp.vec3(math.sin(saz) * math.cos(sel), math.sin(sel), math.cos(saz) * math.cos(sel))
    # the giant sits ahead, low over the horizon
    pel = 0.30
    pdir = wp.vec3(math.sin(az) * math.cos(pel), math.sin(pel), math.cos(az) * math.cos(pel))

    ss_aa = 2
    W, H = int(width) * ss_aa, int(height) * ss_aa
    cam = make_camera(eye, target, fov_deg=64.0, aspect=W / H)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(H, W),
              inputs=[img, cam, sun, pdir, int(ms), int(ss), int(W), int(H)],
              device=device)
    wp.synchronize_device(device)
    hdr = post.downsample(img.numpy(), ss_aa)
    r = max(3, int(min(width, height) * 0.015))
    hdr = post.bloom(hdr, threshold=1.5, strength=0.35, radius=r, passes=3, octaves=3)
    return post.tonemap(hdr, mode="aces", exposure=1.06, preserve_hue=True)


SCENE = Scene(
    name="ringed_vista",
    description="A ground-level alien vista: a huge ringed gas giant low over a "
                "rust-coloured dune sea, its tilted rings passing behind and in "
                "front of the disk, dunes lit warm by a low sun and cool by the "
                "giant. --frames pans the view.",
    renderer=_render,
)
