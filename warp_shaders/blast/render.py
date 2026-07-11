"""Hyper-real nuclear-detonation renderer — landscape + fireball + mushroom.

A single ray-march per pixel that composites a **procedural landscape** (rolling
heightfield + an instanced forest of trees), a **volumetric fireball** (blackbody
emission sized + coloured from `blast.physics`), a rising **mushroom cloud**
(turbulent smoke, lit from the fireball below), an expanding **condensation shock
ring** on the ground, and the **damage** the blast does — trees flattened away
from ground zero and the ground scorched inside the physics-sized rings.

All effect radii come from `blast.physics`; a single display `SCALE` maps metres
to frame units so the physical *proportions* (fireball : shock : mushroom : rings)
are preserved while staying viewable. See ``docs/research/15-nuclear-fireball.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.color import kelvin_to_rgb
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.hash import hash21, hash22
from ..procedural.noise import fbm_perlin3
from . import physics as P

_SCALE = wp.constant(8.0 / 3500.0)      # metres -> frame units (Tsar fireball ~8u)
_CELL = wp.constant(2.4)                # forest cell size (frame units)


# --- landscape --------------------------------------------------------------
@wp.func
def _height(x: float, z: float) -> float:
    p = wp.vec3(x * 0.05, 0.0, z * 0.05)
    return fbm_perlin3(p, 4) * 1.4 - 0.4          # gentle rolling ground


@wp.func
def _gnormal(x: float, z: float) -> wp.vec3:
    e = 0.06
    nx = _height(x - e, z) - _height(x + e, z)
    nz = _height(x, z - e) - _height(x, z + e)
    return wp.normalize(wp.vec3(nx, 2.0 * e, nz))


# --- instanced forest (domain-repeated trees with blast knockdown) ----------
@wp.func
def _tree_de(p: wp.vec3, knock: float, fall_dir: wp.vec2, h: float, rad: float) -> float:
    """One tree: a trunk capsule + a canopy sphere, bent by `knock` (0=up,
    1=flat) toward `fall_dir`. Local coords, base at origin."""
    # bend: shear the upper part sideways as it is knocked down
    bend = knock * h
    q = wp.vec3(p[0] - fall_dir[0] * bend * (p[1] / h), p[1] * (1.0 - 0.6 * knock),
                p[2] - fall_dir[1] * bend * (p[1] / h))
    # trunk (vertical capsule)
    ty = wp.clamp(q[1], 0.0, h * 0.6)
    trunk = wp.length(wp.vec3(q[0], q[1] - ty, q[2])) - (0.035 + 0.02 * (1.0 - ty / h))
    # canopy (sphere near the top)
    ctr = wp.vec3(0.0, h * 0.72, 0.0)
    canopy = wp.length(q - ctr) - rad
    return wp.min(trunk, canopy)


@wp.func
def _forest(p: wp.vec3, shock_r: float, dest_r: float, gz: wp.vec2) -> wp.vec2:
    """Distance to the nearest instanced tree + a char factor (0..1). Trees in
    cells the shock has already reached are knocked flat + charred."""
    base = wp.vec3(p[0], p[1] - _height(p[0], p[2]), p[2])
    best = float(1.0e9)
    char = float(0.0)
    cx = wp.floor(p[0] / _CELL)
    cz = wp.floor(p[2] / _CELL)
    for dj in range(-1, 2):
        for di in range(-1, 2):
            cell = wp.vec2(cx + float(di), cz + float(dj))
            rnd = hash22(cell)
            rnd2 = hash21(cell + wp.vec2(3.1, 7.7))
            # skip ~40% of cells so the forest is not a perfect grid
            if rnd2 > 0.6:
                off = wp.vec2((cell[0] + 0.2 + 0.6 * rnd[0]) * _CELL,
                              (cell[1] + 0.2 + 0.6 * rnd[1]) * _CELL)
                d_gz = wp.length(off - gz)
                # knockdown: 1 where the shock has passed (inside shock_r), fading
                knock = wp.clamp((shock_r - d_gz) / (0.18 * dest_r + 0.5), 0.0, 1.0)
                fall = wp.normalize(off - gz + wp.vec2(1.0e-4, 0.0))
                h = 0.34 + 0.16 * rnd[0]
                rad = 0.12 + 0.06 * rnd[1]
                lp = wp.vec3(base[0] - off[0], base[1], base[2] - off[1])
                d = _tree_de(lp, knock, fall, h, rad)
                if d < best:
                    best = d
                    char = knock
    return wp.vec2(best, char)


# --- volumetric fireball + mushroom -----------------------------------------
@wp.func
def _fireball_density(p: wp.vec3, burst: wp.vec3, r_fb: float, t: float) -> float:
    d = wp.length(p - burst)
    shell = wp.smoothstep(r_fb, r_fb * 0.55, d)               # solid core, soft rim
    turb = 0.6 + 0.7 * fbm_perlin3(p * 0.5 + wp.vec3(0.0, -t * 0.3, 0.0), 4)
    return wp.clamp(shell * turb, 0.0, 1.0)


@wp.func
def _cloud_density(p: wp.vec3, burst: wp.vec3, cap_y: float, cap_r: float,
                   stem_r: float, t: float) -> float:
    # rising cap (flattened sphere) + stem column to the ground
    cap_c = wp.vec3(burst[0], cap_y, burst[2])
    pc = wp.vec3(p[0] - cap_c[0], (p[1] - cap_c[1]) * 1.7, p[2] - cap_c[2])
    cap = wp.smoothstep(cap_r, cap_r * 0.6, wp.length(pc))
    # stem: vertical column from ground to the cap
    rr = wp.length(wp.vec2(p[0] - burst[0], p[2] - burst[2]))
    stem = wp.smoothstep(stem_r, stem_r * 0.4, rr) * wp.smoothstep(cap_y, 0.0, p[1]) \
        * wp.smoothstep(-0.5, 1.5, p[1])
    dens = wp.max(cap, stem * 0.8)
    # billowing cauliflower: high-contrast fbm carves the cloud into lobes
    turb = 0.30 + 1.15 * fbm_perlin3(p * 0.5 + wp.vec3(0.0, -t * 0.2, t * 0.1), 5)
    return wp.clamp(dens * turb, 0.0, 1.0)


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3, glow: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.6 + 0.4, 0.0, 1.0)
    base = wp.vec3(0.30, 0.34, 0.42) * (1.0 - up) + wp.vec3(0.10, 0.16, 0.30) * up
    s = wp.max(wp.dot(rd, sun), 0.0)
    horizon = wp.pow(1.0 - wp.clamp(rd[1] + 0.1, 0.0, 1.0), 4.0)
    return base + wp.cw_mul(glow, wp.vec3(horizon, horizon, horizon) * 0.4)


@wp.kernel
def render_ground_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                         burst: wp.vec3, r_fb: float, shock_r: float, dest_r: float,
                         cap_y: float, cap_r: float, stem_r: float, core_k: float,
                         fb_bright: float, t: float, march_steps: int, vol_steps: int,
                         width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    gz = wp.vec2(burst[0], burst[2])
    fb_glow = kelvin_to_rgb(core_k) * 0.5

    # --- opaque scene: terrain + forest ---
    t_hit = float(400.0)
    hit = int(0)
    tm = float(0.5)
    char = float(0.0)
    is_tree = int(0)
    for _ in range(march_steps):
        p = ro + rd * tm
        dh = p[1] - _height(p[0], p[2])                       # height field gap
        fr = _forest(p, shock_r, dest_r, gz)
        d = wp.min(dh, fr[0])
        if d < 0.004 * tm + 0.002:
            hit = 1
            t_hit = tm
            if fr[0] < dh:
                is_tree = 1
                char = fr[1]
            break
        tm += wp.max(d * 0.7, 0.006 * tm)
        if tm > 400.0:
            break

    if hit == 1:
        p = ro + rd * t_hit
        d_gz = wp.length(wp.vec2(p[0], p[2]) - gz)
        if is_tree == 1:
            leaf = wp.vec3(0.10, 0.22, 0.07)
            bark = wp.vec3(0.12, 0.09, 0.06)
            base = leaf * 0.7 + bark * 0.3
            burnt = wp.vec3(0.03, 0.025, 0.02)
            bg = base * (1.0 - char) + burnt * char
            n = wp.normalize(p - wp.vec3(p[0], _height(p[0], p[2]) + 0.3, p[2]))
            ndl = wp.max(wp.dot(n, sun), 0.2)
            col = bg * (0.5 + ndl)
        else:
            n = _gnormal(p[0], p[2])
            grass = wp.vec3(0.09, 0.24, 0.08)
            rock = wp.vec3(0.26, 0.23, 0.20)
            rocky = wp.smoothstep(0.85, 0.6, n[1])
            albedo = grass * (1.0 - rocky) + rock * rocky
            # scorch follows the shock front + a permanent crater at ground zero
            scorch = wp.smoothstep(shock_r, shock_r * 0.6, d_gz)
            crater = wp.smoothstep(r_fb * 1.7, r_fb * 0.5, d_gz)
            scorch = wp.max(scorch, crater)
            # faint thermal singe out ahead of the shock (thermal precedes blast)
            singe = wp.smoothstep(shock_r * 2.2, shock_r, d_gz) * 0.35
            char_col = wp.vec3(0.04, 0.035, 0.03)
            albedo = albedo * (1.0 - scorch) + char_col * scorch
            albedo = albedo * (1.0 - singe) + wp.vec3(0.14, 0.10, 0.07) * singe
            ndl = wp.max(wp.dot(n, sun), 0.0)
            col = wp.cw_mul(albedo, wp.vec3(1.0, 0.9, 0.75) * (0.4 + 1.6 * ndl))
            # fireball throws an orange light on the ground
            fbdir = wp.normalize(burst - p)
            col = col + wp.cw_mul(albedo, fb_glow) * (wp.max(wp.dot(n, fbdir), 0.0)
                                                      * fb_bright * 0.4 / (1.0 + 0.02 * d_gz * d_gz))
    else:
        col = _sky(rd, sun, fb_glow)

    # --- condensation shock ring on the ground ---
    if hit == 1:
        p = ro + rd * t_hit
        d_gz = wp.length(wp.vec2(p[0], p[2]) - gz)
        ring = P.blast_falloff(d_gz, shock_r, 0.06 * dest_r + 0.4)
        col = col + wp.vec3(0.7, 0.75, 0.8) * (ring * 0.7)

    # --- volumetric fireball + mushroom (front-to-back to the opaque hit) ---
    t_end = wp.min(t_hit, 380.0)
    dt = t_end / float(vol_steps)
    tv = dt * 0.5
    trans = float(1.0)
    acc = wp.vec3(0.0, 0.0, 0.0)
    for _ in range(vol_steps):
        if trans < 0.02:
            break
        p = ro + rd * tv
        # fireball emission (blackbody)
        fd = _fireball_density(p, burst, r_fb, t)
        if fd > 0.001:
            rn = wp.length(p - burst) / (r_fb + 1.0e-3)
            temp = P.fireball_temp_at(core_k, rn)
            bright = 0.25 + 1.5 * wp.smoothstep(1.0, 0.0, rn)   # brightest at core
            emis = kelvin_to_rgb(temp) * (fb_bright * 9.0 * bright)
            a = wp.clamp(fd * 2.6 * dt, 0.0, 1.0)
            acc = acc + emis * (a * trans)
            trans = trans * (1.0 - a)
        # smoke cloud (sky-lit crown, orange fireball underlight, dark core)
        cd = _cloud_density(p, burst, cap_y, cap_r, stem_r, t)
        if cd > 0.001:
            dist_fb = wp.length(p - burst)
            under = wp.exp(-dist_fb / (2.5 * r_fb + 1.0e-3))      # fireball underlight
            topl = wp.clamp((p[1] - burst[1]) / (cap_r * 2.0) + 0.55, 0.0, 1.0)
            smoke = wp.vec3(0.40, 0.38, 0.37) * topl + wp.vec3(0.05, 0.05, 0.06) * (1.0 - topl)
            smoke = smoke + fb_glow * (under * 2.4)               # incandescent base
            a = wp.clamp(cd * 1.7 * dt, 0.0, 1.0)
            acc = acc + smoke * (a * trans)
            trans = trans * (1.0 - a)
        tv += dt
    col = col * trans + acc

    img[i, j] = col


def _counts(name):
    return {"low": (150, 40), "medium": (230, 60), "high": (320, 90),
            "ultra": (460, 130)}.get(name, (230, 60))


def render_ground(width, height, time, mouse, device, yield_kt, flash=True):
    """Render one frame of a ground/air-burst detonation of the given yield."""
    tier = active_tier()
    ms, vs = _counts(tier.name)
    s = 8.0 / 3500.0                                  # metres -> frame units

    # physics -> frame-unit radii (proportions preserved)
    r_fb_m = float(P.fireball_radius(yield_kt))
    r_fb = r_fb_m * s
    # the fireball flares then fades as it cools + rises over ~a few seconds
    age = min(time / 10.0, 1.0)
    fb_bright = max(0.12, 1.6 * math.exp(-time * 0.16))
    core_k = float(P.fireball_temp(age))
    r_fb_now = r_fb * (0.4 + 0.6 * min(time / 2.0, 1.0))

    burst_alt = 4000.0 * s
    # the fireball itself buoyantly lifts off the ground as it ages
    rise = max(time - 2.5, 0.0)
    lift = float(P.mushroom_height(rise, yield_kt)) * s * 0.30
    burst = (0.0, burst_alt + lift, 0.0)
    shock_r = float(P.shock_radius(max(time, 0.05), yield_kt)) * s
    dest_r = float(P.destruction_radius(yield_kt)) * s

    # mushroom: the cap rides on the rising fireball; a wide stem to the ground
    cap_y = burst_alt + lift
    cap_r = r_fb * (1.3 + 0.7 * min(rise / 6.0, 1.0))
    stem_r = r_fb * 0.7

    az = 0.0 + float(mouse[0]) * 0.01
    dist = 52.0
    eye = (math.sin(az) * dist, 2.4, math.cos(az) * dist)
    target = (0.0, burst_alt + lift * 0.6 + 2.0, 0.0)
    cam = make_camera(eye, target, fov_deg=62.0, aspect=width / height)
    sun = wp.normalize(wp.vec3(0.4, 0.5, 0.3))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_ground_kernel, dim=(height, width),
              inputs=[img, cam, sun, wp.vec3(*burst), float(r_fb_now), float(shock_r),
                      float(dest_r), float(cap_y), float(cap_r), float(stem_r),
                      float(core_k), float(fb_bright), float(time), int(ms), int(vs),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    # early-time thermal flash — a full-frame white pulse
    if flash and time < 1.2:
        f = (1.2 - time) / 1.2
        hdr = hdr + np.array([1.0, 0.95, 0.85], np.float32) * (f * f * 2.0)

    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5, radius=r, passes=3)
    out = post.tonemap(hdr, mode="aces", exposure=1.1)
    out = post.chromatic_aberration(out, 0.0018)
    return post.vignette(out, 0.32)
