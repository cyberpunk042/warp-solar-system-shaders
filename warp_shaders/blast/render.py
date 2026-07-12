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

from ..buildings.sdf import _rep, _repid, sd_block, sd_house, sd_tower, window_mask
from ..engine import post
from ..engine.color import kelvin_to_rgb
from ..engine.intersect import sphere_t
from ..engine.sky import starfield
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.hash import hash21, hash22
from ..procedural.noise import fbm_perlin3
from ..procedural.sdf import op_union, sd_box
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
                   stem_r: float, r_fb: float, t: float) -> float:
    # rising cap (flattened sphere) + stem column to the ground
    cap_c = wp.vec3(burst[0], cap_y, burst[2])
    pc = wp.vec3(p[0] - cap_c[0], (p[1] - cap_c[1]) * 1.7, p[2] - cap_c[2])
    cap = wp.smoothstep(cap_r, cap_r * 0.6, wp.length(pc))
    # stem: vertical column from ground to the cap (flares out near the cap)
    rr = wp.length(wp.vec2(p[0] - burst[0], p[2] - burst[2]))
    flare = stem_r * (1.0 + 1.2 * wp.clamp(p[1] / (cap_y + 1.0), 0.0, 1.0))
    stem = wp.smoothstep(flare, flare * 0.4, rr) * wp.smoothstep(cap_y, 0.0, p[1]) \
        * wp.smoothstep(-0.5, 1.5, p[1])
    dens = wp.max(cap, stem * 0.85)
    # billowing cauliflower: two oct bands of fbm carve the cloud into lobes
    turb = 0.28 + 0.95 * fbm_perlin3(p * 0.5 + wp.vec3(0.0, -t * 0.2, t * 0.1), 5) \
        + 0.35 * fbm_perlin3(p * 1.5 + wp.vec3(t * 0.1, 0.0, 0.0), 3)
    dens = dens * turb
    # carve a hollow where the incandescent fireball sits (so the core shows)
    dens = dens * wp.smoothstep(r_fb * 0.5, r_fb * 1.15, wp.length(p - burst))
    return wp.clamp(dens, 0.0, 1.0)


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

    # --- condensation shock ring on the ground (layered core + glow, adopted
    # from the-virus-block-mc's shockwave_ring) ---
    if hit == 1:
        p = ro + rd * t_hit
        d_gz = wp.length(wp.vec2(p[0], p[2]) - gz)
        rw = 0.04 * dest_r + 0.3
        ringv = P.shock_ring(d_gz, shock_r, rw, rw * 3.0)
        core_c = wp.vec3(0.85, 0.90, 1.0)          # bright condensation core
        glow_c = wp.vec3(0.55, 0.50, 0.42)         # warm dust glow behind it
        cb = wp.smoothstep(0.0, 0.5, ringv)
        ring_col = glow_c + (core_c - glow_c) * cb
        col = col + ring_col * (ringv * 0.8)

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
        # smoke cloud: dust-laden brown stem -> white condensation crown, dark
        # underside, orange fireball underlight
        cd = _cloud_density(p, burst, cap_y, cap_r, stem_r, r_fb, t)
        if cd > 0.001:
            dist_fb = wp.length(p - burst)
            under = wp.exp(-dist_fb / (2.5 * r_fb + 1.0e-3))      # fireball underlight
            hn = wp.clamp(p[1] / (cap_y + cap_r + 1.0e-3), 0.0, 1.0)
            dust = wp.vec3(0.24, 0.17, 0.11)                      # brown dust (stem)
            crown = wp.vec3(0.66, 0.67, 0.70)                     # condensation (cap)
            mixf = wp.smoothstep(0.32, 0.78, hn)
            body = dust + (crown - dust) * mixf
            topl = wp.clamp((p[1] - burst[1]) / (cap_r * 1.6) + 0.5, 0.0, 1.0)
            body = body * (0.32 + 0.68 * topl)                   # dark underside
            smoke = body + fb_glow * (under * 2.2)               # incandescent base
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
    dist = 40.0 + 1.6 * r_fb                       # pull back for bigger yields
    eye = (math.sin(az) * dist, 0.02 * dist, math.cos(az) * dist)
    target = (0.0, burst_alt + lift * 0.6 + 0.25 * r_fb, 0.0)
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


# --- the nuke tested on a built-up area (buildings collapse by overpressure) --
@wp.func
def _collapse_at(d_gz: float, front_r: float, dest_r: float, sev_r: float) -> float:
    """Collapse factor [0,1] for a lot `d_gz` from ground zero: 0 = intact,
    1 = flattened. One-sided from the front inward (0 at/beyond `front_r`, and at
    t=0), harder toward the 20 psi `sev_r`. See ``docs/research/18-nuke-the-city.md``."""
    passed = wp.smoothstep(front_r, front_r - 0.20 * dest_r, d_gz)
    depth = wp.clamp((dest_r - d_gz) / (dest_r - sev_r + 1.0e-3), 0.0, 1.0)
    collapse = passed * (0.5 + 0.5 * depth)
    collapse = wp.max(collapse, passed * wp.smoothstep(sev_r, sev_r * 0.65, d_gz))
    return wp.clamp(collapse, 0.0, 1.0)


@wp.func
def _blast_de(p: wp.vec3, kind: float, front_r: float, dest_r: float, sev_r: float,
              gz: wp.vec2, lot: float, seed: float) -> wp.vec2:
    """A domain-repeated built-up area whose buildings **collapse** by their
    overpressure grade. `kind < 0.5` = a downtown of towers/blocks
    (`buildings.city_de`), else a suburb of pitched-roof houses
    (`buildings.suburb_de`). Returns ``(dist, collapse)``."""
    idx = _repid(p[0], lot)
    idz = _repid(p[2], lot)
    rnd = hash21(wp.vec2(idx + seed, idz - seed))
    rnd2 = hash21(wp.vec2(idx * 1.7 + 5.3, idz * 2.3 + 9.1))
    qx = _rep(p[0], lot)
    qz = _rep(p[2], lot)
    d_gz = wp.length(wp.vec2(idx * lot, idz * lot) - gz)      # lot centre -> GZ
    collapse = _collapse_at(d_gz, front_r, dest_r, sev_r)

    if kind < 0.5:
        # --- city: tower or low block ---
        h = 4.0 + 19.0 * rnd * rnd                           # tall towers rarer
        w = lot * 0.5 * (0.30 + 0.18 * rnd2)                 # footprint (rest = street)
        h_eff = h * (1.0 - 0.95 * collapse)                  # crush height down
        qb = wp.vec3(qx, p[1] - h_eff, qz)
        if rnd2 < 0.28:
            d = sd_block(qb, wp.vec3(w, h_eff * 0.45, w), 1.5)
        else:
            d = sd_tower(qb, wp.vec3(w, h_eff, w), 1.6, 0.5)
        rub_h = collapse * (0.5 + 1.3 * rnd)
        rw = w * 1.25
    else:
        # --- suburb: a pitched-roof house crushed toward the ground ---
        hw = 2.2 + 0.9 * rnd
        hd = 2.8 + 1.0 * rnd
        bh = 1.7 + 0.7 * rnd2                                # body half-height
        roof = 2.0 + 0.9 * rnd2
        bh_eff = bh * (1.0 - 0.9 * collapse)
        roof_eff = roof * (1.0 - 0.96 * collapse)            # roof caves in first
        qb = wp.vec3(qx, p[1] - bh_eff, qz)
        d = sd_house(qb, wp.vec3(hw, bh_eff, hd), roof_eff)
        rub_h = collapse * (0.4 + 0.7 * rnd)
        rw = hw * 1.1

    # rubble mound piling up in the footprint where the building came down
    rubble = sd_box(wp.vec3(qx, p[1] - rub_h, qz), wp.vec3(rw, rub_h + 0.05, rw))
    return wp.vec2(op_union(d, rubble), collapse)


@wp.func
def _city_blast_de(p: wp.vec3, front_r: float, dest_r: float, sev_r: float,
                   gz: wp.vec2, lot: float, seed: float) -> wp.vec2:
    """City (tower) specialization of :func:`_blast_de` — kept for the collapse
    unit test; the render kernel calls :func:`_blast_de` directly with `kind`."""
    return _blast_de(p, 0.0, front_r, dest_r, sev_r, gz, lot, seed)


@wp.func
def _blast_map(p: wp.vec3, kind: float, front_r: float, dest_r: float, sev_r: float,
               gz: wp.vec2, lot: float, seed: float) -> float:
    return wp.min(p[1] - _height(p[0], p[2]),
                  _blast_de(p, kind, front_r, dest_r, sev_r, gz, lot, seed)[0])


@wp.func
def _blast_normal(p: wp.vec3, kind: float, front_r: float, dest_r: float, sev_r: float,
                  gz: wp.vec2, lot: float, seed: float) -> wp.vec3:
    e = 0.03
    dx = _blast_map(p + wp.vec3(e, 0.0, 0.0), kind, front_r, dest_r, sev_r, gz, lot, seed) \
        - _blast_map(p - wp.vec3(e, 0.0, 0.0), kind, front_r, dest_r, sev_r, gz, lot, seed)
    dy = _blast_map(p + wp.vec3(0.0, e, 0.0), kind, front_r, dest_r, sev_r, gz, lot, seed) \
        - _blast_map(p - wp.vec3(0.0, e, 0.0), kind, front_r, dest_r, sev_r, gz, lot, seed)
    dz = _blast_map(p + wp.vec3(0.0, 0.0, e), kind, front_r, dest_r, sev_r, gz, lot, seed) \
        - _blast_map(p - wp.vec3(0.0, 0.0, e), kind, front_r, dest_r, sev_r, gz, lot, seed)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ground_smoke(p: wp.vec3, front_r: float, gz: wp.vec2, top: float, t: float) -> float:
    """Sparse rising smoke wisps over the burning zone — thin, patchy plumes that
    veil the fires without smothering them (not a solid pall)."""
    rr = wp.length(wp.vec2(p[0] - gz[0], p[2] - gz[2]))
    disc = wp.smoothstep(front_r * 1.05, front_r * 0.55, rr)
    high = wp.smoothstep(top, 2.0, p[1])                      # rises off the ground
    n1 = fbm_perlin3(p * 0.11 + wp.vec3(0.0, -t * 0.2, t * 0.05), 4)
    wisp = wp.smoothstep(0.2, 0.62, n1)                       # only strong-noise patches
    return wp.clamp(disc * high * wisp, 0.0, 1.0)


@wp.kernel
def render_blast_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                        kind: float, burst: wp.vec3, r_fb: float, front_r: float,
                        dest_r: float, sev_r: float, cap_y: float, cap_r: float,
                        stem_r: float, core_k: float, fb_bright: float, lot: float,
                        seed: float, t: float, march_steps: int, vol_steps: int,
                        width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    gz = wp.vec2(burst[0], burst[2])
    fb_glow = kelvin_to_rgb(core_k) * 0.5
    smoke_top = dest_r * 0.55                             # height of the burning pall

    # --- opaque scene: terrain + collapsing built-up area ---
    t_hit = float(600.0)
    hit = int(0)
    tm = float(0.5)
    collapse = float(0.0)
    is_bld = int(0)
    for _ in range(march_steps):
        p = ro + rd * tm
        dh = p[1] - _height(p[0], p[2])
        cb = _blast_de(p, kind, front_r, dest_r, sev_r, gz, lot, seed)
        d = wp.min(dh, cb[0])
        if d < 0.004 * tm + 0.002:
            hit = 1
            t_hit = tm
            if cb[0] < dh:
                is_bld = 1
                collapse = cb[1]
            break
        tm += wp.max(d * 0.45, 0.005 * tm)                # smaller step -> less ghosting
        if tm > 600.0:
            break

    if hit == 1:
        p = ro + rd * t_hit
        d_gz = wp.length(wp.vec2(p[0], p[2]) - gz)
        if is_bld == 1:
            n = _blast_normal(p, kind, front_r, dest_r, sev_r, gz, lot, seed)
            standing = 1.0 - collapse
            idx = _repid(p[0], lot)
            idz = _repid(p[2], lot)
            rnd2 = hash21(wp.vec2(idx * 1.7 + 5.3, idz * 2.3 + 9.1))
            if kind < 0.5:
                intact = wp.vec3(0.15, 0.16, 0.19)            # dark dusk concrete
            else:
                # plaster wall vs terracotta roof (roof = geometry near the top)
                bh = (1.7 + 0.7 * rnd2) * (1.0 - 0.9 * collapse)
                roof_m = wp.step(p[1] - 1.7 * bh) * standing
                plaster = wp.vec3(0.34, 0.29, 0.23)
                tile = wp.vec3(0.30, 0.10, 0.06)
                intact = plaster * (1.0 - roof_m) + tile * roof_m
            rubble = wp.vec3(0.10, 0.09, 0.08)                # broken concrete/dust
            burnt = wp.vec3(0.03, 0.028, 0.025)
            mat = intact * (1.0 - collapse) + rubble * collapse
            mat = mat * (1.0 - 0.6 * collapse) + burnt * (0.6 * collapse)
            ndl = wp.max(wp.dot(n, sun), 0.0)
            col = wp.cw_mul(mat, wp.vec3(1.0, 0.86, 0.66) * (0.12 + 0.5 * ndl))   # dim warm sun
            col = col + wp.cw_mul(mat, wp.vec3(0.10, 0.14, 0.22)) * (0.35 + 0.3 * n[1])  # low sky
            # the fireball floods the standing faces that see it with hot light
            fbdir = wp.normalize(burst - p)
            col = col + wp.cw_mul(mat, fb_glow) * (wp.max(wp.dot(n, fbdir), 0.0)
                                                   * fb_bright * 1.1 / (1.0 + 0.010 * d_gz * d_gz))
            # emissive lit windows on STANDING buildings — they go dark as the
            # building collapses (the blast wave extinguishes the city)
            fl = wp.floor(p[1] / 2.0)
            lh = hash21(wp.vec2(idx * 13.0 + fl, idz * 7.0 - fl))
            win = window_mask(p, 1.7, 2.0)
            lit = win * wp.step(lh - 0.5) * standing
            wc = wp.vec3(1.0, 0.80, 0.46) + wp.vec3(0.0, 0.06, 0.22) * (lh - 0.5)
            col = col + wc * (lit * 1.9)
            # collapsed buildings SMOULDER — patchy fires broken up by noise so the
            # rubble reads as scattered burning debris, not a uniform glowing top
            fh = hash21(wp.vec2(idx * 5.3 - fl * 2.0, idz * 3.7 + fl))
            turb = wp.clamp(0.35 + 0.75 * fbm_perlin3(p * 0.55, 3), 0.0, 1.0)
            ember = collapse * wp.smoothstep(0.32, 0.95, fh) * turb
            fire = wp.vec3(1.0, 0.30, 0.06) + wp.vec3(0.0, 0.36, 0.0) * fh   # deep orange->amber
            col = col + fire * (ember * 2.7)
        else:
            n = _gnormal(p[0], p[2])
            street = wp.vec3(0.07, 0.07, 0.08)                # dark asphalt/lawn
            n_road = wp.smoothstep(0.9, 0.7, n[1])
            albedo = street * (1.0 - n_road) + wp.vec3(0.05, 0.05, 0.05) * n_road
            # everything the front has passed is a burnt scar (out to the 5 psi ring)
            scorch = wp.smoothstep(front_r, front_r * 0.9, d_gz)
            crater = wp.smoothstep(r_fb * 1.8, r_fb * 0.5, d_gz)
            scorch = wp.max(scorch, crater)
            char_col = wp.vec3(0.03, 0.025, 0.02)
            albedo = albedo * (1.0 - scorch) + char_col * scorch
            ndl = wp.max(wp.dot(n, sun), 0.0)
            col = wp.cw_mul(albedo, wp.vec3(1.0, 0.9, 0.75) * (0.2 + 0.9 * ndl))
            col = col + wp.cw_mul(albedo, wp.vec3(0.09, 0.12, 0.20)) * 0.5   # sky ambient
            fbdir = wp.normalize(burst - p)
            col = col + wp.cw_mul(albedo, fb_glow) * (wp.max(wp.dot(n, fbdir), 0.0)
                                                      * fb_bright * 0.8 / (1.0 + 0.015 * d_gz * d_gz))
            # the burnt ground smoulders — a patchy field of embers in the scar
            gh = hash21(wp.vec2(wp.floor(p[0] * 0.7), wp.floor(p[2] * 0.7)))
            gturb = wp.clamp(0.3 + 0.8 * fbm_perlin3(p * 0.4, 3), 0.0, 1.0)
            gember = scorch * wp.smoothstep(0.55, 0.95, gh) * gturb
            col = col + wp.vec3(1.0, 0.26, 0.05) * (gember * 1.4)
    else:
        col = _sky(rd, sun, fb_glow)

    # --- condensation shock ring riding the overpressure front ---
    if hit == 1:
        p = ro + rd * t_hit
        d_gz = wp.length(wp.vec2(p[0], p[2]) - gz)
        rw = 0.03 * dest_r + 0.4
        ringv = P.shock_ring(d_gz, front_r, rw, rw * 3.0)
        core_c = wp.vec3(0.85, 0.90, 1.0)
        glow_c = wp.vec3(0.55, 0.50, 0.42)
        cbl = wp.smoothstep(0.0, 0.5, ringv)
        ring_col = glow_c + (core_c - glow_c) * cbl
        col = col + ring_col * (ringv * 0.7)

    # --- volumetric fireball + mushroom (front-to-back to the opaque hit) ---
    t_end = wp.min(t_hit, 560.0)
    dt = t_end / float(vol_steps)
    tv = dt * 0.5
    trans = float(1.0)
    acc = wp.vec3(0.0, 0.0, 0.0)
    for _ in range(vol_steps):
        if trans < 0.02:
            break
        p = ro + rd * tv
        fd = _fireball_density(p, burst, r_fb, t)
        if fd > 0.001:
            rn = wp.length(p - burst) / (r_fb + 1.0e-3)
            temp = P.fireball_temp_at(core_k, rn)
            bright = 0.25 + 1.5 * wp.smoothstep(1.0, 0.0, rn)
            emis = kelvin_to_rgb(temp) * (fb_bright * 9.0 * bright)
            a = wp.clamp(fd * 2.6 * dt, 0.0, 1.0)
            acc = acc + emis * (a * trans)
            trans = trans * (1.0 - a)
        cd = _cloud_density(p, burst, cap_y, cap_r, stem_r, r_fb, t)
        if cd > 0.001:
            dist_fb = wp.length(p - burst)
            under = wp.exp(-dist_fb / (2.5 * r_fb + 1.0e-3))
            hn = wp.clamp(p[1] / (cap_y + cap_r + 1.0e-3), 0.0, 1.0)
            dust = wp.vec3(0.24, 0.17, 0.11)
            crown = wp.vec3(0.66, 0.67, 0.70)
            mixf = wp.smoothstep(0.32, 0.78, hn)
            body = dust + (crown - dust) * mixf
            topl = wp.clamp((p[1] - burst[1]) / (cap_r * 1.6) + 0.5, 0.0, 1.0)
            body = body * (0.32 + 0.68 * topl)
            smoke = body + fb_glow * (under * 2.2)
            a = wp.clamp(cd * 1.7 * dt, 0.0, 1.0)
            acc = acc + smoke * (a * trans)
            trans = trans * (1.0 - a)
        # ground-hugging smoke pall over the burning zone — dark, drifting, and
        # underlit orange by the fires below (a burning-city haze)
        sm = _ground_smoke(p, front_r, gz, smoke_top, t)
        if sm > 0.001:
            glow = wp.smoothstep(smoke_top, 0.0, p[1])       # brighter near the fires
            sc = wp.vec3(0.11, 0.09, 0.09) + wp.vec3(1.0, 0.34, 0.10) * (glow * 0.7)
            a = wp.clamp(sm * 0.5 * dt, 0.0, 1.0)            # thin — veils, not smothers
            acc = acc + sc * (a * trans)
            trans = trans * (1.0 - a)
        tv += dt
    col = col * trans + acc

    img[i, j] = col


def _render_blast(width, height, time, mouse, device, yield_kt, kind, lot, eye_y,
                  dist, target_y, fov):
    """Render one frame of a detonation over a built-up area — the buildings
    collapse into a burning field of rubble as the overpressure front sweeps out to
    the 5 psi destruction ring, under a mushroom rising from the centre. `kind` = 0
    downtown towers / 1 suburb houses. The front expands over the shot (the
    visualization compresses the few real seconds of wave propagation); final damage
    radii are from `blast.physics`. See ``docs/research/18-nuke-the-city.md``."""
    tier = active_tier()
    ms, vs = _counts(tier.name)
    s = 60.0 / float(P.destruction_radius(yield_kt))   # 5 psi ring -> 60 frame units

    r_fb = float(P.fireball_radius(yield_kt)) * s
    dest_r = float(P.destruction_radius(yield_kt)) * s
    sev_r = float(P.severe_radius(yield_kt)) * s
    seed = 3.0

    # the fireball flares then fades; the overpressure front expands to the 5 psi
    # ring over ~6 s and rolls a little past it
    age = min(time / 10.0, 1.0)
    fb_bright = max(0.15, 2.2 * math.exp(-time * 0.14))
    core_k = float(P.fireball_temp(age))
    r_fb_vis = r_fb * 2.2                                # the fireball reads bigger
    r_fb_now = r_fb_vis * (0.5 + 0.5 * min(time / 2.0, 1.0))
    front_r = dest_r * min(time / 6.0, 1.0) * 1.12

    burst_alt = r_fb_vis * 1.4 + 10.0                   # low air-burst above downtown
    rise = max(time - 1.5, 0.0)
    # a moderate mushroom rising from the burning area (kept low enough that it
    # does not occlude ground zero — the destruction is the subject)
    lift = float(P.mushroom_height(rise, yield_kt)) * s * 1.05
    burst = (0.0, burst_alt + lift, 0.0)

    cap_y = burst_alt + lift
    cap_r = r_fb_vis * (1.8 + 1.6 * min(rise / 6.0, 1.0))
    stem_r = r_fb_vis * 1.0

    # aerial 3/4 looking DOWN into the flattened, burning zone — the devastation
    # (a scorched rubble field of embers) fills the midground; the mushroom rises
    # behind ground zero; standing lit buildings ring the far perimeter.
    az = 0.7 + float(mouse[0]) * 0.01
    eye = (math.sin(az) * dist, eye_y + float(mouse[1]) * 0.1, math.cos(az) * dist)
    cam = make_camera(eye, (0.0, target_y, 0.0), fov_deg=fov, aspect=width / height)
    sun = wp.normalize(wp.vec3(0.5, 0.32, 0.35))        # low dusk sun

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_blast_kernel, dim=(height, width),
              inputs=[img, cam, sun, float(kind), wp.vec3(*burst), float(r_fb_now),
                      float(front_r), float(dest_r), float(sev_r), float(cap_y),
                      float(cap_r), float(stem_r), float(core_k), float(fb_bright),
                      float(lot), float(seed), float(time), int(ms), int(vs),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    if time < 1.2:
        f = (1.2 - time) / 1.2
        hdr = hdr + np.array([1.0, 0.95, 0.85], np.float32) * (f * f * 2.2)

    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=0.8, strength=0.6, radius=r, passes=3)
    out = post.tonemap(hdr, mode="aces", exposure=0.92)
    out = post.chromatic_aberration(out, 0.0002)
    return post.vignette(out, 0.4)


def render_city(width, height, time, mouse, device, yield_kt):
    """A detonation over a **downtown** of SDF towers/blocks (see
    :func:`_render_blast`)."""
    return _render_blast(width, height, time, mouse, device, yield_kt, kind=0.0,
                         lot=15.0, eye_y=92.0, dist=138.0, target_y=6.0, fov=62.0)


def render_suburb(width, height, time, mouse, device, yield_kt):
    """A detonation over a **suburb** of pitched-roof houses — smaller, lower
    buildings, so a lower camera reads the human-scale devastation."""
    return _render_blast(width, height, time, mouse, device, yield_kt, kind=1.0,
                         lot=8.0, eye_y=64.0, dist=104.0, target_y=3.0, fov=60.0)


# --- vacuum burst over a planet (no atmosphere -> no blast/fireball/mushroom) -
@wp.func
def _planet_shade(p: wp.vec3, pc: wp.vec3, rp: float, sun: wp.vec3, rd: wp.vec3) -> wp.vec3:
    n = wp.normalize(p - pc)
    cont = fbm_perlin3(n * 2.4 + wp.vec3(11.0, 3.0, 7.0), 5)
    land = wp.smoothstep(0.48, 0.56, cont * 0.5 + 0.5)
    ocean = wp.vec3(0.02, 0.12, 0.32)
    green = wp.vec3(0.08, 0.26, 0.10)
    base = ocean * (1.0 - land) + green * land
    cloud = wp.smoothstep(0.58, 0.72, fbm_perlin3(n * 3.3 + wp.vec3(-4.0, 9.0, 2.0), 5) * 0.5 + 0.5)
    base = base * (1.0 - cloud * 0.8) + wp.vec3(0.9, 0.92, 0.95) * (cloud * 0.8)
    ndl = wp.dot(n, sun)
    day = wp.clamp(ndl, 0.0, 1.0)
    col = base * (0.04 + 1.15 * day)
    rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
    col = col + wp.vec3(0.25, 0.45, 0.85) * (rim * (0.3 + 0.7 * day))     # atmosphere limb
    return col


@wp.kernel
def render_space_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                        pc: wp.vec3, rp: float, burst: wp.vec3, shell_r: float,
                        shell_w: float, core_k: float, flash: float, t: float,
                        vol_steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    # background: planet or stars
    pt = sphere_t(ro, rd, pc, rp)
    t_hit = float(1.0e9)
    if pt > 0.0:
        t_hit = pt
        col = _planet_shade(ro + rd * pt, pc, rp, sun, rd)
    else:
        col = starfield(rd)

    # Starfish-Prime aurora: a faint glow on the planet near the burst footprint
    if pt > 0.0:
        sp = ro + rd * pt
        fp = pc + wp.normalize(burst - pc) * rp                # burst ground track
        au = wp.exp(-wp.length(sp - fp) / (0.5 * rp)) * wp.smoothstep(shell_r * 0.2, shell_r, wp.length(sp - fp) + shell_r * 0.2)
        col = col + wp.vec3(0.1, 0.6, 0.3) * (au * 0.5)

    # expanding plasma / debris shell (ballistic; no blast wave, no mushroom)
    t_end = wp.min(t_hit, 400.0)
    dt = t_end / float(vol_steps)
    tv = dt * 0.5
    trans = float(1.0)
    acc = wp.vec3(0.0, 0.0, 0.0)
    shell_col = kelvin_to_rgb(core_k)
    for _ in range(vol_steps):
        if trans < 0.02:
            break
        p = ro + rd * tv
        d = wp.length(p - burst)
        # thin bright shell at the expanding front (hollow inside)
        e = (d - shell_r) / (shell_w + 1.0e-3)
        shell = wp.exp(-e * e)
        # radial debris streaks, concentrated at the front
        dir = wp.normalize(p - burst + wp.vec3(1.0e-4, 0.0, 0.0))
        streak = wp.pow(fbm_perlin3(dir * 10.0, 4) * 0.5 + 0.5, 4.0)
        dens = shell * (0.7 + 2.5 * streak)
        if dens > 0.001:
            emis = shell_col * (dens * flash * 3.0)
            a = wp.clamp(dens * 0.7, 0.0, 1.0)
            acc = acc + emis * (a * trans)
            trans = trans * (1.0 - a)
        tv += dt
    # faint hollow after-glow just inside the shell
    col = col * trans + acc

    img[i, j] = col


def render_space(width, height, time, mouse, device, yield_kt):
    """Render one frame of a vacuum burst above a planet — no atmosphere, so no
    blast wave, no incandescent fireball, no mushroom: a ballistic plasma shell."""
    tier = active_tier()
    _, vs = _counts(tier.name)

    rp = 26.0
    pc = (0.0, -6.0, 0.0)
    burst = (10.0, 22.0, 6.0)                         # above the planet limb
    # ballistic shell: radius linear in time (per blast.debris_shell_radius)
    shell_r = 1.5 + 5.0 * time
    shell_w = 0.5 + 0.22 * time                       # thickens + dims as it expands
    age = min(time / 6.0, 1.0)
    core_k = float(P.fireball_temp(age * 0.7))        # X-ray-hot -> cools; stays bluer
    flash = max(0.1, 1.5 * math.exp(-time * 0.3)) / (1.0 + 0.05 * shell_r)

    az = 0.2 + float(mouse[0]) * 0.01
    dist = 80.0
    eye = (math.sin(az) * dist, 16.0, math.cos(az) * dist)
    cam = make_camera(eye, (0.0, 6.0, 0.0), fov_deg=52.0, aspect=width / height)
    sun = wp.normalize(wp.vec3(0.7, 0.35, 0.5))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_space_kernel, dim=(height, width),
              inputs=[img, cam, sun, wp.vec3(*pc), float(rp), wp.vec3(*burst),
                      float(shell_r), float(shell_w), float(core_k), float(flash),
                      float(time), int(vs), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    if time < 0.8:
        f = (0.8 - time) / 0.8
        hdr = hdr + np.array([0.9, 0.95, 1.0], np.float32) * (f * f * 3.0)   # X-ray flash
    r = max(3, int(min(width, height) * 0.016))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.45, radius=r, passes=3)
    out = post.tonemap(hdr, mode="aces", exposure=1.05)
    return post.vignette(out, 0.34)
