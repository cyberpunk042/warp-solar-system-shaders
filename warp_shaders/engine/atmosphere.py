"""Physically based atmospheric scattering (Warp device functions).

Analytic single-scattering, raymarched: march the view ray; at each sample
light-march toward the sun for optical depth; accumulate Rayleigh + Mie
in-scattering with their phase functions and Beer-Lambert transmittance. Ground
intersection shadows samples (correct terminator / night). No precomputed LUTs,
so it runs at every tier including CPU; sample counts scale with the quality tier.

Sources (docs/research): Nishita et al. 1993; Sean O'Neil, GPU Gems 2 ch.16;
Cornette-Shanks phase (Mie). Earth-like coefficients in SI (metres).

Precomputed transmittance/multiscatter LUTs (Bruneton 2008 / Hillaire 2020) are a
drop-in acceleration for high/ultra later; the analytic path stays as the low tier.
"""

import warp as wp

from ..textures import sample2d

_PI = 3.14159265

# Earth-like atmosphere (metres / per-metre)
_RG = 6360000.0                 # planet (ground) radius
_RA = 6420000.0                 # atmosphere top radius
_HR = 8000.0                    # Rayleigh scale height
_HM = 1200.0                    # Mie scale height
_BETA_R = wp.constant(wp.vec3(5.8e-6, 13.5e-6, 33.1e-6))  # Rayleigh scattering
_BETA_M = 21e-6                 # Mie scattering (grey)
_MIE_G = 0.76                   # Mie anisotropy
_SUN_I = 22.0                   # sun intensity


@wp.func
def _ray_sphere(ro: wp.vec3, rd: wp.vec3, radius: float) -> wp.vec2:
    """Intersect a ray with a sphere at the origin. Miss -> (1e30, -1e30)."""
    b = wp.dot(ro, rd)
    c = wp.dot(ro, ro) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return wp.vec2(1e30, -1e30)
    s = wp.sqrt(disc)
    return wp.vec2(-b - s, -b + s)


@wp.func
def _vexp(v: wp.vec3) -> wp.vec3:
    return wp.vec3(wp.exp(v[0]), wp.exp(v[1]), wp.exp(v[2]))


@wp.func
def atmosphere(ro: wp.vec3, rd: wp.vec3, sun: wp.vec3,
               view_samples: int, light_samples: int) -> wp.vec3:
    """In-scattered sky radiance along a view ray (ro in planet-centred metres)."""
    atm = _ray_sphere(ro, rd, _RA)
    t0 = wp.max(atm[0], 0.0)
    t1 = atm[1]
    if t1 < 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    gnd = _ray_sphere(ro, rd, _RG)
    if gnd[0] > 0.0 and gnd[1] > 0.0:
        t1 = wp.min(t1, gnd[0])

    seg = (t1 - t0) / float(view_samples)
    od_r = float(0.0)
    od_m = float(0.0)
    sum_r = wp.vec3(0.0, 0.0, 0.0)
    sum_m = wp.vec3(0.0, 0.0, 0.0)

    mu = wp.dot(rd, sun)
    phase_r = 3.0 / (16.0 * _PI) * (1.0 + mu * mu)
    g2 = _MIE_G * _MIE_G
    phase_m = (3.0 / (8.0 * _PI)) * ((1.0 - g2) * (1.0 + mu * mu)) \
        / ((2.0 + g2) * wp.pow(1.0 + g2 - 2.0 * _MIE_G * mu, 1.5))

    t = t0 + 0.5 * seg
    for _ in range(view_samples):
        p = ro + rd * t
        h = wp.length(p) - _RG
        hr = wp.exp(-h / _HR) * seg
        hm = wp.exp(-h / _HM) * seg
        od_r += hr
        od_m += hm

        la = _ray_sphere(p, sun, _RA)
        lg = _ray_sphere(p, sun, _RG)
        blocked = int(0)
        if lg[0] > 0.0 and lg[1] > 0.0:
            blocked = 1
        if blocked == 0:
            seg_l = la[1] / float(light_samples)
            od_lr = float(0.0)
            od_lm = float(0.0)
            tl = 0.5 * seg_l
            for _ in range(light_samples):
                pl = p + sun * tl
                hl = wp.length(pl) - _RG
                od_lr += wp.exp(-hl / _HR) * seg_l
                od_lm += wp.exp(-hl / _HM) * seg_l
                tl += seg_l
            tau = _BETA_R * (od_r + od_lr) + wp.vec3(_BETA_M, _BETA_M, _BETA_M) * (1.1 * (od_m + od_lm))
            att = _vexp(-tau)
            sum_r += att * hr
            sum_m += att * hm
        t += seg

    return (wp.cw_mul(sum_r, _BETA_R) * phase_r
            + sum_m * (_BETA_M * phase_m)) * _SUN_I


@wp.func
def sky_radiance(ro: wp.vec3, rd: wp.vec3, sun: wp.vec3,
                 view_samples: int, light_samples: int) -> wp.vec3:
    """Atmosphere in-scatter plus the sun disk (attenuated by the atmosphere)."""
    col = atmosphere(ro, rd, sun, view_samples, light_samples)
    mu = wp.dot(rd, sun)
    # sun disk (~0.53 deg): only when the sun is above the local horizon
    disk = wp.smoothstep(0.99985, 0.99992, mu)
    if disk > 0.0:
        gnd = _ray_sphere(ro, rd, _RG)
        if not (gnd[0] > 0.0 and gnd[1] > 0.0):
            col = col + wp.vec3(1.0, 0.95, 0.85) * (disk * _SUN_I * 18.0)
    return col


def sample_counts(tier_name: str):
    """(view_samples, light_samples) per quality tier."""
    return {
        "low": (16, 6), "medium": (24, 8), "high": (32, 12), "ultra": (48, 16),
    }.get(tier_name, (24, 8))


# ---- precomputed transmittance LUT (Bruneton/Hillaire-style) ----------------
# T(h, mu): transmittance from altitude h, sun-zenith cosine mu, to the atmosphere
# top. Precomputing it once removes the per-view-sample inner sun light-march.

@wp.kernel
def bake_transmittance(lut: wp.array2d(dtype=wp.vec3), size: int, steps: int):
    i, j = wp.tid()                                  # i = altitude, j = mu
    mu = (float(j) + 0.5) / float(size) * 2.0 - 1.0
    h = (float(i) + 0.5) / float(size) * (_RA - _RG)
    ro = wp.vec3(0.0, _RG + h, 0.0)
    dir = wp.vec3(wp.sqrt(wp.max(1.0 - mu * mu, 0.0)), mu, 0.0)
    gnd = _ray_sphere(ro, dir, _RG)
    if gnd[0] > 0.0 and gnd[1] > 0.0:
        lut[i, j] = wp.vec3(0.0, 0.0, 0.0)           # sun below horizon -> blocked
        return
    atm = _ray_sphere(ro, dir, _RA)
    seg = atm[1] / float(steps)
    od_r = float(0.0)
    od_m = float(0.0)
    t = 0.5 * seg
    for _ in range(steps):
        hh = wp.length(ro + dir * t) - _RG
        od_r += wp.exp(-hh / _HR) * seg
        od_m += wp.exp(-hh / _HM) * seg
        t += seg
    tau = _BETA_R * od_r + wp.vec3(_BETA_M, _BETA_M, _BETA_M) * (1.1 * od_m)
    lut[i, j] = _vexp(-tau)


def build_transmittance_lut(size=64, device="cpu", steps=32):
    lut = wp.zeros((size, size), dtype=wp.vec3, device=device)
    wp.launch(bake_transmittance, dim=(size, size), inputs=[lut, int(size), int(steps)],
              device=device)
    wp.synchronize_device(device)
    return lut


@wp.func
def transmittance_lut(lut: wp.array2d(dtype=wp.vec3), h: float, mu: float) -> wp.vec3:
    u = mu * 0.5 + 0.5
    v = wp.clamp(h / (_RA - _RG), 0.0, 1.0)
    return sample2d(lut, u, v, 0, 0)


# ---- precomputed multiple-scattering LUT (Hillaire 2020) ---------------------
# psi_ms(h, mu): the infinite-order, isotropic multiple-scattered luminance
# response to unit sun illuminance, per unit local scattering coefficient. At
# runtime a view sample adds  psi_ms * sigma_s(h) * view_transmittance  — no sun
# phase, no inner sun march — so it brightens twilight / horizon / shadowed sky
# without touching the phase-weighted single-scatter term. Consumes the
# transmittance LUT (sun-path attenuation + ground visibility already baked in).

_MS_GROUND_ALBEDO = 0.30        # Lambertian ground bounce into the light field
_MS_SCALE = 0.22                # calibration into this model's tuned radiance budget
                                # (single-scatter here is artistically boosted; a full-
                                #  strength physical MS would double-count — this scales
                                #  the Hillaire term to a realistic, non-blowout lift)


@wp.func
def _fib_sphere(k: int, n: int) -> wp.vec3:
    """Deterministic uniform sphere sample k of n (Fibonacci sphere; y = up)."""
    ga = 2.39996323             # golden angle = pi * (3 - sqrt(5))
    z = 1.0 - (2.0 * float(k) + 1.0) / float(n)     # (k+0.5)-centred cosine in (-1,1)
    r = wp.sqrt(wp.max(1.0 - z * z, 0.0))
    phi = ga * float(k)
    return wp.vec3(r * wp.cos(phi), z, r * wp.sin(phi))


@wp.kernel
def bake_multiscatter(ms_lut: wp.array2d(dtype=wp.vec3),
                      tr_lut: wp.array2d(dtype=wp.vec3),
                      size: int, dir_samples: int, steps: int):
    i, j = wp.tid()                                  # i = altitude, j = sun-mu
    mu = (float(j) + 0.5) / float(size) * 2.0 - 1.0
    h = (float(i) + 0.5) / float(size) * (_RA - _RG)
    p0 = wp.vec3(0.0, _RG + h, 0.0)
    sun = wp.vec3(wp.sqrt(wp.max(1.0 - mu * mu, 0.0)), mu, 0.0)

    L2 = wp.vec3(0.0, 0.0, 0.0)                       # 2nd-order (sun illum = 1)
    fms = wp.vec3(0.0, 0.0, 0.0)                      # transfer (response to unit ambient)

    for k in range(dir_samples):
        w = _fib_sphere(k, dir_samples)
        atm = _ray_sphere(p0, w, _RA)
        tmax = atm[1]
        gnd = _ray_sphere(p0, w, _RG)
        hit_ground = int(0)
        if gnd[0] > 0.0:
            tmax = wp.min(tmax, gnd[0])
            hit_ground = 1
        if tmax > 0.0:
            seg = tmax / float(steps)
            thr = wp.vec3(1.0, 1.0, 1.0)             # throughput T(p0, x') along w
            l_dir = wp.vec3(0.0, 0.0, 0.0)
            f_dir = wp.vec3(0.0, 0.0, 0.0)
            t = 0.5 * seg
            for _ in range(steps):
                x = p0 + w * t
                hh = wp.length(x) - _RG
                dr = wp.exp(-hh / _HR)
                dm = wp.exp(-hh / _HM)
                sig_s = _BETA_R * dr + wp.vec3(_BETA_M, _BETA_M, _BETA_M) * dm
                sig_t = _BETA_R * dr + wp.vec3(_BETA_M, _BETA_M, _BETA_M) * (1.1 * dm)
                mus = wp.dot(wp.normalize(x), sun)
                tsun = transmittance_lut(tr_lut, hh, mus)   # illum 1; visibility folded in
                # isotropic: phase(1/4pi) x sphere(4pi) cancel -> average over N
                l_dir += wp.cw_mul(thr, wp.cw_mul(sig_s, tsun)) * seg
                f_dir += wp.cw_mul(thr, sig_s) * seg
                thr = wp.cw_mul(thr, _vexp(-sig_t * seg))
                t += seg
            if hit_ground == 1:
                xg = p0 + w * tmax
                mug = wp.dot(wp.normalize(xg), sun)
                if mug > 0.0:
                    tsg = transmittance_lut(tr_lut, 0.0, mug)
                    l_dir += wp.cw_mul(thr, tsg) * (_MS_GROUND_ALBEDO * mug / _PI)
            L2 += l_dir
            fms += f_dir

    inv = 1.0 / float(dir_samples)
    L2 = L2 * inv
    fms = fms * inv
    d = wp.vec3(1.0, 1.0, 1.0) - fms                 # geometric-series denominator
    ms_lut[i, j] = wp.vec3(L2[0] / wp.max(d[0], 1e-4),
                           L2[1] / wp.max(d[1], 1e-4),
                           L2[2] / wp.max(d[2], 1e-4))


def build_multiscatter_lut(tr_lut, size=32, device="cpu", dir_samples=32, steps=32):
    """Bake the multiple-scattering LUT (needs the transmittance LUT, same device)."""
    ms = wp.zeros((size, size), dtype=wp.vec3, device=device)
    wp.launch(bake_multiscatter, dim=(size, size),
              inputs=[ms, tr_lut, int(size), int(dir_samples), int(steps)], device=device)
    wp.synchronize_device(device)
    return ms


@wp.func
def multiscatter_lut(ms_lut: wp.array2d(dtype=wp.vec3), h: float, mu: float) -> wp.vec3:
    u = mu * 0.5 + 0.5
    v = wp.clamp(h / (_RA - _RG), 0.0, 1.0)
    return sample2d(ms_lut, u, v, 0, 0)


@wp.func
def atmosphere_lut(ro: wp.vec3, rd: wp.vec3, sun: wp.vec3, view_samples: int,
                   lut: wp.array2d(dtype=wp.vec3),
                   ms_lut: wp.array2d(dtype=wp.vec3)) -> wp.vec3:
    """Sky in-scatter (single + Hillaire multiple scattering) via the LUTs."""
    atm = _ray_sphere(ro, rd, _RA)
    t0 = wp.max(atm[0], 0.0)
    t1 = atm[1]
    if t1 < 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    gnd = _ray_sphere(ro, rd, _RG)
    if gnd[0] > 0.0 and gnd[1] > 0.0:
        t1 = wp.min(t1, gnd[0])

    seg = (t1 - t0) / float(view_samples)
    od_r = float(0.0)
    od_m = float(0.0)
    sum_r = wp.vec3(0.0, 0.0, 0.0)
    sum_m = wp.vec3(0.0, 0.0, 0.0)
    sum_ms = wp.vec3(0.0, 0.0, 0.0)
    mu = wp.dot(rd, sun)
    phase_r = 3.0 / (16.0 * _PI) * (1.0 + mu * mu)
    g2 = _MIE_G * _MIE_G
    phase_m = (3.0 / (8.0 * _PI)) * ((1.0 - g2) * (1.0 + mu * mu)) \
        / ((2.0 + g2) * wp.pow(1.0 + g2 - 2.0 * _MIE_G * mu, 1.5))

    t = t0 + 0.5 * seg
    for _ in range(view_samples):
        p = ro + rd * t
        h = wp.length(p) - _RG
        hr = wp.exp(-h / _HR) * seg
        hm = wp.exp(-h / _HM) * seg
        od_r += hr
        od_m += hm
        up = wp.normalize(p)
        t_view = _vexp(-(_BETA_R * od_r + wp.vec3(_BETA_M, _BETA_M, _BETA_M) * (1.1 * od_m)))
        t_sun = transmittance_lut(lut, h, wp.dot(up, sun))
        att = wp.cw_mul(t_view, t_sun)
        sum_r += att * hr
        sum_m += att * hm
        # Hillaire multiple scattering: isotropic, no sun-phase, no t_sun
        psi = multiscatter_lut(ms_lut, h, wp.dot(up, sun))
        scat = _BETA_R * hr + wp.vec3(_BETA_M * hm, _BETA_M * hm, _BETA_M * hm)  # sigma_s * seg
        sum_ms += wp.cw_mul(psi, wp.cw_mul(scat, t_view))
        t += seg
    single = wp.cw_mul(sum_r, _BETA_R) * phase_r + sum_m * (_BETA_M * phase_m)
    return (single + sum_ms * _MS_SCALE) * _SUN_I


@wp.func
def sky_radiance_lut(ro: wp.vec3, rd: wp.vec3, sun: wp.vec3, view_samples: int,
                     lut: wp.array2d(dtype=wp.vec3),
                     ms_lut: wp.array2d(dtype=wp.vec3)) -> wp.vec3:
    col = atmosphere_lut(ro, rd, sun, view_samples, lut, ms_lut)
    mu = wp.dot(rd, sun)
    disk = wp.smoothstep(0.99985, 0.99992, mu)
    if disk > 0.0:
        gnd = _ray_sphere(ro, rd, _RG)
        if not (gnd[0] > 0.0 and gnd[1] > 0.0):
            col = col + wp.vec3(1.0, 0.95, 0.85) * (disk * _SUN_I * 18.0)
    return col
