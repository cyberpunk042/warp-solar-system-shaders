"""One star across its whole life — a normalized-time evolution timeline.

`phase_state(t, mass)` is a pure host function: given a normalized time
``t in [0, 1]`` and an initial mass (solar masses), it returns the star's
appearance now — body **kind**, **radius**, **temperature**, **activity**, an
**envelope** descriptor (protostar cradle / planetary nebula / supernova ejecta),
a supernova **flash** intensity, and the current **HR-diagram** coordinates +
**phase name**. `render_lifecycle(t, mass, ...)` reuses the star library from
:mod:`.bodies` to draw the evolving star (envelopes + the HR inset are layered on
in later phases).

The timeline is compressed by *visual interest*, not real duration — every phase
gets screen-time and the fast, dramatic ends get room. See
``docs/research/11-stellar-evolution.md``. Physics anchors (HR track, mass fork,
timescales) are cited there.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3, ridged3
from .bodies import (BLACK_HOLE, NEUTRON, SUN, WHITE_DWARF, StarConfig,
                     body_corona, make_star, neutron_axis, pulsar_beams,
                     render_star, shade_body)
from .blackhole import make_black_hole, render_black_hole

# envelope kinds (int, for the kernel)
ENV_NONE = 0
ENV_CRADLE = 1        # protostar birth cloud (filled)
ENV_PLANETARY = 2     # planetary nebula (expanding shell)
ENV_SUPERNOVA = 3     # supernova ejecta (expanding shell)

_ENV_CODE = {"none": ENV_NONE, "cradle": ENV_CRADLE,
             "planetary": ENV_PLANETARY, "supernova": ENV_SUPERNOVA}

# ZAMS mass boundaries for the end state (solar masses) — standard division.
WD_MAX = 8.0        # below -> white dwarf
NS_MAX = 20.0       # below -> neutron star, above -> black hole


@dataclasses.dataclass
class PhaseState:
    phase: str                     # human-readable phase name
    kind: int = SUN                # bodies.SUN / NEUTRON / WHITE_DWARF / BLACK_HOLE
    radius: float = 1.0            # world radius (MS sun ~ 1.0)
    temp: float = 0.55             # 0 cool-red .. 1 hot-blue (feeds the blackbody ramp)
    activity: float = 0.5
    spin: float = 1.0
    precess: float = 0.0
    env: str = "none"              # "none" / "cradle" / "planetary" / "supernova"
    env_radius: float = 0.0        # expanding shell radius (envelope)
    env_intensity: float = 0.0
    flash: float = 0.0             # supernova flash spike (0..~2)
    hr_temp: float = 0.55          # HR x: 0 cool .. 1 hot
    hr_lum: float = 0.5            # HR y: 0 faint .. 1 luminous (log)


def _lerp(u, a, b):
    return a + (b - a) * u


# Each phase: (t_start, t_end, dict of anchor params). Continuous fields are
# lerped from *_a (start) to *_b (end) across the phase; discrete fields (kind,
# env, phase name) are constant within the phase. Fields absent take defaults.
def _low_mass_phases():
    return [
        (0.00, 0.12, dict(name="protostar (T Tauri)", kind=SUN, env="cradle",
                          r=(1.9, 1.2), temp=(0.10, 0.32), act=(1.0, 0.7),
                          envi=(1.1, 0.5), hrt=(0.28, 0.5), hrl=(0.78, 0.55))),
        (0.12, 0.48, dict(name="main sequence", kind=SUN, env="none",
                          r=(1.0, 1.0), temp=(0.55, 0.55), act=(0.5, 0.5),
                          hrt=(0.55, 0.55), hrl=(0.5, 0.5))),
        (0.48, 0.56, dict(name="subgiant", kind=SUN, env="none",
                          r=(1.0, 1.65), temp=(0.55, 0.40), act=(0.5, 0.6),
                          hrt=(0.55, 0.42), hrl=(0.5, 0.6))),
        (0.56, 0.72, dict(name="red giant", kind=SUN, env="none",
                          r=(1.65, 2.7), temp=(0.40, 0.12), act=(0.6, 0.8),
                          hrt=(0.42, 0.15), hrl=(0.6, 0.85))),
        (0.72, 0.80, dict(name="helium flash / horizontal branch", kind=SUN, env="none",
                          r=(2.7, 1.5), temp=(0.13, 0.44), act=(0.9, 0.6),
                          hrt=(0.15, 0.46), hrl=(0.85, 0.6), flash_head=0.35)),
        (0.80, 0.90, dict(name="asymptotic giant branch", kind=SUN, env="none",
                          r=(1.5, 3.0), temp=(0.42, 0.12), act=(0.8, 1.0),
                          hrt=(0.46, 0.13), hrl=(0.6, 0.92), pulse=0.22)),
        (0.90, 0.955, dict(name="planetary nebula", kind=WHITE_DWARF, env="planetary",
                           r=(2.6, 0.18), temp=(0.30, 0.92), act=(0.5, 0.3),
                           envi=(1.2, 0.7), envr=(0.4, 3.6), hrt=(0.2, 0.95),
                           hrl=(0.9, 0.35))),
        (0.955, 1.001, dict(name="white dwarf (cooling)", kind=WHITE_DWARF, env="planetary",
                            r=(0.17, 0.14), temp=(0.92, 0.60), act=(0.25, 0.15),
                            envi=(0.6, 0.0), envr=(3.6, 5.2), hrt=(0.95, 0.72),
                            hrl=(0.35, 0.05))),
    ]


def _high_mass_phases(remnant_kind):
    rem_name = "neutron star (pulsar)" if remnant_kind == NEUTRON else "black hole"
    return [
        (0.00, 0.08, dict(name="protostar", kind=SUN, env="cradle",
                          r=(2.3, 1.6), temp=(0.30, 0.72), act=(1.0, 0.7),
                          envi=(1.1, 0.4), hrt=(0.4, 0.72), hrl=(0.85, 0.78))),
        (0.08, 0.42, dict(name="main sequence (O/B)", kind=SUN, env="none",
                          r=(1.6, 1.7), temp=(0.90, 0.90), act=(0.6, 0.6),
                          hrt=(0.9, 0.9), hrl=(0.82, 0.82))),
        (0.42, 0.68, dict(name="red supergiant", kind=SUN, env="none",
                          r=(1.7, 3.3), temp=(0.90, 0.12), act=(0.6, 0.85),
                          hrt=(0.9, 0.14), hrl=(0.82, 0.97), pulse=0.10)),
        # the core collapses to a point during the flash; the ejecta then expands
        (0.68, 0.74, dict(name="supernova (Type II)", kind=SUN, env="supernova",
                          r=(0.55, 0.10), temp=(0.85, 1.0), act=(1.0, 1.0),
                          envi=(1.1, 0.55), envr=(0.5, 4.6), hrt=(0.14, 0.9),
                          hrl=(0.97, 0.5), flash_head=0.5)),
        (0.74, 1.001, dict(name=rem_name, kind=remnant_kind, env="supernova",
                           r=(0.12, 0.12), temp=(1.0, 1.0), act=(0.9, 0.7),
                           envi=(0.7, 0.0), envr=(4.2, 6.5), hrt=(0.9, 0.85),
                           hrl=(0.5, 0.12), precess=0.6)),
    ]


def remnant_kind(mass: float) -> int:
    """The end-state body kind for an initial `mass` (solar masses)."""
    if mass < WD_MAX:
        return WHITE_DWARF
    if mass < NS_MAX:
        return NEUTRON
    return BLACK_HOLE


def phase_state(t: float, mass: float = 1.0) -> PhaseState:
    """The star's appearance at normalized time `t` for an initial `mass`."""
    t = float(np.clip(t, 0.0, 1.0))
    phases = _low_mass_phases() if mass < WD_MAX else _high_mass_phases(remnant_kind(mass))
    seg = phases[-1]
    for p in phases:
        if t < p[1]:
            seg = p
            break
    t0, t1, d = seg
    u = (t - t0) / max(t1 - t0, 1e-9)
    u = float(np.clip(u, 0.0, 1.0))

    st = PhaseState(phase=d["name"], kind=int(d["kind"]), env=d.get("env", "none"))
    st.radius = _lerp(u, *d["r"])
    st.temp = float(np.clip(_lerp(u, *d["temp"]), 0.0, 1.0))
    st.activity = _lerp(u, *d["act"])
    st.hr_temp = _lerp(u, *d["hrt"])
    st.hr_lum = _lerp(u, *d["hrl"])
    st.precess = d.get("precess", 0.0)
    if "envi" in d:
        st.env_intensity = _lerp(u, *d["envi"])
    if "envr" in d:
        st.env_radius = _lerp(u, *d["envr"])
    elif st.env == "cradle":
        st.env_radius = 3.4                          # a fixed cradle around the protostar
    # thermal pulses (AGB / supergiant): ripple radius + activity
    if "pulse" in d:
        ph = math.sin(u * math.pi * 7.0)
        st.radius *= 1.0 + d["pulse"] * ph
        st.activity = min(1.2, st.activity + 0.2 * abs(ph))
    # supernova / helium-flash spike near the phase head
    if "flash_head" in d:
        head = d["flash_head"]
        if u < head:
            st.flash = (1.0 - u / head) ** 2 * (2.0 if st.env == "supernova" else 0.8)
    return st


# --------------------------------------------------------------------------- #
# envelope (birth cradle + expanding shells) — emissive volume integration     #
# --------------------------------------------------------------------------- #

@wp.func
def _env_at(p: wp.vec3, ekind: int, radius: float, seed: float,
            time: float) -> wp.vec4:
    """Emission colour + density of the envelope at `p` (origin-centred)."""
    s = wp.vec3(seed, seed * 1.7, seed * 2.3)
    r = wp.length(p)
    if ekind == ENV_CRADLE:                          # filled molecular cloud
        rn = r / wp.max(radius, 1e-4)
        if rn > 1.0:
            return wp.vec4(0.0, 0.0, 0.0, 0.0)
        d = p * (1.0 / wp.max(radius, 1e-4))
        base = fbm3(d * 2.2 + s + wp.vec3(time * 0.02, 0.0, 0.0), 5)
        fil = ridged3(d * 4.6 + s, 4)
        dens = wp.clamp((base * 0.6 + fil * 0.5 - 0.52) * 1.9, 0.0, 1.0)
        dens = dens * wp.smoothstep(1.0, 0.2, rn)
        warm = wp.smoothstep(0.6, 0.0, rn)           # lit warm by the protostar
        cool = wp.vec3(0.34, 0.18, 0.46)             # dusty violet
        glow = wp.vec3(0.95, 0.52, 0.34)             # warm dust near the star
        col = cool * (1.0 - warm) + glow * warm
        return wp.vec4(col[0], col[1], col[2], dens)
    # expanding shell (planetary nebula / supernova): a thin, patchy gaussian
    # bubble at |p| = radius — translucent coloured gas with voids, not a wash
    thick = 0.12 * radius + 0.05
    dd = (r - radius) / thick
    gauss = wp.exp(-dd * dd)
    if gauss < 0.004:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)
    dir = p * (1.0 / wp.max(r, 1e-4))
    turb = fbm3(dir * 5.0 + s + wp.vec3(time * 0.05, 0.0, 0.0), 5)
    fil = ridged3(dir * 10.0 + s, 4)
    dens = gauss * wp.clamp(0.55 * turb + 0.85 * fil - 0.18, 0.0, 1.0)
    k = wp.clamp(0.5 + 0.5 * dd, 0.0, 1.0)           # 0 inside the shell .. 1 outside
    if ekind == ENV_PLANETARY:
        inner = wp.vec3(0.15, 0.85, 0.78)            # O III teal (ionized inner)
        outer = wp.vec3(0.95, 0.26, 0.44)            # H-alpha rose (outer)
        col = inner * (1.0 - k) + outer * k
    else:                                            # supernova: fire + a thin shock
        interior = wp.vec3(1.0, 0.32, 0.09)          # incandescent orange-red ejecta
        shock = wp.vec3(0.5, 0.72, 1.0)              # blue-white leading shock
        kk = wp.smoothstep(0.6, 1.0, k)              # blue only at the very outer edge
        col = interior * (1.0 - kk) + shock * kk
    return wp.vec4(col[0], col[1], col[2], dens)


@wp.func
def _march_env(ro: wp.vec3, rd: wp.vec3, ekind: int, radius: float,
               intensity: float, seed: float, time: float) -> wp.vec3:
    """Front-to-back emission integral of the envelope along the ray."""
    if ekind == ENV_NONE or intensity <= 0.0 or radius <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    bound = radius
    if ekind != ENV_CRADLE:
        bound = radius * 1.4
    b = wp.dot(ro, rd)
    c = wp.dot(ro, ro) - bound * bound
    disc = b * b - c
    if disc < 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    sq = wp.sqrt(disc)
    t0 = wp.max(-b - sq, 0.0)
    t1 = -b + sq
    if t1 <= t0:
        return wp.vec3(0.0, 0.0, 0.0)
    steps = 44
    seg = (t1 - t0) / float(steps)
    t = t0 + 0.5 * seg
    acc = wp.vec3(0.0, 0.0, 0.0)
    trans = float(1.0)
    gain = 0.9
    if ekind == ENV_CRADLE:
        gain = 1.1
    for _ in range(steps):
        p = ro + rd * t
        nv = _env_at(p, ekind, radius, seed, time)
        dn = nv[3] * seg * gain
        if dn > 0.001:
            em = wp.vec3(nv[0], nv[1], nv[2])
            acc = acc + em * (dn * trans)
            trans = trans * (1.0 - wp.clamp(dn, 0.0, 1.0))
            if trans < 0.03:
                break
        t += seg
    return acc * intensity


@wp.kernel
def _lifecycle_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, cfg: StarConfig,
                      ekind: int, env_radius: float, env_intensity: float,
                      seed: float, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)

    col = stars(rd)
    R = cfg.radius
    b = wp.dot(ro, rd)
    c = wp.dot(ro, ro) - R * R
    disc = b * b - c
    tca = -b
    hit = int(0)
    if disc > 0.0:
        tt = -b - wp.sqrt(disc)
        if tt > 0.0:
            p = ro + rd * tt
            col = shade_body(wp.normalize(p), rd, cfg, time)
            hit = 1
    if hit == 0 and tca > 0.0:
        d = wp.sqrt(wp.max(wp.dot(ro, ro) - tca * tca, 0.0))
        col = col + body_corona(d / R, cfg, time)
    if cfg.kind == NEUTRON:
        ax = neutron_axis(cfg, time)
        col = col + pulsar_beams(ro, rd, ax, cfg, time)

    # emissive envelope glows over everything (near gas is in front of the star)
    env = _march_env(ro, rd, ekind, env_radius, env_intensity, seed, time)
    col = col + env
    img[i, j] = col


# --------------------------------------------------------------------------- #
# host render                                                                  #
# --------------------------------------------------------------------------- #

def render_lifecycle(t: float, mass: float = 1.0, width: int = 640, height: int = 400,
                     device: str = "cpu", dist: float = 6.5, fov: float = 40.0,
                     anim: float = 20.0):
    """Render the evolving star + its envelope at time `t` (HR inset lands in
    SL3). Reuses the star library, the envelope integrator, and — for a
    black-hole remnant — the lensing pass."""
    st = phase_state(t, mass)

    if st.kind == BLACK_HOLE:
        # the collapsed core lenses light; draw it with the accretion-disk pass.
        bh = make_black_hole(radius=0.6, activity=0.8, spin=1.0, seed=7.0)
        return render_black_hole(bh, width, height, time=t * anim, device=device,
                                 dist=11.0, fov=fov)

    cfg = make_star(kind=st.kind, radius=st.radius, temp=st.temp,
                    activity=st.activity, spin=1.0, precess=st.precess, seed=7.0)
    ekind = _ENV_CODE.get(st.env, ENV_NONE)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_lifecycle_kernel, dim=(height, width),
              inputs=[img, make_camera((dist * 0.15, dist * 0.12, dist),
                                       (0.0, 0.0, 0.0), fov_deg=fov,
                                       aspect=width / height),
                      cfg, int(ekind), float(st.env_radius), float(st.env_intensity),
                      7.0, float(t * anim), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    # supernova / helium flash: a global whiteout burst that bloom smears wide
    if st.flash > 0.0:
        hdr = hdr + np.float32([1.0, 0.95, 0.85]) * (st.flash * 0.7)
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.5 + 0.9 * st.flash, radius=r,
                     passes=4)
    return post.tonemap(hdr, mode="aces", exposure=1.02 + 0.5 * st.flash)
