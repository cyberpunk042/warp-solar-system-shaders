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

from .bodies import BLACK_HOLE, NEUTRON, SUN, WHITE_DWARF, make_star, render_star

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
        (0.68, 0.74, dict(name="supernova (Type II)", kind=SUN, env="supernova",
                          r=(3.3, 0.12), temp=(0.12, 1.0), act=(1.0, 1.0),
                          envi=(1.6, 0.8), envr=(0.3, 4.2), hrt=(0.14, 0.9),
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


def render_lifecycle(t: float, mass: float = 1.0, width: int = 640, height: int = 400,
                     device: str = "cpu", dist: float = 6.5, fov: float = 40.0,
                     anim: float = 20.0):
    """Render the evolving **star** at time `t` (envelopes + HR inset land in the
    later phases). Reuses ``bodies.render_star`` with the phase's StarConfig."""
    st = phase_state(t, mass)
    if st.kind == BLACK_HOLE:
        # the black-hole remnant is drawn by the lensing pass (added in SL2); for
        # now fall back to a dim compact core so the star path still renders.
        cfg = make_star(kind=WHITE_DWARF, radius=0.12, temp=1.0, activity=0.2,
                        seed=7.0)
    else:
        cfg = make_star(kind=st.kind, radius=st.radius, temp=st.temp,
                        activity=st.activity, spin=1.0, precess=st.precess, seed=7.0)
    bloom = 0.7 + 0.9 * st.flash
    return render_star(cfg, width, height, time=t * anim, device=device,
                       dist=dist, fov=fov, bloom_strength=bloom)
