"""Nuclear-detonation physics — declassified scaling laws (Glasstone & Dolan).

Pure functions (NumPy host + a few Warp device ``@wp.func``) that size a nuclear
explosion from its **yield**: the fireball, the blast-overpressure damage rings,
the thermal-burn radius, the Sedov–Taylor shock front, the mushroom-cloud rise,
and the blackbody cooling of the fireball. Everything is calibrated to the
measured **Tsar Bomba** (50 Mt) test so the numbers are life-like, not arbitrary.

See ``docs/research/15-nuclear-fireball.md`` for the derivations + citations.
These are textbook civil-defence relations — a visualisation of published
physics, not weapon-design data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

# --- constants --------------------------------------------------------------
KT_J = 4.184e12               # joules per kiloton of TNT
HIROSHIMA_KT = 13.0           # "Little Boy" ~= 13 kt
TSAR_KT = 5.0e4               # Tsar Bomba tested yield: 50 Mt
SUPER_TSAR_KT = 5.0e5         # hypothetical x10: 500 Mt
RHO_AIR = 1.2                 # kg/m^3 at sea level

# calibrated constants (see research doc — anchored to the 50 Mt Tsar test)
_FIREBALL_C = 46.0            # R_fireball = 46 * W^0.4  m   -> 3.5 km at 50 Mt
_THERMAL_C = 1316.0           # R_thermal  = 1316 * W^0.4 m  -> 100 km at 50 Mt
_SEDOV_XI = 1.03             # Sedov-Taylor constant for air (gamma = 1.4)

# overpressure ring reference radii R0 (km per kt^(1/3)); cube-root scaling
_PSI_RINGS = {20.0: 0.28, 5.0: 1.03, 1.0: 2.93}


# --- host scaling laws ------------------------------------------------------
def fireball_radius(w_kt):
    """Maximum luminous fireball radius (m). Scales ~ W^0.4."""
    return _FIREBALL_C * np.power(w_kt, 0.4)


def thermal_radius(w_kt):
    """Radius (m) for third-degree burns (~W^0.4; 100 km at 50 Mt)."""
    return _THERMAL_C * np.power(w_kt, 0.4)


def overpressure_radius(w_kt, psi):
    """Radius (m) of a given peak overpressure (psi) by cube-root scaling.

    Interpolates the canonical 20 / 5 / 1 psi reference radii as a power law."""
    # power fit R0(psi) = 2.93 * psi^(-0.783) km through the three anchors
    r0_km = 2.93 * np.power(psi, -0.783)
    return r0_km * 1000.0 * np.power(w_kt, 1.0 / 3.0)


def destruction_radius(w_kt):
    """Total-destruction radius (m) — the 5 psi contour (~35 km at 50 Mt)."""
    return _PSI_RINGS[5.0] * 1000.0 * np.power(w_kt, 1.0 / 3.0)


def severe_radius(w_kt):
    """20 psi contour (m) — reinforced concrete destroyed."""
    return _PSI_RINGS[20.0] * 1000.0 * np.power(w_kt, 1.0 / 3.0)


def light_radius(w_kt):
    """1 psi contour (m) — window breakage, injuries."""
    return _PSI_RINGS[1.0] * 1000.0 * np.power(w_kt, 1.0 / 3.0)


def shock_radius(t, w_kt, rho=RHO_AIR):
    """Sedov–Taylor blast-front radius (m) at time `t` (s). ~ t^(2/5)."""
    e = w_kt * KT_J
    return _SEDOV_XI * np.power(e * t * t / rho, 0.2)


def mushroom_height(t, w_kt, tau=18.0):
    """Rising mushroom-cap top altitude (m) — saturating buoyant rise.

    Ceiling scales weakly with yield (atmospheric stability limited); 67 km at
    50 Mt."""
    h_max = 67000.0 * np.power(w_kt / TSAR_KT, 0.2)
    return h_max * (1.0 - np.exp(-t / tau))


def fireball_temp(t_norm):
    """Effective blackbody temperature (K) of the fireball vs normalised age
    `t_norm` in [0, 1] — from ~30,000 K blue-white to ~1,500 K dull red."""
    return 1500.0 + 28500.0 * np.exp(-3.0 * np.clip(t_norm, 0.0, 1.0))


def debris_shell_radius(t, w_kt, m_debris_kg=1.0e4):
    """Vacuum burst: ballistic plasma/debris shell radius (m), R ~ v·t with
    v = sqrt(2E/m). No blast wave, no fireball, no mushroom (see research doc)."""
    e = w_kt * KT_J
    v = math.sqrt(2.0 * e / m_debris_kg)
    return v * np.asarray(t, dtype=float)


@dataclass
class BlastParams:
    """All static effect radii (metres) for a named device of a given yield."""
    name: str
    yield_kt: float

    @property
    def fireball(self):
        return float(fireball_radius(self.yield_kt))

    @property
    def thermal(self):
        return float(thermal_radius(self.yield_kt))

    @property
    def destruction(self):
        return float(destruction_radius(self.yield_kt))

    @property
    def severe(self):
        return float(severe_radius(self.yield_kt))

    @property
    def light(self):
        return float(light_radius(self.yield_kt))

    @property
    def hiroshimas(self):
        return self.yield_kt / HIROSHIMA_KT

    def summary(self):
        return (f"{self.name}: {self.yield_kt/1000:.0f} Mt "
                f"(~{self.hiroshimas:.0f}x Hiroshima) | fireball "
                f"{self.fireball/1000:.1f} km | destruction "
                f"{self.destruction/1000:.0f} km | thermal "
                f"{self.thermal/1000:.0f} km")


TSAR = BlastParams("Tsar Bomba", TSAR_KT)
SUPER_TSAR = BlastParams("Super Tsar", SUPER_TSAR_KT)


# --- device helpers (per-sample, inside the render kernels) -----------------
@wp.func
def fireball_temp_at(core_k: float, r_norm: float) -> float:
    """Blackbody temperature (K) across the fireball: hot core -> cool rim.
    `r_norm` is radius / fireball_radius in [0, 1]."""
    f = wp.max(1.0 - r_norm * r_norm, 0.0)
    return 1200.0 + (core_k - 1200.0) * f * f


@wp.func
def smoothstep(a: float, b: float, x: float) -> float:
    t = wp.clamp((x - a) / (b - a + 1.0e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def blast_falloff(r: float, r_shock: float, width: float) -> float:
    """Thin bright shell at the shock front `r_shock` (for the condensation
    ring): a Gaussian in `r` around the front of the given `width`."""
    d = (r - r_shock) / (width + 1.0e-6)
    return wp.exp(-d * d)


@wp.func
def shock_ring(dist: float, ring_r: float, core_w: float, glow_w: float) -> float:
    """Layered shockwave-ring intensity at horizontal `dist` from ground zero:
    a sharp bright **core** at the front `ring_r`, an **inner glow**, and a soft
    quadratic **outer glow**. Ported from ``the-virus-block-mc``'s
    ``shockwave_ring.glsl`` (``ringContribution``) — gives the shock front real
    structure (a crisp leading edge trailing into a halo) instead of a plain
    Gaussian."""
    dr = wp.abs(dist - ring_r)
    core = 1.0 - smoothstep(0.0, core_w * 0.5, dr)
    inner = 1.0 - smoothstep(0.0, glow_w * 0.5, dr)
    tt = dr / (glow_w + 1.0e-6)
    outer = wp.max(0.0, 1.0 - tt * tt)
    return core * 0.9 + inner * 0.4 + outer * 0.2
