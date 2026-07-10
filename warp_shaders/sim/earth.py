"""Earth under simultaneous global nuclear detonation — a sensitization piece.

Grounded in real numbers. Three selectable outcomes:

  grounded : real arsenal on a gravitationally-bound particle Earth. Global
             detonation flashes; the planet does NOT move — the whole arsenal is
             ~1e-13 of Earth's gravitational binding energy. The reality check.
  toxic    : the honest catastrophe — firestorm soot shrouds the planet
             (nuclear winter), the surface greys out and dims. Rock survives,
             biosphere dies.
  shatter  : HYPOTHETICAL (labeled). Weapon energy scaled to Earth's binding
             energy (~1e13x the real arsenal) — e.g. an alien 'softron' device —
             so the planet disperses into a rock + ice cloud, re-clumping under
             self-gravity.

Arsenals: current (~9,500 warheads), total (~12,500 incl. re-armed retired),
peak (~60,000, 1986). Sim runs on CPU here; identical on CUDA (RTX) at far
higher particle counts.
"""

import math

import numpy as np
import warp as wp

from ..earthgfx import earth_color
from .engine import splat_points

_SUN_DIR = np.array([-0.5, 0.28, 0.82], np.float32)
_SUN_DIR = _SUN_DIR / np.linalg.norm(_SUN_DIR)

# --- real-world constants (for the honest report) ---
G = 6.674e-11
M_EARTH = 5.972e24
R_EARTH = 6.371e6
U_BIND = 0.6 * G * M_EARTH ** 2 / R_EARTH      # ~2.24e32 J
V_ESC = math.sqrt(2 * G * M_EARTH / R_EARTH)   # ~11186 m/s
MT_J = 4.184e15                                # 1 megaton TNT in joules
CHICXULUB_J = 4.2e23                           # dinosaur-killer impact (~1e8 Mt)

# arsenal -> (warheads, total yield in megatons)
ARSENALS = {
    "current": (9500, 3000.0),
    "total": (12500, 3800.0),
    "peak": (60000, 18000.0),
}

# nuclear-power basing clusters: (name, lat, lon, share)
_CLUSTERS = [
    ("US", 40, -100, 0.32), ("Russia", 58, 60, 0.34), ("China", 35, 105, 0.07),
    ("Europe", 48, 10, 0.07), ("India", 22, 79, 0.03), ("Pakistan", 30, 69, 0.03),
    ("UK", 54, -2, 0.03), ("France", 47, 2, 0.03), ("Israel", 31, 35, 0.02),
    ("NKorea", 40, 127, 0.06),
]

# sim units: R = 1, surface gravity chosen so v_esc_sim = sqrt(2*g).
_R = 1.0
_GSURF = 0.6
_VESC_SIM = math.sqrt(2 * _GSURF * _R)


def _unit(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


@wp.kernel
def _grav(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
          dt: float, gsurf: float, radius: float):
    """Central-field self-gravity toward the center of mass (uniform-sphere g)."""
    i = wp.tid()
    p = pos[i]
    r = wp.length(p)
    if r < 1.0e-5:
        return
    if r < radius:
        g = gsurf * (r / radius)
    else:
        g = gsurf * radius * radius / (r * r)
    v = vel[i] - (p / r) * (g * dt)
    pos[i] = p + v * dt
    vel[i] = v


@wp.kernel
def _globe(img: wp.array2d(dtype=wp.vec3), width: int, height: int,
          eye: wp.vec3, right: wp.vec3, upv: wp.vec3, fwd: wp.vec3, f: float,
          sun: wp.vec3, time: float, soot: float):
    """Render the realistic Earth for a camera matching splat_points' projection."""
    i, j = wp.tid()
    cx = ((float(j) + 0.5) - 0.5 * float(width)) / f
    cy = (0.5 * float(height) - (float(i) + 0.5)) / f
    rd = wp.normalize(right * cx + upv * cy + fwd)
    img[i, j] = earth_color(eye, rd, sun, time, soot)


def _render_globe(width, height, eye, target, fov_deg, sun, time, soot, device):
    fwd = np.asarray(target, np.float32) - np.asarray(eye, np.float32)
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, np.array([0, 1, 0], np.float32))
    right /= np.linalg.norm(right) + 1e-9
    upv = np.cross(right, fwd)
    f = (height * 0.5) / np.tan(np.radians(fov_deg) * 0.5)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_globe, dim=(height, width),
              inputs=[img, width, height, wp.vec3(*[float(x) for x in eye]),
                      wp.vec3(*[float(x) for x in right]), wp.vec3(*[float(x) for x in upv]),
                      wp.vec3(*[float(x) for x in fwd]), float(f),
                      wp.vec3(*[float(x) for x in sun]), float(time), float(soot)],
              device=device)
    wp.synchronize_device(device)
    return img.numpy()


def build_earth(n, rng):
    """A volume-filled particle Earth with surface ocean/land/ice + hot interior."""
    rad = np.cbrt(rng.random(n)).astype(np.float32) * _R
    dirs = _unit(rng.normal(size=(n, 3))).astype(np.float32)
    pos = dirs * rad[:, None]
    is_surface = rad > 0.9

    col = np.zeros((n, 3), np.float32)
    # Interior: mantle -> hot core toward the center.
    core = np.array([1.0, 0.45, 0.12], np.float32)
    mantle = np.array([0.45, 0.18, 0.08], np.float32)
    t = np.clip(1.0 - rad / 0.9, 0.0, 1.0)[:, None]
    col[:] = mantle + (core - mantle) * t

    # Surface: continents from low-frequency lat/lon pattern, ice at the poles.
    d = dirs[is_surface]
    lat = np.arcsin(np.clip(d[:, 1], -1, 1))
    lon = np.arctan2(d[:, 2], d[:, 0])
    cont = (np.sin(3 * lon + 1) * np.cos(2 * lat)
            + 0.5 * np.sin(5 * lon) * np.sin(3 * lat + 2)
            + rng.normal(size=len(d)) * 0.15)
    ocean = np.array([0.06, 0.22, 0.5], np.float32)
    land = np.array([0.2, 0.38, 0.15], np.float32)
    ice = np.array([0.85, 0.9, 1.0], np.float32)
    scol = np.tile(ocean, (len(d), 1))
    scol[cont > 0.2] = land
    scol[np.abs(d[:, 1]) > 0.82] = ice
    col[is_surface] = scol

    return pos, col, rad, is_surface, dirs


def make_sites(nsites, rng):
    w = np.array([c[3] for c in _CLUSTERS], np.float64)
    w /= w.sum()
    counts = rng.multinomial(nsites, w)
    lats, lons = [], []
    for (name, lat, lon, share), c in zip(_CLUSTERS, counts):
        lats.append(lat + rng.normal(size=c) * 11)
        lons.append(lon + rng.normal(size=c) * 15)
    lat = np.radians(np.concatenate(lats))
    lon = np.radians(np.concatenate(lons))
    dirs = np.stack([np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon)], 1)
    return (dirs * 0.985).astype(np.float32)   # slightly sub-surface


def report_for(arsenal):
    warheads, mt = ARSENALS[arsenal]
    e = mt * MT_J
    phi = e / U_BIND
    return {
        "arsenal": arsenal, "warheads": warheads, "yield_Mt": mt,
        "energy_J": e, "binding_J": U_BIND, "ratio_of_binding": phi,
        "dv_over_vesc": math.sqrt(phi), "vesc_m_s": V_ESC,
        "chicxulub_x_arsenal": CHICXULUB_J / e,
    }


def simulate_earth(arsenal="total", outcome="grounded", frames=130, n=30000,
                   width=640, height=400, device="cpu", seed=3):
    rng = np.random.default_rng(seed)
    pos, base_col, rad, is_surface, dirs = build_earth(n, rng)
    sites = make_sites(min(ARSENALS[arsenal][0], 700), rng)
    rep = report_for(arsenal)

    d_pos = wp.array(pos, dtype=wp.vec3, device=device)
    d_vel = wp.zeros(n, dtype=wp.vec3, device=device)

    det_frame = 20
    dt = 0.05
    sun = np.array([0.6, 0.5, 0.55], np.float32)
    sun /= np.linalg.norm(sun)

    detonated = False
    flash = 0.0
    toxic_prog = 0.0
    shroud = np.zeros((0, 3), np.float32)
    max_dv = 0.0
    frames_out = []

    for f in range(frames):
        if f == det_frame:
            detonated = True
            flash = 3.2
            if outcome == "shatter":
                # Hypothetical: energy ~ binding -> radial kick. A wide spread means
                # inner/low-kick debris stays below escape and falls back into
                # clumps, outer debris escapes: a shambling rock + ice cloud.
                nrm = _unit(pos)
                mult = 1.1
                kick = _VESC_SIM * math.sqrt(mult) * (0.55 + 0.6 * rng.random((n, 1)))
                d_vel.assign((nrm * kick).astype(np.float32))
                max_dv = float(kick.max())

        # physics: only the shatter cloud evolves (intact Earth is in equilibrium)
        if detonated and outcome == "shatter":
            for _ in range(2):
                wp.launch(_grav, dim=n, inputs=[d_pos, d_vel, dt * 0.5, _GSURF, _R], device=device)
            wp.synchronize_device(device)
            pos = d_pos.numpy()

        # --- camera ---
        az = 0.5 + 0.012 * f
        dist = 3.1
        if outcome == "shatter" and detonated:
            rad80 = float(np.percentile(np.linalg.norm(pos, axis=1), 80))
            dist = max(3.1, 2.1 * rad80)   # track the expanding cloud
        eye = np.array([dist * math.sin(az), 0.7, dist * math.cos(az)], np.float32)
        target = np.array([0, 0, 0], np.float32)

        # --- base frame per outcome ---
        if outcome == "shatter":
            # Dispersed debris: render the particle cloud (a solid globe can't
            # shatter). Fresh debris glows hot and fades as it expands.
            col = base_col.copy()
            bright = np.full(n, 0.9, np.float32)
            if detonated:
                glow = np.clip(1.4 - 0.03 * (f - det_frame), 0.0, 1.0)
                col = col + np.array([1.0, 0.5, 0.2], np.float32) * glow * 0.5
            frame = splat_points(width, height, pos, col, bright, eye, target,
                                 fov_deg=40.0, stamp_radius=2)
        else:
            # Intact planet: render the realistic globe (same shader as the
            # `earth` scene). toxic greys/dims the surface (nuclear winter).
            if outcome == "toxic" and detonated:
                toxic_prog = min(1.0, toxic_prog + 0.02)
            frame = _render_globe(width, height, eye, target, 40.0,
                                  _SUN_DIR, f * dt, toxic_prog, device) * 1.7

        # --- detonation flashes at every site ---
        if flash > 0.03:
            fcol = np.tile(np.array([1.0, 0.95, 0.8], np.float32), (len(sites), 1))
            fb = np.full(len(sites), flash, np.float32)
            frame = frame + splat_points(width, height, sites * 1.0, fcol, fb,
                                         eye, target, fov_deg=40.0, stamp_radius=3)
            flash *= 0.62

        # --- toxic soot shroud accumulates around the globe ---
        if outcome == "toxic" and detonated and len(shroud) < 16000:
            k = 380
            sd = _unit(rng.normal(size=(k, 3)))
            new = (sd * (1.0 + rng.random((k, 1)) * 0.22)).astype(np.float32)
            shroud = np.concatenate([shroud, new], 0)
        if len(shroud):
            scol = np.tile(np.array([0.22, 0.2, 0.19], np.float32), (len(shroud), 1))
            sb = np.full(len(shroud), 0.10, np.float32)
            frame = frame + splat_points(width, height, shroud, scol, sb,
                                         eye, target, fov_deg=40.0, stamp_radius=3)

        frames_out.append(frame)

    rep["outcome"] = outcome
    rep["max_blast_dv_sim"] = max_dv
    if outcome == "grounded":
        rep["verdict"] = ("PLANET INTACT. The arsenal is ~%.0e of Earth's binding "
                          "energy; the dinosaur-killer impact was ~%.0fx larger and "
                          "Earth survived geologically." % (rep["ratio_of_binding"],
                                                            rep["chicxulub_x_arsenal"]))
    elif outcome == "toxic":
        rep["verdict"] = ("PLANET INTACT, BIOSPHERE DEAD. Rock survives; firestorm "
                          "soot -> nuclear winter, fallout, ozone loss. Uninhabitable.")
    else:
        rep["verdict"] = ("HYPOTHETICAL (not real physics): energy scaled to ~1e13x "
                          "the real arsenal (>= Earth's binding energy) to disperse "
                          "the planet into a rock + ice cloud.")
    return frames_out, rep
