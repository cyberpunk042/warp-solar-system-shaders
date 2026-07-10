"""Nuclear + thermonuclear blast simulation on top of the particle engine.

Timeline of a run:

  1. drop      — the device falls under gravity (real ballistic motion) until it
                 reaches the burst altitude.
  2. chain     — a fission chain reaction, modelled with self-limiting point
                 kinetics: one seed neutron multiplies (k_eff > 1) until the fuel
                 burns up and reactivity falls below critical, giving the
                 characteristic neutron-population *pulse*. Visualised as a swarm
                 of neutron sparks in the core.
  3. fireball  — the released energy spawns a hot particle fireball that expands,
                 then rises by buoyancy against gravity + drag -> mushroom cloud.

A thermonuclear run adds a second stage: once the fission *primary* releases
enough energy it *ignites* a fusion *secondary* (Teller–Ulam idea), a second,
much larger pulse and fireball (~25x the yield here).

The physics timescale is dramatised (a real chain reaction is ~1e-8 s per
generation); this maps generations onto animation frames so the cascade is
visible. Energies are in arbitrary units for comparison, not megatons.
"""

import math

import numpy as np

from .engine import ParticleSystem

G = 9.8


def _unit(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


class _Stage:
    """Self-limiting point-kinetics for one fission/fusion stage."""

    def __init__(self, k0, rate, burn_rate, e_per_burn):
        self.k0 = k0
        self.rate = rate
        self.burn_rate = burn_rate
        self.e_per_burn = e_per_burn
        self.n = 0.0          # neutron population (arb.)
        self.fuel = 1.0       # fraction remaining
        self.energy = 0.0
        self.active = False
        self.peak_n = 0.0
        self.gens = 0

    def ignite(self, seed=1.0):
        self.n = seed
        self.active = True

    def step(self, dt):
        if not self.active:
            return 0.0
        # When fuel is gone k_eff -> 0, so the population decays away (sparks fade).
        k_eff = self.k0 * self.fuel
        self.n *= math.exp((k_eff - 1.0) * self.rate * dt)
        self.n = min(self.n, 1e12)
        self.peak_n = max(self.peak_n, self.n)
        d_burn = min(self.n * self.burn_rate * dt, self.fuel)
        self.fuel -= d_burn
        de = d_burn * self.e_per_burn
        self.energy += de
        self.gens += 1
        return de


def _spawn_fireball(ps, center, count, speed, temp, life, rng, up_bias=0.35):
    d = _unit(rng.normal(size=(count, 3)))
    r = rng.random((count, 1))
    vel = d * (speed * (0.4 + 0.6 * r))
    vel[:, 1:2] += speed * up_bias
    pos = center + d * (rng.random((count, 1)) * 0.4)
    # Faster / outer particles spawn cooler -> a hot core with cooler edges.
    tval = np.clip(temp * (1.2 - 0.55 * r[:, 0]), 0.0, 1.2)
    ps.spawn(pos.astype(np.float32), vel.astype(np.float32),
             tval.astype(np.float32), np.full(count, life, np.float32),
             np.full(count, 2, np.int32))


def _spawn_neutrons(ps, center, count, rng):
    if count <= 0:
        return
    d = _unit(rng.normal(size=(count, 3)))
    vel = d * (6.0 + 8.0 * rng.random((count, 1)))
    pos = center + rng.normal(size=(count, 3)) * 0.18
    ps.spawn(pos.astype(np.float32), vel.astype(np.float32),
             np.ones(count, np.float32), np.full(count, 0.14, np.float32),
             np.ones(count, np.int32))


def _spawn_column(ps, top_y, count, rng, temp=0.7):
    """A rising stem filling ground -> cap along the axis."""
    y = rng.random(count) * top_y
    j = rng.normal(size=(count, 3)) * np.array([0.28, 0.0, 0.28])
    pos = j.astype(np.float32)
    pos[:, 1] = y
    vel = np.zeros((count, 3), np.float32)
    vel[:, 1] = 3.0 + 3.0 * rng.random(count)
    ps.spawn(pos, vel, np.full(count, temp, np.float32),
             np.full(count, 6.0, np.float32), np.full(count, 2, np.int32))


def _spawn_base_surge(ps, count, speed, rng):
    """A dust skirt spreading radially outward along the ground."""
    ang = rng.random(count) * 6.2831
    d = np.stack([np.cos(ang), np.zeros(count), np.sin(ang)], 1)
    pos = (d * (0.2 + rng.random((count, 1)) * 0.3)).astype(np.float32)
    pos[:, 1] = 0.12
    vel = (d * speed * (0.5 + 0.5 * rng.random((count, 1)))).astype(np.float32)
    vel[:, 1] += 0.6
    ps.spawn(pos, vel, np.full(count, 0.32, np.float32),
             np.full(count, 6.0, np.float32), np.full(count, 2, np.int32))


def simulate(scenario="nuclear", drop=True, frames=100, dt=0.045,
             width=480, height=270, device="cpu", seed=1):
    """Run a blast. Returns (frames_list, report_dict)."""
    rng = np.random.default_rng(seed)
    ps = ParticleSystem(9000, device)

    thermo = scenario == "thermonuclear"
    burst_alt = 3.2
    drop_h = 9.0

    # The device (falls under gravity if dropping).
    if drop:
        ps.spawn(np.array([[0.0, drop_h, 0.0]], np.float32), np.zeros((1, 3), np.float32),
                 np.zeros(1, np.float32), np.array([100.0], np.float32), np.array([4], np.int32))
        device_slot = (ps.cursor - 1) % ps.max_n

    # Fission primary; thermonuclear fuses far more energy per unit burn.
    # Low burn_rate lets the population grow many decades (prompt-supercritical)
    # before the fuel disassembles and reactivity drops below critical.
    primary = _Stage(k0=2.2, rate=13.0, burn_rate=1.2e-5, e_per_burn=1.0)
    secondary = _Stage(k0=2.8, rate=14.0, burn_rate=1.0e-5, e_per_burn=24.0) if thermo else None

    detonated = False
    ignited = False
    fireball_started = False
    det_f = 0
    det_center = np.array([0.0, burst_alt, 0.0], np.float32)
    flash = 0.0
    report = {"scenario": scenario, "dropped": bool(drop), "generations": [],
              "primary_peak_neutrons": 0.0, "secondary_peak_neutrons": 0.0,
              "primary_energy": 0.0, "secondary_energy": 0.0}

    frames_out = []
    for f in range(frames):
        # --- drop phase: detonate at burst altitude ---
        if drop and not detonated:
            y = ps.pos[device_slot, 1]
            if y <= burst_alt:
                detonated = True
                det_f = f
                det_center = ps.pos[device_slot].copy()
                ps.life[device_slot] = 0.0        # remove the casing
                primary.ignite(1.0)
                flash = 1.6
        elif not drop and not detonated:
            detonated = True
            det_f = f
            primary.ignite(1.0)
            flash = 1.6

        # --- chain reaction ---
        if detonated:
            de = primary.step(dt)
            # neutron sparks scale with log-population (the growing cascade)
            nspark = int(np.clip(np.log10(primary.n + 1.0) * 12.0, 0, 70))
            _spawn_neutrons(ps, det_center, nspark, rng)

            # fireball grows with released energy
            if de > 0.0:
                if not fireball_started and primary.energy > 0.12:
                    _spawn_fireball(ps, det_center, 2000 if thermo else 3400,
                                    6.0 if thermo else 7.0, 1.0, 6.5, rng)
                    _spawn_base_surge(ps, 900, 5.0, rng)
                    fireball_started = True
                _spawn_column(ps, float(det_center[1]), 26, rng)

            # thermonuclear: fission primary ignites the fusion secondary
            if thermo and not ignited and primary.energy > 0.6:
                ignited = True
                secondary.ignite(1.0)
                flash = 2.6

            if thermo and ignited:
                de2 = secondary.step(dt)
                nspark2 = int(np.clip(np.log10(secondary.n + 1.0) * 14.0, 0, 90))
                _spawn_neutrons(ps, det_center, nspark2, rng)
                if de2 > 0.0 and secondary.energy > 0.5 and ps.kind.tolist().count(2) < 7000:
                    _spawn_fireball(ps, det_center, 110, 13.0, 1.15, 7.0, rng)
                    _spawn_column(ps, float(det_center[1]), 32, rng, temp=0.85)

        # --- physics step (with the mushroom-cap vortex) ---
        # Buoyancy > gravity for hot particles (they rise); a poloidal vortex ring
        # centered on the rising cap rolls the edges outward into the cap.
        vortex = 0.0
        cap_y = 0.0
        ring_a = 1.0
        if fireball_started:
            fbm = ps.alive_mask() & (ps.kind == 2)
            if fbm.any():
                cap_y = float(np.mean(ps.pos[fbm, 1]))
                ring_a = 0.5 + 0.045 * float(f - det_f)
                vortex = 10.0
        ps.step(dt, G, buoy=13.0, drag=1.0, cool=0.32,
                vortex=vortex, cap_y=cap_y, ring_a=ring_a)
        if detonated:
            report["generations"].append(
                (f, round(primary.n, 2), round(primary.energy, 3),
                 round(secondary.n, 2) if secondary else 0.0,
                 round(secondary.energy, 3) if secondary else 0.0))

        # --- render (camera tracks the rising cloud) ---
        fb = ps.alive_mask() & (ps.kind == 2)
        cam_y = float(np.clip(np.mean(ps.pos[fb, 1]), 4.0, 40.0)) if fb.any() else 5.0
        eye = np.array([7.0, cam_y, 24.0], np.float32)
        frame = ps.render(width, height, eye, np.array([0.0, cam_y, 0.0], np.float32),
                          fov_deg=48.0, exposure=0.6, stamp_radius=3)
        if flash > 0.02:
            frame = frame + np.array([1.0, 0.97, 0.9], np.float32) * flash
            flash *= 0.6
        frames_out.append(frame)

    report["primary_peak_neutrons"] = primary.peak_n
    report["primary_energy"] = primary.energy
    if secondary:
        report["secondary_peak_neutrons"] = secondary.peak_n
        report["secondary_energy"] = secondary.energy
    report["total_energy"] = primary.energy + (secondary.energy if secondary else 0.0)
    return frames_out, report
