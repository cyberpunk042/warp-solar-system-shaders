"""Transmission lines — the telegrapher's equations, solved in time (simulation of reality, B2).

A PCB trace carrying a GHz signal is not a wire — it is a **transmission line**. Once the edge rate is
fast enough that the trace is electrically long, a signal launched into it propagates as a wave, and
wherever the impedance changes it partly **reflects**. This is the 1-D reduction of the Maxwell field
that ``sim/em.py`` solves in 2-D: for a quasi-TEM line the four Maxwell equations collapse to the
**telegrapher's equations** with distributed series L, R and shunt C, G per unit length,

    ∂V/∂z = -(R + L ∂/∂t) I
    ∂I/∂z = -(G + C ∂/∂t) V

with characteristic impedance Z0 = √(L/C) and phase velocity v = 1/√(LC). We step them with the same
Yee-style leapfrog as the FDTD field: V on integer nodes at step n, I on the half nodes at n+½. A
Thevenin source (series R_s) drives the near end; a resistive load R_L terminates the far end, so the
reflection coefficient at the load is exactly Γ = (R_L − Z0)/(R_L + Z0) — open (Γ=+1) doubles the
voltage, short (Γ=−1) nulls it, matched (Γ=0) absorbs it.

Verification (``tests/test_tline.py``): the measured reflection at open / short / matched / resistive
loads matches the analytic Γ; the pulse travels at v = 1/√(LC); a lossless line (R=G=0) conserves the
energy U = ½Σ(L·I² + C·V²). Normalised units: L = C = 1 ⇒ Z0 = 1, v = 1, Δz = 1, Δt = sc ≤ 1.
"""

import numpy as np


class TransmissionLine:
    """1-D telegrapher-equation solver (V, I leapfrog) with Thevenin source + resistive load."""

    def __init__(self, n=400, sc=0.9, L=1.0, C=1.0, R=0.0, G=0.0):
        assert sc <= 1.0 + 1e-9, f"Courant sc={sc} exceeds the transmission-line CFL limit 1.0"
        self.n = int(n)
        self.dt = float(sc)                       # dz = 1, v = 1/sqrt(LC); with L=C=1, dt=sc
        self.L, self.C, self.R, self.G = float(L), float(C), float(R), float(G)
        self.Z0 = np.sqrt(L / C)
        self.v = 1.0 / np.sqrt(L * C)
        self.V = np.zeros(n, np.float64)          # voltage on integer nodes 0..n-1
        self.I = np.zeros(n - 1, np.float64)      # current on half nodes between V[k] and V[k+1]
        self.rs = self.Z0                          # source series resistance (matched by default)
        self.rl = np.inf                           # load resistance (open by default)
        self.vs_fn = None                          # optional source voltage vs(t)
        self._t = 0

    # ---- terminations -------------------------------------------------------------------------
    def source(self, rs, vs_fn=None):
        self.rs = float(rs)
        self.vs_fn = vs_fn
        return self

    def load(self, rl):
        self.rl = float(rl)
        return self

    def launch_pulse(self, center=0.12, width=0.03, amp=1.0):
        """Seed a right-travelling Gaussian voltage pulse (V and I = V/Z0 so it moves +z)."""
        z = np.arange(self.n) / self.n
        self.V = amp * np.exp(-((z - center) ** 2) / (2.0 * width * width))
        vh = 0.5 * (self.V[:-1] + self.V[1:])
        self.I = vh / self.Z0                       # right-travelling: I = V / Z0
        return float(self.V.max())

    # ---- the leapfrog step (this IS the telegrapher's equations) -------------------------------
    def step(self):
        dt, L, C, R, G = self.dt, self.L, self.C, self.R, self.G
        # I update (half step): dI/dt = -(1/L)(dV/dz) - (R/L) I
        cai = (1.0 - R * dt / (2.0 * L)) / (1.0 + R * dt / (2.0 * L))
        cbi = (dt / L) / (1.0 + R * dt / (2.0 * L))
        self.I = cai * self.I - cbi * (self.V[1:] - self.V[:-1])
        # V update (full step) on interior nodes: dV/dt = -(1/C)(dI/dz) - (G/C) V
        cav = (1.0 - G * dt / (2.0 * C)) / (1.0 + G * dt / (2.0 * C))
        cbv = (dt / C) / (1.0 + G * dt / (2.0 * C))
        self.V[1:-1] = cav * self.V[1:-1] - cbv * (self.I[1:] - self.I[:-1])
        self._t += 1
        # near-end Thevenin source (series rs): half-cell cap C/2, KCL (Vs - V0)/rs = (C/2)dV0/dt + I[0]
        k = C / (2.0 * dt)
        vs = 0.0 if self.vs_fn is None else float(self.vs_fn(self._t * dt))
        gs = 1.0 / self.rs
        self.V[0] = ((k - 0.5 * gs) * self.V[0] - self.I[0] + gs * vs) / (k + 0.5 * gs)
        # far-end resistive load rl: KCL  I[-1] = (C/2)dV/dt + V/rl
        gl = 0.0 if not np.isfinite(self.rl) else 1.0 / self.rl
        self.V[-1] = ((k - 0.5 * gl) * self.V[-1] + self.I[-1]) / (k + 0.5 * gl)

    def run(self, steps):
        for _ in range(int(steps)):
            self.step()
        return self.V

    # ---- honesty instruments -----------------------------------------------------------------
    def energy(self):
        """U = ½Σ C·V² + ½Σ L·I² — conserved on a lossless (R=G=0) matched-open line."""
        ue = 0.5 * self.C * np.sum(self.V * self.V)
        um = 0.5 * self.L * np.sum(self.I * self.I)
        return float(ue + um)

    def gamma(self):
        """Analytic reflection coefficient at the load."""
        if not np.isfinite(self.rl):
            return 1.0
        return (self.rl - self.Z0) / (self.rl + self.Z0)
