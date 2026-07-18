"""Maxwell's equations — a 2-D FDTD (finite-difference time-domain) solver on the Yee grid.

The bottom of the graphics card's reality: everything on the board is a solution of Maxwell's
equations. This is the honest substrate — the step function *is* the physics. We solve the source-free
(with sources) TMz curl equations

    ∂Hx/∂t = -(1/µ) ∂Ez/∂y
    ∂Hy/∂t =  (1/µ) ∂Ez/∂x
    ∂Ez/∂t =  (1/ε)(∂Hy/∂x - ∂Hx/∂y) - (σ/ε) Ez

by Yee's leapfrog: E and H are staggered a half cell in space and a half step in time, so each curl is
a centred difference of the *other* field. This is second-order accurate, explicit, and preserves the
discrete divergence (∇·(εE)=ρ) by construction. Normalised units: ε0 = µ0 = 1, so vacuum light speed
c = 1, cell size Δx = Δy = 1, and the time step is the Courant number ``sc`` (must satisfy the CFL
limit sc ≤ 1/√2 in 2-D, using the fastest medium). Materials enter as per-cell maps: copper is a high
conductivity σ (a reflector), FR-4 is a dielectric ε_r ≈ 4.3 (slows the wave to c/√ε_r).

Verification (see ``tests/test_em.py``): in a closed PEC cavity with σ=0 the total field energy
U = ½Σε|Ez|² + ½µΣ|H|² is **conserved** (bounded, no growth = stable/CFL-respected, no decay =
lossless); a pulse launched in vacuum travels at **speed c**. If either check fails, the step is lying.
"""

import numpy as np

C_VACUUM = 1.0                      # normalised: eps0 = mu0 = 1  ->  c = 1
FR4_EPS_R = 4.3                     # FR-4 dielectric constant (board substrate)
CFL_2D = 1.0 / np.sqrt(2.0)         # Courant limit in 2-D


class EMField:
    """2-D FDTD Maxwell solver (TMz: Ez, Hx, Hy) with per-cell ε and σ material maps."""

    def __init__(self, n=256, sc=0.5, mu=1.0):
        assert sc <= CFL_2D + 1e-9, f"Courant sc={sc} exceeds 2-D CFL limit {CFL_2D:.4f} (unstable)"
        self.n = int(n)
        self.sc = float(sc)                       # Courant number = dt (dx=1, c=1)
        self.mu = float(mu)
        self.Ez = np.zeros((n, n), np.float64)
        self.Hx = np.zeros((n, n), np.float64)
        self.Hy = np.zeros((n, n), np.float64)
        self.eps = np.ones((n, n), np.float64)    # relative permittivity map (1 = vacuum/air)
        self.sigma = np.zeros((n, n), np.float64) # electric conductivity map (0 = lossless)
        self.pec = np.zeros((n, n), bool)         # perfect electric conductor cells (Ez pinned 0)
        self.sources = []                         # (i, j, amp, omega, phase) soft Ez sources
        self._t = 0

    # ---- material helpers (the board's copper + FR-4) -------------------------------------------
    def fill_dielectric(self, eps_r=FR4_EPS_R):
        self.eps[:] = eps_r

    def add_conductor_rect(self, x0, y0, x1, y1, sigma=1e6):
        """A copper region (high σ) — reflects EM waves. Coords are fractions of the grid."""
        n = self.n
        i0, i1 = int(x0 * n), int(x1 * n)
        j0, j1 = int(y0 * n), int(y1 * n)
        self.sigma[i0:i1, j0:j1] = sigma

    def add_pec_border(self):
        """Close the domain with perfect mirrors (for the energy-conservation cavity test)."""
        self.pec[0, :] = self.pec[-1, :] = True
        self.pec[:, 0] = self.pec[:, -1] = True

    def set_absorbing_border(self, width=0.08, sigma_max=0.8):
        """A graded-σ (UPML-lite) lossy border so open-domain waves leave without reflecting."""
        n = self.n
        w = max(1, int(width * n))
        ramp = np.zeros((n, n), np.float64)
        for k in range(w):
            s = sigma_max * ((w - k) / w) ** 3
            ramp[k, :] = np.maximum(ramp[k, :], s)
            ramp[n - 1 - k, :] = np.maximum(ramp[n - 1 - k, :], s)
            ramp[:, k] = np.maximum(ramp[:, k], s)
            ramp[:, n - 1 - k] = np.maximum(ramp[:, n - 1 - k], s)
        self.sigma = np.maximum(self.sigma, ramp)

    def add_source(self, x, y, amp=1.0, omega=0.3, phase=0.0):
        self.sources.append((int(x * self.n), int(y * self.n), float(amp), float(omega), float(phase)))

    def pulse(self, x, y, amp=1.0, width=3.0):
        """Seed a Gaussian bump in Ez (an initial condition — e.g. a switching event on the die)."""
        n = self.n
        ci, cj = x * n, y * n
        ii, jj = np.mgrid[0:n, 0:n]
        self.Ez += amp * np.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2.0 * width * width))

    # ---- the leapfrog step (this IS Maxwell) ----------------------------------------------------
    def step(self):
        sc, mu = self.sc, self.mu
        Ez, Hx, Hy = self.Ez, self.Hx, self.Hy
        # H update at n+1/2 from the spatial curl of E (forward differences)
        Hx[:, :-1] -= (sc / mu) * (Ez[:, 1:] - Ez[:, :-1])
        Hy[:-1, :] += (sc / mu) * (Ez[1:, :] - Ez[:-1, :])
        # E update at n+1 from the curl of H (backward differences), with the lossy (σ) coefficients
        denom = 1.0 + self.sigma * sc / (2.0 * self.eps)
        ca = (1.0 - self.sigma * sc / (2.0 * self.eps)) / denom
        cb = (sc / self.eps) / denom
        curl = np.zeros_like(Ez)
        curl[1:, 1:] = (Hy[1:, 1:] - Hy[:-1, 1:]) - (Hx[1:, 1:] - Hx[1:, :-1])
        Ez[:] = ca * Ez + cb * curl
        if self.pec.any():
            Ez[self.pec] = 0.0
        self._t += 1
        for i, j, amp, omega, phase in self.sources:
            Ez[i, j] += amp * np.sin(omega * self._t + phase)   # soft source (adds, doesn't clamp)

    def run(self, steps):
        for _ in range(int(steps)):
            self.step()
        return self.Ez

    # ---- the honesty instruments ----------------------------------------------------------------
    def energy(self):
        """Total EM field energy U = ½Σε|Ez|² + ½µΣ|H|² (the conserved quantity when σ=0, PEC)."""
        ue = 0.5 * np.sum(self.eps * self.Ez * self.Ez)
        um = 0.5 * self.mu * np.sum(self.Hx * self.Hx + self.Hy * self.Hy)
        return float(ue + um)

    def divergence(self):
        """Discrete ∇·(εE) — should stay equal to the seeded charge (≈0 here) to round-off."""
        d = self.eps * self.Ez
        div = np.zeros_like(d)
        div[1:, 1:] = (d[1:, 1:] - d[:-1, 1:]) + (d[1:, 1:] - d[1:, :-1])
        return div
