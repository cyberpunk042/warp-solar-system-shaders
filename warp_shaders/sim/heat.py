"""Heat — the diffusion equation on the die/board grid (simulation of reality, B5).

Essentially *all* the electrical power the card draws becomes heat, in situ, at the die. So the honest
thermal simulation is the **heat equation** with a source, solved in time:

    ∂T/∂t = α ∇²T + q/(ρ·c_p) − κ·(T − T_amb)

α = k/(ρ·c_p) is the thermal diffusivity (spreading), q is the volumetric power map (the **die
floorplan** — non-uniform: compute-dense regions are hotspots), and the −κ(T−T_amb) term is the
cooler carrying heat away to ambient (Newton's law of cooling, lumped). We step it explicitly (FTCS):

    T^{n+1} = T^n + (α·Δt/Δx²)·∇²T^n + Δt·q/(ρc_p) − Δt·κ·(T^n − T_amb)

which is stable only under the von-Neumann **Fourier limit** r = α·Δt/Δx² ≤ 1/4 (in 2-D) — reality is
not facultative, so the constructor enforces it. Boundaries: Dirichlet (fixed edge), Neumann
(insulated, mirror ghost), or the lumped convective κ term.

Verification (``tests/test_heat.py``): with fixed ends and no source the steady state is the exact
analytic **linear** profile (∇²T=0); a seeded Fourier mode decays at exactly **α·k²**; with insulated
edges the mean temperature rises at exactly **q/(ρc_p)** (the discrete first law). If any check drifts,
the step is not the heat equation.
"""

import numpy as np


class HeatField:
    """2-D heat-diffusion solver (explicit FTCS) with a power-source map + cooling."""

    def __init__(self, n=200, alpha=0.2, dx=1.0, dt=None, rho_cp=1.0, kappa=0.0,
                 t_amb=0.0, boundary="dirichlet", safety=0.9):
        self.n = int(n)
        self.alpha, self.dx, self.rho_cp = float(alpha), float(dx), float(rho_cp)
        self.kappa, self.t_amb = float(kappa), float(t_amb)
        self.boundary = boundary
        dt_max = 0.25 * dx * dx / alpha                 # 2-D Fourier stability limit r <= 1/4
        self.dt = float(dt) if dt is not None else safety * dt_max
        r = alpha * self.dt / (dx * dx)
        assert r <= 0.25 + 1e-9, f"Fourier number r={r:.3f} exceeds the 2-D FTCS limit 1/4 (unstable)"
        self.T = np.full((n, n), float(t_amb), np.float64)
        self.q = np.zeros((n, n), np.float64)           # volumetric power map (die floorplan)

    def set_edge(self, value):
        """Dirichlet: pin the four borders to a fixed temperature."""
        self.T[0, :] = self.T[-1, :] = self.T[:, 0] = self.T[:, -1] = float(value)

    def add_source_rect(self, x0, y0, x1, y1, power):
        n = self.n
        self.q[int(x0 * n):int(x1 * n), int(y0 * n):int(y1 * n)] += float(power)

    def add_source_gauss(self, x, y, power, radius):
        n = self.n
        ii, jj = np.mgrid[0:n, 0:n]
        self.q += power * np.exp(-((ii - x * n) ** 2 + (jj - y * n) ** 2) / (2.0 * (radius * n) ** 2))

    def _laplacian(self):
        T = self.T
        lap = np.zeros_like(T)
        lap[1:-1, 1:-1] = (T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2]
                           - 4.0 * T[1:-1, 1:-1])
        return lap

    def step(self):
        r = self.alpha * self.dt / (self.dx * self.dx)
        self.T[1:-1, 1:-1] += (r * self._laplacian()[1:-1, 1:-1]
                               + self.dt * self.q[1:-1, 1:-1] / self.rho_cp
                               - self.dt * self.kappa * (self.T[1:-1, 1:-1] - self.t_amb))
        if self.boundary == "neumann":                  # insulated: mirror the interior into ghosts
            self.T[0, :] = self.T[1, :]
            self.T[-1, :] = self.T[-2, :]
            self.T[:, 0] = self.T[:, 1]
            self.T[:, -1] = self.T[:, -2]
        # "dirichlet": borders keep whatever set_edge fixed them to (untouched by the interior update)

    def run(self, steps):
        for _ in range(int(steps)):
            self.step()
        return self.T

    def mean(self):
        return float(self.T.mean())

    def hotspot(self):
        return float(self.T.max())
