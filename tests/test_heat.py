"""Tests for the heat-diffusion solver (warp_shaders.sim.heat) — the heat equation, verified.

  1. steady state with fixed ends and no source is the exact analytic LINEAR profile (∇²T = 0);
  2. a seeded Fourier mode decays at exactly α·k² (the diffusion eigenvalue);
  3. with insulated edges the mean temperature rises at exactly q/(ρc_p) (the discrete first law);
  4. the die-floorplan scene forms a hotspot and reaches a bounded steady state (source = cooling).

    python -m tests.test_heat
"""

import numpy as np

from warp_shaders.sim.heat import HeatField


def main():
    # 1. steady state = analytic linear gradient between two fixed edges (no source)
    n = 80
    h = HeatField(n=n, alpha=0.2, boundary="dirichlet")
    h.T[:, 0] = 0.0
    h.T[:, -1] = 100.0
    # hold left/right columns fixed each step; top/bottom free (1-D problem in y)
    for _ in range(60000):
        left = h.T[:, 0].copy(); rightcol = h.T[:, -1].copy()
        h.step()
        h.T[:, 0] = left; h.T[:, -1] = rightcol
        h.T[0, :] = h.T[1, :]; h.T[-1, :] = h.T[-2, :]      # insulated top/bottom
    mid = h.T[n // 2, :]
    analytic = np.linspace(0.0, 100.0, n)
    err = np.abs(mid - analytic).max()
    assert err < 2.0, f"steady state not linear (max err {err:.3f})"
    print(f"  steady state = analytic linear: OK  (max err {err:.3f} of 100)")

    # 2. a Fourier mode decays at alpha*k^2
    n = 120
    L = n - 1
    k = 3
    h = HeatField(n=n, alpha=0.2, boundary="dirichlet")
    x = np.arange(n)
    mode = np.sin(k * np.pi * x / L)
    h.T[:] = mode[None, :] * np.ones((n, 1))
    h.T[:, 0] = 0.0; h.T[:, -1] = 0.0
    amp0 = np.abs(h.T[n // 2, :]).max()
    steps = 400
    for _ in range(steps):
        h.step()
        h.T[0, :] = h.T[1, :]; h.T[-1, :] = h.T[-2, :]
    amp1 = np.abs(h.T[n // 2, :]).max()
    measured = -np.log(amp1 / amp0) / (steps * h.dt)
    expected = h.alpha * (k * np.pi / L) ** 2
    rel = abs(measured - expected) / expected
    assert rel < 0.06, f"mode decay {measured:.5f} != alpha*k^2 {expected:.5f} (rel {rel:.3f})"
    print(f"  mode decays at alpha*k^2: OK  (measured {measured:.5f}, expected {expected:.5f})")

    # 3. discrete first law: insulated box + uniform source -> dT_mean/dt = q/(rho_cp)
    h = HeatField(n=60, alpha=0.2, rho_cp=2.0, boundary="neumann")
    h.q[:] = 3.0
    T0 = h.mean()
    steps = 500
    for _ in range(steps):
        h.step()
    rate = (h.mean() - T0) / (steps * h.dt)
    expected = 3.0 / 2.0
    rel = abs(rate - expected) / expected
    assert rel < 0.02, f"mean-T rise {rate:.4f} != q/(rho_cp) {expected:.4f} (rel {rel:.3f})"
    print(f"  first law (dT/dt = q/rho_cp): OK  (measured {rate:.4f}, expected {expected:.4f})")

    # 4. die floorplan: a hotspot forms and the board reaches a bounded steady state (source = cooling)
    h = HeatField(n=120, alpha=0.25, kappa=0.03, t_amb=20.0, boundary="neumann")
    h.add_source_gauss(0.45, 0.5, power=1.2, radius=0.10)     # the compute-dense die hotspot
    h.add_source_rect(0.30, 0.35, 0.60, 0.65, 0.25)          # the broader die
    T_start = h.hotspot()
    h.run(4000)
    T_hot = h.hotspot()
    T_edge = float(h.T[2, 2])
    assert np.all(np.isfinite(h.T)), "heat field went non-finite"
    assert T_hot > T_start + 5.0, "no hotspot formed"
    assert T_hot > T_edge + 5.0, "hotspot not hotter than the board edge (no gradient)"
    assert T_hot < 500.0, "runaway (cooling did not bound the steady state)"
    print(f"  die hotspot + steady state: OK  (hotspot {T_hot:.1f}, edge {T_edge:.1f}, amb 20.0)")

    # 5. the heat_die scene renders the solved temperature on the real board, and animates
    import warp as wp
    import warp_shaders as ws
    wp.init()
    a = np.asarray(ws.render("heat_die", width=120, height=96, time=3.0), np.float32)   # heating
    b = np.asarray(ws.render("heat_die", width=120, height=96, time=8.0), np.float32)   # cooling
    assert np.all(np.isfinite(a)) and a.max() > 0.1 and a.std() > 0.01, "heat_die: bad frame"
    assert np.abs(a - b).mean() > 1e-3, "heat_die: temperature did not animate"
    print("  scene heat_die: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
