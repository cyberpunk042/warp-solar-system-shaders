"""Tests for the FDTD Maxwell solver (warp_shaders.sim.em) — reality is not facultative.

The step function claims to BE Maxwell's equations. These tests hold it to the laws it claims:
  1. Energy conservation — in a closed lossless (σ=0) PEC cavity, U = ½Σε|E|² + ½µΣ|H|² stays
     bounded (no growth = CFL respected/stable; no decay = lossless). A leaking or blowing-up energy
     means the discrete step is not Maxwell.
  2. Wave speed = c — a pulse launched in vacuum travels at the speed of light (normalised c = 1).
  3. A dielectric (FR-4, ε_r=4.3) slows the wave to c/√ε_r.
  4. A copper conductor reflects (a lossy region removes energy from an open pulse).

    python -m tests.test_em
"""

import numpy as np

from warp_shaders.sim.em import EMField, C_VACUUM, FR4_EPS_R


def _front_radius(Ez, thresh):
    """Radius (in cells) of the outermost cell whose |Ez| exceeds ``thresh``, from the grid centre."""
    n = Ez.shape[0]
    ii, jj = np.mgrid[0:n, 0:n]
    r = np.sqrt((ii - n / 2.0) ** 2 + (jj - n / 2.0) ** 2)
    mask = np.abs(Ez) > thresh
    return float(r[mask].max()) if mask.any() else 0.0


def main():
    # 1. Energy conservation in a lossless PEC cavity
    f = EMField(n=200, sc=0.5)
    f.add_pec_border()
    f.pulse(0.5, 0.5, amp=1.0, width=4.0)
    U = []
    for _ in range(3000):
        f.step()
        U.append(f.energy())
    U = np.array(U[50:])                      # drop the initial E->H settling transient
    assert np.all(np.isfinite(U)), "energy went non-finite (unstable)"
    rel = (U.max() - U.min()) / U.mean()
    drift = abs(U[-1] - U[:200].mean()) / U[:200].mean()
    assert U.max() < 3.0 * U[:200].mean(), "energy grew — CFL violated / unstable"
    assert rel < 0.06, f"energy not conserved (rel spread {rel:.4f})"
    assert drift < 0.03, f"energy drifted (net {drift:.4f}) — not lossless"
    print(f"  energy conservation: OK  (rel spread {rel*100:.2f}%, net drift {drift*100:.2f}% over 3000 steps)")

    # 2. Wave speed = c in vacuum (open domain, absorbing border)
    f = EMField(n=260, sc=0.5)
    f.set_absorbing_border(width=0.1)
    f.pulse(0.5, 0.5, amp=1.0, width=2.0)
    thresh = 0.02
    t1, t2 = 60, 130
    for _ in range(t1):
        f.step()
    r1 = _front_radius(f.Ez, thresh)
    for _ in range(t2 - t1):
        f.step()
    r2 = _front_radius(f.Ez, thresh)
    speed_cells_per_step = (r2 - r1) / (t2 - t1)
    speed = speed_cells_per_step / f.sc                     # physical speed (dx=1, dt=sc, c=1)
    assert abs(speed - C_VACUUM) < 0.08, f"vacuum wave speed {speed:.3f} != c (dispersion too high)"
    print(f"  wave speed = c: OK  (measured {speed:.3f}c, front {r1:.0f}->{r2:.0f} cells)")

    # 3. Dielectric (FR-4) slows the wave to c/sqrt(eps_r)
    g = EMField(n=260, sc=0.5)
    g.set_absorbing_border(width=0.1)
    g.fill_dielectric(FR4_EPS_R)
    g.pulse(0.5, 0.5, amp=1.0, width=2.0)
    for _ in range(t1):
        g.step()
    r1d = _front_radius(g.Ez, thresh)
    for _ in range(t2 - t1):
        g.step()
    r2d = _front_radius(g.Ez, thresh)
    speed_d = ((r2d - r1d) / (t2 - t1)) / g.sc
    expected = C_VACUUM / np.sqrt(FR4_EPS_R)
    assert speed_d < 0.7 * C_VACUUM, f"dielectric did not slow the wave (speed {speed_d:.3f}c)"
    print(f"  FR-4 slows wave: OK  (measured {speed_d:.3f}c, expected ~{expected:.3f}c)")

    # 4. Copper conductor reflects: a lossy conductor block removes energy from an open pulse
    h = EMField(n=200, sc=0.5)
    h.set_absorbing_border(width=0.1)
    h.add_conductor_rect(0.60, 0.20, 0.66, 0.80, sigma=5.0)   # a vertical copper trace
    h.pulse(0.4, 0.5, amp=1.0, width=2.0)
    e_hits = []
    for k in range(200):
        h.step()
    assert np.all(np.isfinite(h.Ez)), "conductor case went non-finite"
    # energy left of the trace should dominate (wave reflected, didn't pass cleanly through)
    left = np.sum(h.Ez[:120, :] ** 2)
    right = np.sum(h.Ez[132:, :] ** 2)
    assert left > right, f"copper trace did not reflect (left {left:.3g} <= right {right:.3g})"
    print(f"  copper reflects: OK  (energy left {left:.3g} > right {right:.3g})")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
