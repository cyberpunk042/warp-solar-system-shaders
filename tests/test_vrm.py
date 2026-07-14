"""Tests for the buck-converter VRM solver (warp_shaders.sim.vrm) — real switching physics.

Holds the switching-ODE step to the laws it claims:
  1. steady state converges to the ideal CCM conversion V_out = D·Vin;
  2. the inductor current ripple matches ΔI_L = (Vin−V_out)·D / (L·f_sw);
  3. a lossless L–C tank conserves ½L·i² + ½C·v² (symplectic integrator — forward Euler would blow up);
  4. N interleaved phases reduce the output-voltage ripple versus a single phase.

    python -m tests.test_vrm
"""

import numpy as np

from warp_shaders.sim.vrm import BuckConverter


def _pp_over_periods(x, dt, fsw, periods):
    """Peak-to-peak of the last whole `periods` switching cycles (avoids partial-cycle bias)."""
    npc = int(round(1.0 / (fsw * dt)))
    tail = x[-npc * periods:]
    return float(tail.max() - tail.min())


def main():
    # 1. steady state -> V_out = D·Vin  (CCM: big enough L for modest ripple, small RC so it settles)
    for D in (0.25, 0.5, 0.75):
        b = BuckConverter(n_phases=1, Vin=12.0, D=D, L=1.0, C=1.0, Rload=1.0, fsw=1.0, dt=1e-3)
        b.run(20000)                                    # settle (RC and LC both ~O(1), well settled)
        vout = b.run_record_vout(4000).mean()
        expect = D * 12.0
        rel = abs(vout - expect) / expect
        assert rel < 0.05, f"D={D}: V_out {vout:.3f} != D·Vin {expect:.3f} (rel {rel:.3f})"
        print(f"  V_out = D·Vin  (D={D}): OK  ({vout:.3f} V vs {expect:.3f} V)")

    # 2. inductor ripple matches (Vin - Vout)·D / (L·fsw)
    Vin, D, L, fsw = 12.0, 0.5, 1.0, 1.0
    b = BuckConverter(n_phases=1, Vin=Vin, D=D, L=L, C=1.0, Rload=1.0, fsw=fsw, dt=5e-4)
    b.run(20000)
    iL = b.run_record_iL(int(2 / b.dt))
    ripple = _pp_over_periods(iL, b.dt, fsw, 1)
    expect = (Vin - D * Vin) * D / (L * fsw)
    rel = abs(ripple - expect) / expect
    assert rel < 0.15, f"ripple {ripple:.3f} != (Vin-Vout)·D/(L·fsw) {expect:.3f} (rel {rel:.3f})"
    print(f"  inductor ripple: OK  (ΔI_L {ripple:.3f} A vs formula {expect:.3f} A)")

    # 3. lossless free L–C tank conserves energy (symplectic); forward Euler would grow without bound
    b = BuckConverter(n_phases=1, Vin=0.0, D=0.0, L=1.0, C=1.0, Rload=np.inf, R=0.0, dt=0.02)
    b.iL[0] = 1.0
    b.vC = 0.0
    U = np.array([(b.step(), b.energy_lc())[1] for _ in range(6000)])
    assert np.all(np.isfinite(U)), "tank energy non-finite (unstable integrator)"
    rel = (U.max() - U.min()) / U.mean()
    drift = abs(U[-1] - U[0]) / U[0]
    assert rel < 0.03 and drift < 0.02, f"tank energy not conserved (spread {rel:.4f}, drift {drift:.4f})"
    print(f"  L–C tank energy: OK  (spread {rel*100:.2f}%, drift {drift*100:.2f}% over 6000 steps)")

    # 4. interleaving: N phases cut the output-voltage ripple vs a single phase (same regulation point)
    def out_ripple(n):
        b = BuckConverter(n_phases=n, Vin=12.0, D=0.3, L=1.0, C=0.5, Rload=1.0, fsw=1.0, dt=2e-4)
        b.run(30000)
        v = b.run_record_vout(int(2 / b.dt))
        return _pp_over_periods(v, b.dt, 1.0, 1)
    r1 = out_ripple(1)
    r6 = out_ripple(6)
    assert r6 < 0.6 * r1, f"interleaving did not cut ripple enough (1-phase {r1:.4g}, 6-phase {r6:.4g})"
    print(f"  interleaving cuts ripple: OK  (1-phase {r1:.4g} -> 6-phase {r6:.4g}, {r1/r6:.1f}x)")

    # 5. the vrm_power scene renders the solved currents on the real board, and animates
    import warp as wp
    import warp_shaders as ws
    wp.init()
    a = np.asarray(ws.render("vrm_power", width=120, height=96, time=0.15), np.float32)
    b = np.asarray(ws.render("vrm_power", width=120, height=96, time=0.65), np.float32)
    assert np.all(np.isfinite(a)) and a.max() > 0.1 and a.std() > 0.01, "vrm_power: bad frame"
    assert np.abs(a - b).mean() > 1e-3, "vrm_power: current did not animate"
    print("  scene vrm_power: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
