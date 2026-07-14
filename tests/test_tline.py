"""Tests for the transmission-line solver (warp_shaders.sim.tline) — the telegrapher's equations.

The load-node voltage of a reflecting line settles the physics: a right-travelling incident pulse of
amplitude Vinc arriving at a load reflects with Γ = (R_L−Z0)/(R_L+Z0), so the total voltage AT the
load peaks at Vinc·(1+Γ) — doubles for an open (Γ=+1), nulls for a short (Γ=−1), passes for a matched
load (Γ=0). We launch a pulse, absorb re-reflections at a matched source, and check the load peak
against the analytic Γ. Plus: wave speed = 1/√(LC), and a lossless line conserves energy.

    python -m tests.test_tline
"""

import numpy as np

from warp_shaders.sim.tline import TransmissionLine


def _load_peak_over_incident(rl, n=500):
    """Launch a pulse into a matched-source line terminated by rl; return peak V at the load / Vinc."""
    tl = TransmissionLine(n=n, sc=0.9)
    tl.source(rs=tl.Z0)                 # matched source absorbs the returning wave (clean measurement)
    tl.load(rl)
    vinc = tl.launch_pulse(center=0.12, width=0.025, amp=1.0)
    peak = 0.0
    for _ in range(int(1.3 * n / 0.9)):            # long enough for the pulse to reach + reflect
        tl.step()
        peak = max(peak, abs(tl.V[-1]))
    return peak / vinc, tl.gamma()


def main():
    # 1. reflection coefficient at open / short / matched / resistive loads matches analytic Γ
    Z0 = 1.0
    cases = [
        ("open",      np.inf,   2.0),   # Γ=+1  -> 1+Γ = 2
        ("short",     1e-6,     0.0),   # Γ=-1  -> 0
        ("matched",   Z0,       1.0),   # Γ=0   -> 1
        ("R=3·Z0",    3.0 * Z0, 1.5),   # Γ=0.5 -> 1.5
        ("R=Z0/3",    Z0 / 3.0, 0.5),   # Γ=-0.5 -> 0.5
    ]
    for name, rl, expect in cases:
        ratio, g = _load_peak_over_incident(rl)
        assert abs(ratio - expect) < 0.12, f"{name}: load peak {ratio:.3f} != 1+Γ {expect:.3f} (Γ={g:+.2f})"
        print(f"  reflection {name:8s}: OK  (load peak {ratio:.3f} = 1+Γ, Γ={g:+.3f})")

    # 2. wave speed = 1/sqrt(LC)
    for L, C in [(1.0, 1.0), (2.0, 1.0), (1.0, 4.0)]:
        tl = TransmissionLine(n=600, sc=0.9, L=L, C=C)
        tl.source(rs=tl.Z0)
        tl.launch_pulse(center=0.1, width=0.02, amp=1.0)
        p0 = np.argmax(tl.V) / tl.n
        steps = 260
        for _ in range(steps):
            tl.step()
        p1 = np.argmax(tl.V) / tl.n
        measured = (p1 - p0) * tl.n / (steps * tl.dt)     # cells/time
        expected = tl.v
        rel = abs(measured - expected) / expected
        assert rel < 0.06, f"L={L},C={C}: wave speed {measured:.3f} != 1/sqrt(LC)={expected:.3f}"
        print(f"  wave speed L={L},C={C}: OK  (measured {measured:.3f}, 1/sqrt(LC)={expected:.3f})")

    # 3. a lossless line (R=G=0), open both ends, conserves energy
    tl = TransmissionLine(n=500, sc=0.9)
    tl.source(rs=np.inf)                 # open source end too -> closed lossless resonator
    tl.load(np.inf)
    tl.launch_pulse(center=0.5, width=0.03, amp=1.0)
    U = [tl.energy() for _ in range(1) ]
    for _ in range(4000):
        tl.step()
        U.append(tl.energy())
    U = np.array(U[20:])
    assert np.all(np.isfinite(U)), "energy non-finite (unstable)"
    rel = (U.max() - U.min()) / U.mean()
    drift = abs(U[-1] - U[:200].mean()) / U[:200].mean()
    assert rel < 0.06 and drift < 0.03, f"energy not conserved (spread {rel:.3f}, drift {drift:.3f})"
    print(f"  energy conservation: OK  (spread {rel*100:.2f}%, drift {drift*100:.2f}% over 4000 steps)")

    # 4. the trace_signal scene renders the solved signal on the real board, and animates
    import warp as wp
    import warp_shaders as ws
    wp.init()
    a = np.asarray(ws.render("trace_signal", width=120, height=96, time=2.0), np.float32)
    b = np.asarray(ws.render("trace_signal", width=120, height=96, time=5.0), np.float32)
    assert np.all(np.isfinite(a)) and a.max() > 0.1 and a.std() > 0.01, "trace_signal: bad frame"
    assert np.abs(a - b).mean() > 1e-3, "trace_signal: signal did not animate"
    print("  scene trace_signal: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
