"""The VRM — a multiphase buck converter, solved in time (simulation of reality, B3).

The board turns the 12 V from the connector into the ~1 V core rail the die drinks at hundreds of
amps, through a bank of ~20 interleaved **synchronous buck converters** (one "phase" = a driver + a
high/low MOSFET pair + a power inductor). Each phase is a switching converter whose two state
variables are the **inductor current** i_L and the **output-capacitor voltage** v_C, governed by

    L di_L/dt = s(t)·Vin − v_C − i_L·R          (switch on: s=1 → Vin−v_C ; off: s=0 → −v_C)
    C dv_C/dt = Σ i_L − v_C / R_load             (all phases feed one shared output cap)

with the switch s(t) a PWM pulse at duty D and frequency f_sw. The phases are **interleaved** — phase
k fires at offset k/N of the cycle — so their ripple currents partially cancel and the output ripple
runs at N·f_sw. We integrate with **symplectic (semi-implicit) Euler**: update i_L from the old v_C,
then v_C from the *new* i_L, so the L–C tank conserves its energy honestly (plain forward Euler on an
oscillator gains energy and blows up — the wrong physics).

Verification (``tests/test_vrm.py``): steady state converges to the ideal **V_out = D·Vin**; the
inductor ripple matches **ΔI_L = (Vin−V_out)·D / (L·f_sw)**; a lossless L–C tank conserves
½L·i² + ½C·v²; and N interleaved phases cut the output ripple versus a single phase.
"""

import numpy as np


class BuckConverter:
    """A (multiphase, interleaved) synchronous buck converter — switching-ODE state stepper."""

    def __init__(self, n_phases=1, Vin=12.0, D=0.5, L=1.0, C=100.0, Rload=4.0,
                 R=0.0, fsw=1.0, dt=1.0e-3):
        self.n = int(n_phases)
        self.Vin, self.D = float(Vin), float(D)
        self.L, self.C, self.Rload, self.R = float(L), float(C), float(Rload), float(R)
        self.fsw, self.dt = float(fsw), float(dt)
        self.iL = np.zeros(self.n, np.float64)     # per-phase inductor current
        self.vC = 0.0                              # shared output-capacitor voltage
        self.t = 0.0

    def switch(self, k):
        """PWM state of phase k at the current time (1 = high-side on), interleaved by k/N."""
        ph = (self.t * self.fsw + k / self.n) % 1.0
        return 1.0 if ph < self.D else 0.0

    def phase_currents(self):
        return self.iL.copy()

    def step(self):
        dt, L, C = self.dt, self.L, self.C
        # symplectic: inductor currents first (using the OLD v_C)
        for k in range(self.n):
            s = self.switch(k)
            vL = s * self.Vin - self.vC - self.iL[k] * self.R
            self.iL[k] += dt / L * vL
        # then the output cap from the NEW total current
        gl = 0.0 if not np.isfinite(self.Rload) else 1.0 / self.Rload
        self.vC += dt / C * (np.sum(self.iL) - self.vC * gl)
        self.t += dt

    def run(self, steps):
        for _ in range(int(steps)):
            self.step()
        return self.vC

    def energy_lc(self):
        """Tank energy ½L·Σi² + ½C·v² — conserved on a lossless free L–C (Vin=0, R=0, Rload=∞)."""
        return float(0.5 * self.L * np.sum(self.iL * self.iL) + 0.5 * self.C * self.vC * self.vC)

    def run_record_vout(self, steps):
        """Advance, returning the v_C trace (for steady-state / ripple measurement)."""
        out = np.empty(int(steps), np.float64)
        for i in range(int(steps)):
            self.step()
            out[i] = self.vC
        return out

    def run_record_iL(self, steps, k=0):
        out = np.empty(int(steps), np.float64)
        for i in range(int(steps)):
            self.step()
            out[i] = self.iL[k]
        return out
