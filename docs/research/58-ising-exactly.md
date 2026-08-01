# The Ising model, exactly — Onsager, Yang, Kramers-Wannier, live

The square-lattice Ising ferromagnet is the crown jewel of exactly-solvable
statistical mechanics: a genuinely interacting system with a genuine phase
transition whose critical point, energy, and order parameter are all **closed
forms**. This round puts the theorems and a **live simulation** in the same
frame: seeded, deterministic Metropolis dynamics running on the Warp lattice,
measured against the exact solution — in the scenes *and* in the test suite.
Measured, not asserted — twice over. Engine: `warp_shaders/engine/ising.py`.
Scenes: `ising_quench`, `ising_magnetization`, `ising_duality`.

## The dictionary

```
T_c = 2/ln(1+√2) ≈ 2.269185        Onsager 1944 (self-duality sinh(2/T_c) = 1 — machine-exact, asserted)
M(T) = (1 − sinh(2/T)^-4)^(1/8)     Yang 1952 — spontaneous magnetization below T_c, EXACTLY 0 above
β = 1/8                             the exact critical exponent (log-slope asserted = 0.12500)
U = −coth(2K)[1 + (2/π)κ′K₁(κ)]     Onsager's energy per spin (elliptic K₁ by AGM)
U(T→0) = −2,  U(T_c) = −√2          ground state; the elliptic coefficient vanishes AT criticality (asserted)
sinh(2/T)·sinh(2/T*) = 1            Kramers-Wannier 1941 — order and disorder are the same model read twice
```

Conventions: J = k_B = 1, K = 1/T, κ = 2 sinh(2K)/cosh²(2K), κ′ = 2tanh²(2K) − 1.

## Three theorems

**Onsager's critical point comes from symmetry alone.** Kramers and Wannier
noticed (1941 — three years *before* the solution) that the model's
high-temperature expansion and low-temperature expansion are the *same series*:
each configuration of excited bonds at temperature T maps to a configuration of
flipped domains at the dual temperature T\* with `sinh(2/T)·sinh(2/T*) = 1`.
The map is an involution (asserted), so if the model has exactly one critical
point it must sit at the fixed point: `sinh(2/T_c) = 1`, i.e.
`T_c = 2/ln(1+√2)` (asserted at machine precision). The critical temperature
was known before anyone could compute anything at it.

**Onsager's energy is elliptic — and simple exactly at criticality.** The
internal energy per spin involves the complete elliptic integral K₁(κ)
(computed here by the arithmetic-geometric mean — quadratic convergence, ~8
iterations to machine precision). Its modulus κ reaches 1 exactly at T_c —
where the *coefficient* `2tanh²(2K) − 1` simultaneously vanishes, taming the
logarithmic divergence into the famous log-singular specific heat and leaving
the clean value `U(T_c) = −coth(2K_c) = −√2` (asserted at 10⁻⁹). Ground state
`U → −2` and high-temperature tail `U ≈ −2/T` both asserted.

**Yang's magnetization has an eighth-root.** The spontaneous magnetization is
`M = (1 − sinh(2/T)^{−4})^{1/8}` — an exact 1/8 power, not the mean-field 1/2:
fluctuations in two dimensions are strong enough to reshape the exponent, and
the exact solution proves it. The log-slope of the closed form is asserted to
be 0.12500. Above T_c, M is exactly zero — no analytic continuation, a genuine
broken symmetry.

## The simulation meets the theorems

Every scene runs real Metropolis dynamics — seeded and deterministic, so any
frame is replayable — and the suite runs the same dynamics headless and asserts
the measurement lands on the closed form (|M_sim(1.5) − Yang| < 0.03; disorder
above T_c).

* **`ising_quench`** — a 256² lattice on the Warp lattice, swept hot → critical
  → cold → reheat. Paramagnetic snow; then, approaching T_c, correlated
  clusters at *every* scale (the critical point is the only temperature with no
  characteristic size); then symmetry breaking and domain coarsening. The
  ledgers put the theorem next to the measurement: amber T against the white
  T_c line, cyan Yang's exact M(T), magenta the live block-local order
  parameter chasing it. A fast quench traps opposite domains — the global mean
  would hide the order each domain carries inside, so the live measurement is
  block-local: honest non-equilibrium physics.
* **`ising_magnetization`** — the money plot: Yang's curve in cyan, Onsager's
  T_c in white, and nine independent live lattices as amber dots at
  (T_k, |M_k|), rising from random initial disorder and *landing on the
  theorem* as they equilibrate. The cyan ledger is the mean |simulation − Yang|
  error, and it shrinks (structurally asserted).
* **`ising_duality`** — Kramers-Wannier live: two lattices, the left at T
  sweeping cold → hot → cold, the right always at the dual T\*. When one is
  ordered the other is disordered; they cross T_c *together* — the only
  rendezvous the map allows. The magenta ledger is the live product
  `sinh(2/T)·sinh(2/T*)`, pinned at 1 under its white line, never moving.

## Sources

- L. Onsager, *Crystal statistics I*, Phys. Rev. 65, 117 (1944) — the solution
- C. N. Yang, *The spontaneous magnetization of a two-dimensional Ising model*,
  Phys. Rev. 85, 808 (1952) — the order parameter (announced by Onsager in 1949
  on a conference blackboard, unproven)
- H. A. Kramers, G. H. Wannier, *Statistics of the two-dimensional ferromagnet*,
  Phys. Rev. 60, 252 (1941) — duality and the critical point
- B. McCoy, T. T. Wu, *The Two-Dimensional Ising Model* (1973) — the reference
- Research 31 (`31-states-of-matter.md`) — this repo's phase-transition strand
  (boiling, crystallization, Bose-Einstein); the Ising round is its exact core
