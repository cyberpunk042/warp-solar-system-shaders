# The vacuum is alive — Unruh, Casimir, Schwinger, and the third temperature

Seven rounds of horizons produced two of physics' great temperatures: Hawking's
`T = κ/2π` ([research 45](45-ads-cft-engine-holography.md)) and Gibbons–Hawking's
`T = H/2π` ([research 51](51-desitter-cosmic-horizon.md)). This set delivers the third
— and with it the punchline that none of them were ever about gravity. Empty space is
a physical medium: accelerate through it and it glows; bound it and it pushes; stress
it and it short-circuits. New engine module: `warp_shaders/engine/vacuum.py` (all
closed forms test-asserted). Scenes: `unruh_horizon`, `casimir_plates`,
`schwinger_pairs`.

## The dictionary

```
T = a/2π                          Unruh: acceleration alone makes heat (asserted)
κ/2π = H/2π = a/2π                the temperature trilogy — one identity (asserted)
x² − t² = 1/a²                    the accelerated worldline (identity asserted)
d_horizon = 1/a                   your private horizon trails right behind you
P = −π²/240·d⁻⁴                   Casimir: halve the gap → 16× (asserted)
E/A = −π²/720·d⁻³                 with P = −∂(E/A)/∂d (asserted)
k_n = nπ/d                        the modes that survive between mirrors
Γ ∝ E²·exp(−πE_c/E), E_c = m²/e   Schwinger: the vacuum's breakdown voltage
```

## Part I — the third temperature: `unruh_horizon`

Unruh 1976 (with Fulling and Davies): an observer under uniform proper acceleration
`a` does not see the Minkowski vacuum as empty — they measure a thermal bath at
`T = a/2π`. The scene plays it on a live spacetime diagram. The observer rides the
exact hyperbola `x² − t² = 1/a²` (identity asserted), forever confined to the right
wedge; the null line `x = t` is their **Rindler horizon** — a horizon with no black
hole anywhere — trailing at proper distance `1/a`. As the acceleration ramps, the
hyperbola tightens into the corner, the private horizon closes in, and the bath around
the observer brightens as exactly `a/2π`.

The suite asserts the trilogy numerically: `unruh_temperature(x) ==
gibbons_hawking_temperature(x)` — Hawking, Gibbons–Hawking and Unruh are one theorem
wearing three metrics. *A horizon plus quantum fields equals heat.* The event horizon
of a black hole is, locally, just the Rindler horizon of the observers hovering
outside it; Hawking radiation is the Unruh bath of the hoverers, redshifted to
infinity.

## Part II — the vacuum pushes: `casimir_plates`

Casimir 1948: between two ideal mirrors, only standing waves with `k_n = nπ/d` exist;
outside, the full continuum. The imbalance in zero-point pressure attracts the plates:

```
P = −π²/240 · 1/d⁴
```

The fourth power is savage — halve the gap, sixteenfold the force (asserted exactly,
along with the thermodynamic identity `P = −∂(E/A)/∂d`). And it is not a philosophy
seminar: measured to ~1% (Lamoreaux 1997, torsion pendulum; Mohideen–Roy 1998, AFM),
and a genuine failure mode in MEMS engineering, where it snaps micro-cantilevers shut.
The scene squeezes the gap while `allowed_modes` counts the survivors: the modes are
**evicted one by one**, and the eviction *is* the force — the amber arrows grow by the
exact `1/d⁴`.

## Part III — the vacuum breaks down: `schwinger_pairs`

Sauter 1931, Schwinger 1951: a strong electric field does work on the virtual e⁺e⁻
pairs of the vacuum; when the work across a Compton wavelength rivals `2m` — at
`E_c = m²/e ≈ 1.3×10¹⁸ V/m` — the pairs come real, at the non-perturbative rate

```
Γ ∝ E² · exp(−π E_c/E).
```

The exponential makes the vacuum a near-perfect insulator with a sharp breakdown
voltage: at `E_c/10` the rate is suppressed by `e^{−10π} ≈ 10⁻¹⁴` (asserted) —
nothing — and above `E_c`, an avalanche whose created charges screen and collapse the
very field that made them. The scene charges the capacitor on the exact rate curve:
silence, silence, a drizzle near `0.6 E_c`, avalanche, breakdown flash, quiet. No
laboratory field has reached `E_c` yet — the ELI-class petawatt lasers are built to
close that gap — but the same physics *has* been seen in analogues, and Hawking
radiation is its gravitational twin: pair production in a gravitational, rather than
electric, potential drop.

One sentence closes the arc: **horizon heat, Casimir push and vacuum breakdown are the
same lesson — the vacuum is not the absence of everything; it is the thing that
remains.**

## Sources

- W. G. Unruh, *Notes on black-hole evaporation*, PRD 14, 870 (1976); S. Fulling
  (1973), P. Davies (1975)
- H. B. G. Casimir, *On the attraction between two perfectly conducting plates*,
  Proc. K. Ned. Akad. Wet. 51, 793 (1948)
- S. Lamoreaux, *Demonstration of the Casimir force in the 0.6 to 6 μm range*, PRL
  78, 5 (1997); U. Mohideen, A. Roy, PRL 81, 4549 (1998)
- F. Sauter, Z. Phys. 69, 742 (1931); J. Schwinger, *On gauge invariance and vacuum
  polarization*, Phys. Rev. 82, 664 (1951)
- L. Crispino, A. Higuchi, G. Matsas, *The Unruh effect and its applications*, Rev.
  Mod. Phys. 80, 787 (2008), [0710.5373](https://arxiv.org/abs/0710.5373)
- A. Fedotov et al., *Advances in QED with intense background fields*, Phys. Rep.
  1010, 1 (2023), [2203.00019](https://arxiv.org/abs/2203.00019) — the ELI-era review
