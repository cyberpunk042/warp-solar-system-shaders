# The universe as the ultimate horizon — de Sitter space, inflation, and the thermal sky

Every horizon so far (research 44→50) belonged to a black hole: a place *over there*
you could fall into. This set inverts it. In a universe dominated by a cosmological
constant — ours, increasingly — the horizon wraps **around the observer**: light from
beyond proper radius `1/H` never arrives, and Gibbons & Hawking showed in 1977 that this
inside-out horizon obeys the same thermodynamics as a black hole's. New engine module:
`warp_shaders/engine/desitter.py` (all closed forms test-asserted). Scenes:
`ds_horizon`, `ds_inflation`, `ds_thermal`.

## The dictionary

```
a(t) = e^{Ht}                     the de Sitter expansion
r_H = 1/H                        the cosmic event horizon, per observer
T = H/2π                          Gibbons-Hawking temperature      (asserted)
S = A/4 = π/H²                    horizon entropy                  (S = A/4 asserted)
χ_EH(t) = e^{−Ht}/H               comoving horizon — SHRINKS        (asserted)
t_k = ln(k/H)/H                   mode k exits at k = aH: log-spaced (asserted)
n_s = 1 + 2η − 6ε                 slow-roll tilt; P(k) ∝ k^{n_s−1}  (power law asserted)
1+z = 1/√(1 − H²r²)               static-patch redshift; diverges at r_H (asserted)
```

For our universe: `T ≈ 10⁻³⁰ K` (real, unmeasurably cold) and `S ≈ 10¹²²` — the famous
number, the largest entropy our observable patch will ever hold (Bousso's bound).

## Part I — the black hole turned inside out: `ds_horizon`

Watched from the centre of your patch, a galaxy carried off by the Hubble flow does
*exactly* what an infalling astronaut does when watched from outside a black hole: it
redshifts (`1+z = 1/√(1−H²r²)`, diverging **at** the horizon), slows, freezes, and
fades — never seen to cross. The scene rides two dozen comoving galaxies out of the
flow, colours each by its exact redshift, and then lets **dark energy strengthen**:
H ramps up, the horizon `1/H` **contracts**, its Gibbons-Hawking glow warms, and the
sky empties early. The final frame is the far future of an accelerating universe: a
faint warm ring, a lone observer, and nothing left to see.

## Part II — the origin of everything: `ds_inflation`

Inflation's mechanism is invisible in proper coordinates and obvious in comoving ones,
so the scene works in comoving coordinates: the grid points *never move* — the horizon
does. During inflation `χ_EH = e^{−Ht}/H` collapses exponentially past the standing
quantum waves; a mode freezes the instant its scale no longer fits (`k = aH`), largest
first, and because the scales are octave-spaced they freeze at **equal time intervals**
— the asserted logarithmic law, visible as a metronome of freeze-outs. Each locks at
the exact tilted amplitude `k^{(n_s−1)/2}` with `n_s = 1 + 2η − 6ε ≈ 0.964` — the red
tilt Planck actually measures, the fingerprint of the inflaton rolling downhill.

Then reheating flashes, the horizon regrows, the frozen modes **re-enter** — and where
their interference is constructive, matter condenses: proto-galaxies precipitate out
of the pattern, weighted by the exact spectrum. That is the full sentence: *the CMB's
speckle and every galaxy in the sky are vacuum fluctuations, frozen at horizon exit
and worn at cosmological size.* (Mukhanov–Chibisov 1981, the first calculation of it.)

## Part III — the thermal sky and the last ledger: `ds_thermal`

Gibbons–Hawking: an inertial observer in empty Λ-space measures a thermal bath at
`T = H/2π` coming *from their own horizon*, which carries `S = π/H²`. The scene sweeps
the cosmological constant up and back, streaming thermal quanta inward from the ring
while cyan **entropy tiles** around it count `S = A/4` live — and exposes de Sitter's
strangest clause, inverted from naive intuition but exactly parallel to black holes:
**hotter means smaller means less entropy**. Feed Λ and you warm your bath, shrink
your world, and shrink the total information it can ever contain.

The arc closes where it started: research 44 opened with a horizon whose area counts
entanglement; this one ends with the horizon that counts *us* — the bound on
everything our patch of universe will ever know.

## Sources

- G. Gibbons, S. Hawking, *Cosmological event horizons, thermodynamics, and particle
  creation*, PRD 15, 2738 (1977)
- A. Guth, *Inflationary universe*, PRD 23, 347 (1981)
- V. Mukhanov, G. Chibisov, *Quantum fluctuations and a nonsingular universe*, JETP
  Lett. 33, 532 (1981) — the origin of structure
- T. Bunch, P. Davies, *Quantum field theory in de Sitter space*, Proc. R. Soc. A 360,
  117 (1978) — the vacuum the modes freeze out of
- R. Bousso, *The holographic principle*, Rev. Mod. Phys. 74, 825 (2002),
  [hep-th/0203101](https://arxiv.org/abs/hep-th/0203101)
- Planck Collaboration, *Planck 2018 results. X. Constraints on inflation*,
  [1807.06211](https://arxiv.org/abs/1807.06211) — n_s = 0.9649 ± 0.0042
