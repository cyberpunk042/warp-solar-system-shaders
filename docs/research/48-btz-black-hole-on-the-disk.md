# The BTZ black hole on the disk — quotient, plateau, ringdown

The AdS/CFT arc ([research 46](46-ads-cft-holography.md)) ran on two stages: the Poincaré
disk (the map) and the ray-traced AdS box (the territory). This set unites them in the one
geometry where everything is exactly solvable: **AdS₃ and its black hole** — the
Bañados–Teitelboim–Zanelli (BTZ) hole. Three scenes, three closed-form stories:
`btz_quotient` (the hole is an identification), `btz_entanglement` (the entanglement
plateau), `btz_ringdown` (the only black hole whose ringing is known exactly).

## Part I — a black hole with no curvature: `btz_quotient`

Gravity in 2+1 dimensions has **no local degrees of freedom** — no gravitational waves, no
Newtonian attraction, every solution locally isometric to pure AdS₃. It shocked everyone
in 1992 when Bañados, Teitelboim and Zanelli found a black hole in it anyway
([hep-th/9204099](https://arxiv.org/abs/hep-th/9204099); geometry in
[gr-qc/9302012](https://arxiv.org/abs/gr-qc/9302012)),
because a BTZ hole is not curvature: it is **global identification**. Quotient the
Poincaré-disk time slice by a single hyperbolic isometry — a translation by hyperbolic
distance λ along a geodesic axis — and the axis becomes a **closed geodesic: the
horizon**, with

```
f(r) = −M + r²/L²      r_h = L·√M       T = r_h/(2πL²)       S = 2πr_h/4G = λL/4G
```

(engine: `btz_horizon_radius`, `btz_temperature`, `btz_entropy`,
`horizon_translation_length` — all exact, no bisection needed in 3D). The entropy is
*literally a length on the disk*, the length Strominger later reproduced exactly by
counting CFT states with the Cardy formula
([hep-th/9712251](https://arxiv.org/abs/hep-th/9712251)).

The scene draws the construction live: the horizon axis glowing across the disk, the
fundamental-domain walls — geodesics orthogonal to the axis at `x_n = tanh(n·λ/2)`
(engine: `quotient_wall_position`, monotonicity test-asserted) — **accumulating at the
isometry's two boundary fixed points**, and one wall-to-wall strip tinted: that strip is
the *entire* black-hole exterior; every other strip is the same universe again. The hole
breathes once per cycle, and the walls spread as it grows: entropy as geometry, watched.

## Part II — the entanglement plateau: `btz_entanglement`

Ryu–Takayanagi at finite temperature. A boundary interval of size θ on the BTZ boundary
has the finite-T Calabrese–Cardy entropy (engine: `thermal_interval_entropy`):

```
S_th(θ) = (c/3)·ln( (β/πε)·sinh(πθ/β) )
```

with two honest limits, both test-asserted: the **vacuum log** `(c/3)ln(θ/ε)` for θ ≪ β
(UV physics is blind to temperature), and **extensivity** `S → (πc/3)·T·θ` for θ ≫ β —
the interval stops measuring geometry and just counts thermal excitations.

But there is a second candidate surface, because RT carries a **homology constraint**:
the surface must be deformable to the interval without crossing the horizon. The interval
can be capped by its own geodesic (**direct**, `S_th(θ)`) — or by the *complement's*
geodesic plus the **horizon itself**, which the surface is forced to wrap
(**wrapped**, `S_th(2π−θ) + S_BH`). RT takes the minimum (engine:
`thermal_entanglement`) — the **fourth saddle competition of the arc**, after mutual
information, Hawking–Page and the island. The swap at the plateau angle θ*
(engine: `plateau_angle`, bisected; θ* = 5.149 for the scene's parameters,
test-asserted) is the **entanglement plateau** of Hubeny–Maxfield–Rangamani–Tonni
([1306.4004](https://arxiv.org/abs/1306.4004)): past θ*, S(θ) freezes at the wrapped
value, and the Araki–Lieb inequality `|S(A) − S(Ā)| ≤ S_BH` is **saturated exactly**
(asserted to 1e-9 across the plateau) — the interval's entanglement wedge now contains
the whole black hole.

The scene sweeps θ through θ* once per cycle: the direct geodesic hangs bright, then at
the plateau the drawing swaps — the complement's arc and the **horizon ring ignite
violet** and the wedge tint floods everything outside the complement's little cap. The
black hole belongs to the interval. (The arcs are pure-AdS orthocircles — schematic in
the BTZ metric; the entropies, θ*, and the saturation are the honest closed forms.)

## Part III — the black hole that rings in closed form: `btz_ringdown`

Kick a black hole and it rings — damped quasinormal oscillations, the signal LIGO hears
as merged holes settle. For Schwarzschild and Kerr the QNM spectrum is numerical. For BTZ
it is **exact** (Birmingham–Sachs–Solodukhin,
[hep-th/0112055](https://arxiv.org/abs/hep-th/0112055); engine: `btz_qnm`):

```
ω = ±k − 4πi·T·(n + Δ/2)
```

Three theorems in one line, all test-asserted: the oscillation frequency is just the
momentum k; the damping rate is set *only* by the temperature; the overtones are spaced
exactly `4πT`. And by the dictionary these frequencies are precisely the **poles of the
boundary CFT's retarded thermal correlator** (Horowitz–Hubeny,
[hep-th/9909056](https://arxiv.org/abs/hep-th/9909056)): the rate at which the horizon
settles IS the rate at which the dual plasma thermalizes. Ringdown = thermalization —
one number, two languages. It closes the loop with Part VIII of research 46: *this* is
the fast boundary clock against which the interior's complexity keeps growing.

The scene kicks the horizon with a k = 2 quadrupole once per cycle: the outline rings at
`ω_re = k` and damps as `e^{−4πT(Δ/2)t}`, cyan ripples carry the perturbation outward
along retarded time, and the boundary ring flickers and settles at exactly the same
rate, until the disk is quiet and the next kick lands.

## The set

With these three, the holography shelf reads: `ads_cft` (the map), `ads_bulk` (the
territory), `ads_hawking_page` (the ensemble), `ads_entanglement` (geometry from
entanglement), `ads_wormhole` (ER=EPR), `ads_confinement` (the plasma screen),
`ads_page_curve` (unitarity), `ads_complexity` (the interior clock) — and now
`btz_quotient` / `btz_entanglement` / `btz_ringdown`: the one spacetime where every one
of those stories can be told in exact closed form.

## Sources

- M. Bañados, C. Teitelboim, J. Zanelli, *The Black Hole in Three Dimensional Space-Time*,
  [hep-th/9204099](https://arxiv.org/abs/hep-th/9204099); with M. Henneaux, *Geometry of
  the 2+1 black hole*, [gr-qc/9302012](https://arxiv.org/abs/gr-qc/9302012)
- A. Strominger, *Black Hole Entropy from Near-Horizon Microstates*,
  [hep-th/9712251](https://arxiv.org/abs/hep-th/9712251) — Cardy counting of S_BTZ
- P. Calabrese, J. Cardy, *Entanglement Entropy and Quantum Field Theory*,
  [hep-th/0405152](https://arxiv.org/abs/hep-th/0405152) — the finite-T interval entropy
- V. Hubeny, H. Maxfield, M. Rangamani, E. Tonni, *Holographic entanglement plateaux*,
  [1306.4004](https://arxiv.org/abs/1306.4004)
- D. Birmingham, I. Sachs, S. N. Solodukhin, *Conformal Field Theory Interpretation of
  Black Hole Quasi-normal Modes*, [hep-th/0112055](https://arxiv.org/abs/hep-th/0112055)
- G. Horowitz, V. Hubeny, *Quasinormal Modes of AdS Black Holes and the Approach to
  Thermal Equilibrium*, [hep-th/9909056](https://arxiv.org/abs/hep-th/9909056)
