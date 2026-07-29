# The chirp — gravitational waves from the closed forms that measured them

Every prior round drew spacetimes standing still (or in eternal equilibrium). This set
draws spacetime *ringing*: binary inspiral at leading (quadrupole / 2.5PN) order,
where — remarkably — everything LIGO needed to weigh its first black holes is closed
form. Einstein 1916/1918, Peters & Mathews 1963, Peters 1964; heard 2015-09-14. New
engine module: `warp_shaders/engine/gw.py` (all laws test-asserted). Scenes:
`gw_inspiral`, `gw_chirp`, `gw_orbits`. Units G = c = 1.

## The dictionary

```
M_c = (m₁m₂)^{3/5}/M^{1/5}          the chirp mass — what the waveform measures
f_gw = 2·f_orb                       quadrupole: two arms, twice the frequency (asserted)
T = 5a⁴/256·m₁m₂M                    merger time — halve a, 16× less time (asserted)
a(t) = (256/5·m₁m₂M·(t_c−t))^{1/4}   the inspiral trajectory (inversion asserted)
f(t) ∝ M_c^{−5/8}(t_c−t)^{−3/8}      THE CHIRP (exponent asserted; = Kepler on a(t), asserted)
h ∝ M_c^{5/3}f^{2/3}/D               louder as it climbs
da/dt, de/dt (Peters 1964)           exact circularization (asserted by integration)
```

## Part I — the last orbits: `gw_inspiral`

The pair rides the exact trajectory `a(t) = (256/5·m₁m₂M·(t_c−t))^{1/4}` — the
inversion of Peters' fourth-power merger time, asserted — while Kepler's `Ω = √(M/a³)`
spins it ever faster (closed-form phase `φ ∝ (t_c−t)^{5/8}`). The radiated pattern is
a **two-armed spiral**: gravity radiates through the quadrupole, and a binary's mass
distribution repeats every *half* orbit, so `f_gw = 2f_orb` (asserted) and the pattern
grows two arms. Amplitude climbs with pitch as `h ∝ M_c^{5/3}f^{2/3}`. Merger, flash,
damped ringdown ripples ([the BTZ round](48-btz-black-hole-on-the-disk.md) did the
full QNM story), re-arm.

The fourth power is the whole drama of compact binaries: Hulse–Taylor's pulsar
(a ≈ R_sun) has ~300 Myr left; the same masses at a thousand km would have minutes.
LIGO's audible second is the trajectory's last breath.

## Part II — the waveform: `gw_chirp`

The most famous curve in modern physics, *generated*, not sampled. Top panel: the
frequency track

```
f_gw(t) = (5/256)^{3/8} / (π·M_c^{5/8}·(t_c−t)^{3/8})
```

— flat for ages, then screaming upward; the −3/8 exponent is asserted numerically,
and so is the consistency check that this law is nothing but Kepler evaluated on the
Peters trajectory (`chirp_frequency(t) ≡ f_gw(a(t))` to 10⁻⁹). Bottom panel: the
strain `h(t)` with the closed-form phase and the amplitude rising as `f^{2/3}` —
faster AND louder together, which is what "chirp" means — then the post-merger damped
ringdown. A sweep line reveals both curves as the cycle plays; nothing is
hand-animated.

The physics punchline: the track's *shape* is set by `M_c` alone. Read the sweep rate
at any frequency and the chirp mass falls out — GW150914's `M_c ≈ 30 M_sun` (hence two
~35+30 M_sun black holes) was measured exactly this way, from a curve first written
down four decades before the detector worked.

## Part III — why the orbits are round: `gw_orbits`

Peters 1964, exact at leading order:

```
da/dt = −64/5·m₁m₂M/a³·(1−e²)^{−7/2}(1 + 73e²/24 + 37e⁴/96)
de/dt = −304/15·e·m₁m₂M/a⁴·(1−e²)^{−5/2}(1 + 121e²/304)
```

Emission peaks at pericenter — closest, fastest — so each passage bleeds away exactly
the motion that made the orbit eccentric. The scene integrates the coupled system
(RK2) and plays it: the live ellipse `r = a(1−e²)/(1+e·cosν)` shrinking through ghost
epochs, the body flaring orange at pericenter where the radiation leaves, amber and
cyan ledger bars tracking `e` and `a` off the integration. The suite asserts both
monotonicities and the net circularization (`e/e₀ < a/a₀` over the inspiral) — with a
subtlety the bars display honestly: at high eccentricity the `(1−e²)^{−7/2}`
enhancement makes `a` decay *fractionally faster*, and `e` only wins the race late.
Integrated to the end, the waves erase the orbit's memory: LIGO's binaries arrive
round to a part in a thousand. Hulse–Taylor (`e = 0.617` today, orbital decay matching
Peters to 0.2% — the 1993 Nobel) is mid-flight on exactly this curve.

## Sources

- A. Einstein, *Über Gravitationswellen*, Sitzungsber. Preuss. Akad. Wiss. (1918) —
  the quadrupole formula
- P. C. Peters, J. Mathews, *Gravitational radiation from point masses in a Keplerian
  orbit*, Phys. Rev. 131, 435 (1963)
- P. C. Peters, *Gravitational radiation and the motion of two point masses*, Phys.
  Rev. 136, B1224 (1964) — da/dt, de/dt, the merger time
- R. A. Hulse, J. H. Taylor, ApJ 195, L51 (1975); J. M. Weisberg, Y. Huang, ApJ 829,
  55 (2016), [1606.02744](https://arxiv.org/abs/1606.02744) — the binary pulsar
- B. P. Abbott et al. (LIGO/Virgo), *Observation of gravitational waves from a binary
  black hole merger*, PRL 116, 061102 (2016),
  [1602.03837](https://arxiv.org/abs/1602.03837) — GW150914
- M. Maggiore, *Gravitational Waves, Vol. 1* (OUP 2008) — the leading-order derivations
