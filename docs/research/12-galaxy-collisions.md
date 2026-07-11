# Research 12 — Colliding galaxies: bridges and tails

Sources and reasoning behind `warp_shaders/cosmos/galaxy_dynamics.py` — two
galaxies passing close enough to raise **tidal tails and bridges**, the largest
set-piece in the cosmos strand. Where [Research 10](10-solar-system.md) put a few
bodies on Kepler orbits and [Research 11](11-stellar-evolution.md) put one star on
a clock, this puts *thousands* of stars in a gravitational encounter between two
whole galaxies — the Antennae / Mice look.

## The restricted N-body model (Toomre & Toomre 1972)

The classic result is Alar & Juri Toomre's 1972 paper *"Galactic Bridges and
Tails"* ([ApJ 178, 623](https://www.giss.nasa.gov/pubs/abs/to03000u.html);
[Wikipedia](https://en.wikipedia.org/wiki/Galactic_Bridges_and_Tails)), which
showed for the first time that the long filaments seen in interacting galaxies are
just **tidal relics of a close encounter** — no exotic physics needed. Their
model is beautifully cheap, which is exactly why it renders in real time:

- **Two point-mass cores.** Each galaxy is idealized as a single massive point
  (its bulge + halo). The two cores attract *each other* with full gravity and
  move on a roughly **parabolic** fly-by orbit — approach, pericenter, recede.
- **Massless test-particle disks.** Each galaxy's visible stars are a disk of
  **noninteracting test particles** on initially **circular** orbits around their
  own core. They feel the gravity of *both* cores but exert **no** force — on each
  other or on the cores. So the cost is `N` independent two-body-in-a-moving-field
  integrations, not an `O(N²)` N-body.
- **The encounter geometry decides everything.** Whether a galaxy grows a
  spectacular tail depends on the **inclination** of its disk to the orbit plane
  and its **spin sense**: a **prograde** disk (spinning the same way the companion
  swings by) throws out a long, graceful tail and a bridge toward the companion;
  a **retrograde** disk barely responds. The Antennae's twin tails come from two
  near-prograde disks.

## Why tails and bridges form

At pericenter the companion's tide stretches each disk along the line to the
companion. Particles on the **near** side are pulled *toward* the companion into a
**bridge**; particles on the **far** side are left behind and, because the outer
disk orbits slowest (Keplerian `v ∝ r^-1/2`), they lag into a long trailing
**tail**. Prograde particles stay in resonance with the passing companion for
longer, so their response is huge; retrograde particles are swept through quickly
and barely move. The tails are **kinematic** — thin streams of stars on diverging
orbits, not a shock or a wave.

## How the simulation is set up

`galaxy_dynamics.py` builds the encounter as data:

- **Cores.** Two masses `m1, m2` with a **softened** gravitational acceleration
  `a = -G·m·r / (|r|² + ε²)^(3/2)` (the softening `ε` avoids the singularity at
  close approach). They start apart with velocities set for a near-parabolic
  approach, and are advanced with **velocity-Verlet** (symplectic, energy-stable).
- **Disks.** For each galaxy, `N` particles are laid on rings from `r_in` to
  `r_out`, each on a circular orbit `v = √(G·m_core / r)` in a plane tilted by the
  disk's **inclination** and given a **prograde/retrograde** spin. Each particle is
  advanced under the summed softened pull of *both* cores.
- **Time.** The whole fly-by is normalized to `t ∈ [0, 1]`: the galaxies approach,
  reach pericenter around the middle, and the tails unfurl toward the end.

The output is just particle positions per frame — the renderer
([Research: rendering, below]) projects them and splats glowing points, coloured
per galaxy, over a starfield.

## Rendering the star clouds

Thousands of point stars are drawn by **projecting** each particle through the
camera and **additively splatting** it into an HDR buffer (host-side, so the cost
is `O(N)` not `O(pixels × N)`); the post **bloom** pass then blooms the points
into the soft luminous haze of a galaxy. The two disks carry distinct blackbody
palettes (a warm-white and a cooler blue), the cores are bright bulges, and the
tails inherit their disk's colour so the tidal debris is legible as "stars torn
from *that* galaxy". Starfield behind, ACES tonemap in front.

## Cross-references

- [Research 10 — the solar system](10-solar-system.md): Kepler orbits, softened
  gravity, velocity-Verlet, and the starfield/post this reuses.
- Toomre & Toomre 1972: [NASA GISS abstract](https://www.giss.nasa.gov/pubs/abs/to03000u.html),
  [Wikipedia](https://en.wikipedia.org/wiki/Galactic_Bridges_and_Tails),
  [NRAO Antennae notes](https://www.cv.nrao.edu/~jhibbard/students/CPower/numerical/num_antennae/tt72.html).
