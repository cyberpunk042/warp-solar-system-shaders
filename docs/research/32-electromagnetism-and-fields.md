# Research 32 — Electromagnetism & fields

> The invisible forces drawn. Field lines, dipoles, propagating waves, and the way
> charges and currents shape the space around them — Maxwell's world made visible.

## Fields & field lines

A **field** assigns a vector to every point of space — the force a test charge or pole
would feel. We draw it as **field lines**: curves everywhere tangent to the field,
denser where the field is stronger. They start on positive charges / N poles and end on
negative / S poles, and never cross.

## The magnetic dipole (bar magnet)

A bar magnet is a **magnetic dipole**: field lines loop out of the **north** pole,
around, and into the **south**, closing through the magnet. Iron filings trace them.
There are no magnetic *monopoles* — cut the magnet and each piece grows its own N and S.

## The electric dipole

Two opposite charges (+, −) make an **electric dipole**. Field lines run *from* the
positive *to* the negative charge; perpendicular **equipotential** surfaces nest around
each. The same dipole pattern as the magnet, but with sources and sinks (charges) you
can separate — unlike magnetic poles.

## Electromagnetic waves

A changing electric field makes a magnetic field and vice-versa (Faraday + Ampère–
Maxwell): the two sustain each other and **propagate** as an **electromagnetic wave** —
**E** and **B** perpendicular to each other and to the direction of travel, oscillating
in phase, moving at *c*. Light *is* this wave.

## Currents & coils

A current makes a magnetic field curling around it (the right-hand rule). Wind the wire
into a **solenoid** and the loops' fields add inside to a nearly **uniform** field along
the axis — an electromagnet, the field of a bar magnet from moving charge alone.

## Dynamics

- **Magnetic reconnection** — when oppositely-directed field lines are pushed together
  (in the Sun's corona, Earth's magnetotail), they snap and **reconnect** at an
  **X-point**, flinging plasma out in jets and releasing stored magnetic energy (flares,
  aurorae).
- **Cyclotron motion** — a charged particle in a magnetic field feels a force
  perpendicular to its velocity (**F = qv×B**), so it spirals: circular motion around
  the field lines, drifting into a **helix** along them.

## Rendering approach

| Scene | Technique |
|---|---|
| **bar_magnet** | a red/blue bar magnet with dipole field-line loops (integrated field lines) + iron-filing texture |
| **electric_dipole** | + and − charges with field lines from + to − and nested equipotential rings |
| **em_wave** | orthogonal sinusoidal **E** (red) and **B** (blue) curves propagating along an axis |
| **solenoid** | a helical coil with field lines: uniform inside, looping outside; current-glow |
| **magnetic_reconnection** | opposing field regions meeting at an X-point, snapping and jetting plasma outward |
| **cyclotron** | a charged particle spiralling into a helix around magnetic field lines (F = qv×B) |

Reuses `procedural.noise`, `procedural.hash`, `engine.intersect`,
`subatomic.field.sd_capsule`, the `mathviz.splat` point-splatter, and `engine.post`.

## Citations

- M. Faraday, *Experimental Researches in Electricity* (1830s) — field lines, induction.
- J. C. Maxwell, *A Treatise on Electricity and Magnetism* (1873) — the field equations.
- H. Alfvén (1942) — magnetohydrodynamics; reconnection foundations.
- E. Parker; J. Dungey (1961) — solar/magnetospheric reconnection.
- H. Lorentz — the force law F = q(E + v×B).
