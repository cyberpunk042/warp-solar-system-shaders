# `warp_shaders.molecules` — ball-and-stick chemistry

Sphere-traced molecules: atoms as CPK-coloured spheres, bonds as grey capsule
sticks, with studio lighting (key + fill + rim + ambient occlusion). Background:
[Research 22 — Chemistry & molecules](../research/22-chemistry-and-molecules.md).

## `molecules.render`

| Symbol | Kind | Purpose |
|---|---|---|
| `render_molecule(w, h, t, mouse, dev, atoms, bonds, …)` | host | ray-march a ball-and-stick molecule. `atoms` = list of `(pos3, radius, colour3)`, `bonds` = list of `(i, j)` |

## `molecules.data`

CPK element colours (`H`, `C`, `O`, `N`, `Cl`, `Na`, `S`) + geometry builders,
each returning `(atoms, bonds)`:

| Builder | Molecule | Shape |
|---|---|---|
| `water()` | H₂O | bent, 104.5° |
| `carbon_dioxide()` | CO₂ | linear |
| `methane()` | CH₄ | tetrahedral |
| `ammonia()` | NH₃ | trigonal pyramidal |
| `benzene()` | C₆H₆ | planar aromatic ring |

Scenes: `water`, `carbon_dioxide`, `methane`, `ammonia`, `benzene` (molecules),
plus `salt_crystal` (NaCl lattice), `combustion` (CH₄+2O₂→CO₂+2H₂O) and
`periodic_table` under `warp_shaders/scenes/`.

The rest of this round's scenes — the origin & large-scale universe (`big_bang`,
`cmb`, `cosmic_web`, `first_stars`, `structure_formation`), the living body
(`neural_net`, `neuron`, `heartbeat`, `dna_transcription`, `red_blood_cells`), and
Earth & weather (`hurricane`, `lightning_storm`, `plate_tectonics`,
`ocean_currents`, `water_cycle`) — are self-contained scene modules reusing the
engine's `procedural.noise`, `intersect`, `post` and the `subatomic` SDF helpers.
