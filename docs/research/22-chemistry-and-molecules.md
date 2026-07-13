# Research 22 — Chemistry & molecules

> The rung **up** from the atoms of Research 21: atoms bonding into molecules,
> molecules stacking into crystals, and molecules reacting. Grounds the chemistry
> strand — geometries, colours, and how we render ball-and-stick molecules.

## Atoms → molecules: the bond

Atoms bond by sharing or transferring electrons:

- **Covalent** — shared electron pairs (H₂O, CH₄, CO₂, benzene). Directional; sets
  molecular *shape*.
- **Ionic** — electron transfer → charged ions held by electrostatics (NaCl).
- **Metallic** — a sea of delocalised electrons (not modelled here).

## Molecular geometry (VSEPR)

Electron pairs repel, so a molecule's shape is set by counting bonding + lone
pairs around the central atom (Valence-Shell Electron-Pair Repulsion):

| Molecule | Central | Shape | Bond angle |
|---|---|---|---|
| **water** H₂O | O (2 bonds, 2 lone) | bent | 104.5° |
| **carbon dioxide** CO₂ | C (2 double bonds) | linear | 180° |
| **methane** CH₄ | C (4 bonds) | tetrahedral | 109.5° |
| **ammonia** NH₃ | N (3 bonds, 1 lone) | trigonal pyramidal | 107° |
| **benzene** C₆H₆ | ring | planar hexagon (aromatic) | 120° |

Bond lengths (Å): O–H 0.96, C=O 1.16, C–H 1.09, N–H 1.01, C–C(aromatic) 1.39.

## Crystals — molecules in a lattice

- **Rock salt** (NaCl) — a face-centred-cubic lattice of alternating Na⁺ and Cl⁻,
  each ion octahedrally surrounded by six of the other.
- **Diamond** — carbon in a tetrahedral covalent network (each C bonded to four),
  the diamond cubic lattice.

## CPK colours (the standard convention)

| Element | Colour |
|---|---|
| Hydrogen H | white |
| Carbon C | dark grey / black |
| Oxygen O | red |
| Nitrogen N | blue |
| Chlorine Cl | green |
| Sodium Na | violet |
| Sulfur S | yellow |

Ball-and-stick radii are shrunk from the true van-der-Waals radii so the **bonds**
(sticks) are visible; space-filling (CPK) uses the full radii.

## Reactions

A chemical reaction rearranges bonds, conserving atoms. **Combustion of methane**:

    CH₄ + 2 O₂ → CO₂ + 2 H₂O   (+ heat + light)

The reactant molecules collide, bonds break, atoms recombine into the products —
exothermic, so the transition releases a burst of light/heat.

## Rendering approach

We render molecules as **ball-and-stick** signed-distance fields, sphere-traced:

| Ingredient | Technique |
|---|---|
| atom | `sd_sphere` at the atom position, CPK-coloured, PBR-shaded (diffuse + Blinn spec + fresnel rim) |
| bond | `sd_capsule` between bonded atoms, a neutral grey stick |
| lattice | domain-repeat the sphere/bond field over the unit cell |
| lighting | a key + fill + rim light, plus SDF **ambient occlusion** (5-tap) and a soft studio-gradient background |
| reaction | interpolate atom positions from reactant → product layout over time, with an energy flash at the transition |

Reuses the engine's `intersect`, `post` (bloom + ACES), and `procedural.noise`; the
new geometry + molecular data live in `warp_shaders/molecules/`.

## Citations

- L. Pauling, *The Nature of the Chemical Bond*, 3rd ed. — covalent/ionic bonding,
  bond lengths.
- R. J. Gillespie & R. S. Nyholm, *Inorganic stereochemistry* (VSEPR), Q. Rev.
  Chem. Soc. 11 (1957) — molecular geometry from electron-pair repulsion.
- CPK colouring: R. B. Corey, L. Pauling (1953); W. Koltun (1965).
- CRC Handbook of Chemistry and Physics — bond lengths, lattice constants.
