# Research 05 — molecular to cell: the bottom of the "show life" ladder

The engine's life strand grows *up* from a grammar (L-Systems → grass → tree).
This note covers the strand growing *down* — the sub-plant scales the operator
named: **DNA → protein → cell**. Sources behind `warp_shaders/life/molecular.py`
and `warp_shaders/life/cell.py`.

## Two rendering styles, on purpose

The atom strand (quark → proton → atom → elements) draws everything as
**glow-impostors** — no geometry, just glow accumulated along each camera ray
(`warp_shaders/particles.py`: `emitter`, `flux`, `ray_sphere`). The plant strand
draws **solid ray-traced triangle meshes** (`life/render.py`). The molecular
scales bridge the two, and the bridge is deliberate: DNA and protein are drawn as
**solid meshes** (tangible molecular machines you could hold), the cell as a
**glow volume** (a soft, translucent, living thing). Going up the ladder the look
shifts glow → solid → soft-glow → solid plants — the visual metaphor for matter
becoming life.

## DNA — the double helix (`build_helix`)

Watson & Crick (1953) fixed the geometry of B-DNA: two antiparallel
sugar-phosphate **backbones** wound as a right-handed double helix, joined by
**base pairs** (A–T, G–C) stacked like rungs. The canonical numbers — **~10.5
base pairs per turn**, **~3.4 Å rise per base pair**, **~20 Å diameter** — are the
ratio `build_helix` reproduces (in arbitrary units): `bp_per_turn=10.5`,
`rise=0.34`, `radius=1.0`.

Each backbone is a helical polyline `(r·cosθ, k·rise, r·sinθ)` tessellated as a
tapered tube (reusing the plant `build_mesh` tube path); the second rail is the
same curve phase-offset by π. Each base pair is a two-colour rung (the base and
its complement) drawn in the four base colours. The scene assembles the helix
**base-pair by base-pair** as `time` advances, then turns it in place.

## Protein — folding (`build_protein`)

A protein is a chain of amino acids that **folds** into a specific 3D shape;
Anfinsen's dogma (1961) is that the fold is determined by the sequence. The two
staple secondary-structure motifs are the **α-helix** (a tight right-handed coil,
~3.6 residues/turn) and the **β-strand** (an extended zig-zag; strands pack into
sheets). `build_protein` tubes the backbone along a path that **interpolates from
an extended chain (`fold=0`) to a compact fold (`fold=1`)** mixing an α-helix
flowing through a turn into an antiparallel β-strand, coloured **N→C** along its
length (the standard rainbow cue in molecular viewers). The scene runs `fold` up
over `time`, so the chain visibly collapses into its fold.

## Cell — and its division (`cell.py`)

The smallest complete life: a **membrane** enclosing **cytoplasm**, a **nucleus**,
and **organelles**. Rendered in the glow style — a two-blob **metaball** field
(`_field`) gives the membrane its soft round silhouette and rim; the cytoplasm is
the field's translucent interior; the nucleus and organelles are `emitter`s (the
organelles packed on a Fibonacci sphere). As `divide` rises 0→1 the two metaball
centres separate, so the membrane **pinches** and the nucleus and organelles
**partition** into two daughters — **mitosis**. The scene animates one rest cell
dividing into two.

## Where this sits

This closes the bottom of the ladder: **DNA → protein → cell** now render beneath
**grass → herb → tree → fern**. Next up the ladder are the deepened plants
(flower, bush, a whole meadow) and then the **mind** — the decision layer that
will steer the tropisms the plants already obey. See
[Research 04 — L-Systems](04-lsystems.md) and
[Research 06 — the mind](06-the-mind.md).
