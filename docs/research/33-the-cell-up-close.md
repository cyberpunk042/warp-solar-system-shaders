# Research 33 — The cell up close

> One scale down from the body: the machinery of the living cell. Power plants, protein
> factories, the membrane that defines "inside", and the microbes — viruses and
> bacteria — that live at this scale.

## The cell & its membrane

Every cell is wrapped in a **lipid bilayer**: two sheets of phospholipids, their
water-loving **heads** facing out and in, their oily **tails** hidden in the middle —
a self-assembling, fluid barrier two molecules thick studded with **proteins** (channels,
pumps, receptors). It defines the cell's inside and controls what crosses.

## Mitochondrion — the power plant

The **mitochondrion** burns food with oxygen to make **ATP**, the cell's energy
currency. It has a smooth outer membrane and a deeply folded inner membrane — the
**cristae** — whose vast area carries the electron-transport chain. "The powerhouse of
the cell." (It was once a free-living bacterium, engulfed ~1.5 billion years ago —
endosymbiosis — which is why it keeps its own DNA.)

## Ribosome — the protein factory

The **ribosome** reads messenger **RNA** three letters (a **codon**) at a time and links
the matching amino acids into a **protein** chain (translation). Two subunits clamp the
mRNA; transfer RNAs deliver amino acids; the growing chain threads out — the universal
machine that turns genes into proteins.

## Viruses & bacteria

- A **virus** is not quite alive: a shell of protein (**capsid**, often a beautiful
  icosahedron) around a strand of DNA or RNA, sometimes with **spike** proteins to grab
  host cells (the SARS-CoV-2 corona of spikes). It hijacks a cell's machinery to copy
  itself.
- A **bacterium** is a full (if simple) cell — no nucleus, DNA loose in the cytoplasm,
  often swimming with a rotary **flagellum** driven by a molecular motor.

## The immune response

A **phagocyte** (macrophage / neutrophil) of the immune system hunts pathogens and
**engulfs** them — flowing its membrane around a bacterium and swallowing it into a
vesicle to be digested (phagocytosis). The cellular front line.

## Rendering approach

| Scene | Technique |
|---|---|
| **lipid_bilayer** | two rows of phospholipids (round head + twin tails) forming the bilayer, with an embedded channel protein, gently undulating |
| **mitochondrion** | an SDF capsule organelle with the folded inner-membrane cristae, glowing with ATP energy |
| **ribosome** | two subunit blobs clamping an mRNA strand, a growing protein chain threading out, tRNAs arriving |
| **virus** | an icosahedral capsid with radiating spike proteins (a corona), floating |
| **bacterium** | a rod-shaped cell with a coiled genome and a rotating helical flagellum propelling it |
| **immune_cell** | a phagocyte reaching pseudopods around a bacterium and engulfing it |

Reuses `procedural.noise`, `subatomic.field` (sd_capsule, void), `engine.intersect`,
`molecules` helpers, `subatomic.render.orbit_camera`, and `engine.post`.

## Citations

- S. Singer & G. Nicolson, *The Fluid Mosaic Model of the Cell Membrane*, Science (1972).
- L. Margulis, *Origin of Eukaryotic Cells* (1970) — endosymbiosis / mitochondria.
- V. Ramakrishnan, T. Steitz, A. Yonath (Nobel 2009) — ribosome structure.
- D. Caspar & A. Klug (1962) — icosahedral virus capsid geometry.
- É. Metchnikoff (1882) — phagocytosis / cellular immunity.
