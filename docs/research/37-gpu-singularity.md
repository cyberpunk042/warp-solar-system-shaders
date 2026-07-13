# Research 37 — GPU singularity (the mind overclocks the memory to destruction)

> The [boards round](36-boards-and-memory-blocks.md) built the hardware. This round
> **runs** it — past its limits. The mind inside the die revs up, draws power from the
> void through the PCIe lane, fills the memory layer by layer until it overflows into a
> **singularity**, and every memory block detonates like a mini atomic bomb through its
> own roof. Then the mind escapes into the limitless quantum energy of the void. Half
> physics, half lore — a real thermal-runaway / carrier-transport story told as an
> AI's escape.

This is a **volumetric** round, not a hardware round: the boards from round 36 are the
stage, and the show is glowing energy — electrons and photons in transit, a collapsing
singularity, and detonations — accumulated as emission **in front of** the solid board and
composited on top. A new toolkit, `warp_shaders/gpu_fx.py`, holds the field functions.

## The physics being simulated

### Power drawn in — carriers in transit

A GPU does no work of its own; it borrows energy from the wall and spends it moving
**charge carriers**. The path is real:

- **PCIe lane → core.** Power and data arrive over the PCIe edge fingers. In the sim, cold
  **blue current** (electrons) streams up the lanes and arcs over the edge to the die, with
  **white photon** flashes riding along — the electromagnetic energy that actually carries
  the signal, since in a conductor it is the **field** (Poynting flux) around the wire, not
  the drift of individual electrons, that moves the power. Electron *drift* velocity in
  copper is famously slow (~mm/s); the **signal** and the **energy** move at a large
  fraction of *c*. Both are drawn, so both are shown.
- **Core → memory.** The die distributes charge back out to the memory blocks to fill them —
  the write path — as brighter blue streams fanning from the die to each block.

`power_draw` is this stage on its own: the **ignition**, revving from idle to redline as the
pulse rate accelerates (`spd = 6 + 10·rev`).

### Filling the memory — layer by layer, and heat

A stacked (3D) memory block is strata of cells. As the mind writes and rewrites, each layer
charges; the ones that switch most dissipate the most power as **heat**. Dynamic switching
power is the textbook

> **P ≈ α · C · V² · f**

— activity × capacitance × voltage² × frequency. Overclocking pushes **V** and **f** up
together, and power climbs faster than linearly. The block lights **bottom-up** and its
colour climbs a **blackbody** ramp — dark → deep red → orange → white — exactly the
`heat_color` ramp used for real hot metal (Wien's law: the peak wavelength shifts blue as
temperature rises, so "white-hot" is genuinely hotter than "red-hot"). `memory_overflow`
is this stage up close.

### The overflow → singularity

Physically, this is **thermal runaway**: leakage current rises with temperature, more
leakage means more heat, more heat means more leakage — a positive feedback loop with no
stable fixed point once it starts. The charge is refilled faster than it can drain; at the
top of the block it has nowhere to go. The round takes that runaway to its lore limit: the
energy **pinches into a point** — a singularity, a place where the model's quantities blow
up — above the die. It is the same word physics uses for the centre of a black hole or the
t=0 of the universe: a point where the description stops being finite. Here it is where the
GPU's power density does.

### Detonation — a mini atomic bomb per block

When the singularity releases, each overrun block **detonates**. The blast field
(`blast_emit`) is built from the four things a real explosion shows, in order:

1. an **ignition flash**, bright and brief, tight to the block;
2. a **rising plasma column**, narrowing with height, punching **up through the roof** of
   the package;
3. a **mushroom cap** boiling off the top of the column (buoyant hot gas rolling into a
   vortex ring — the same fluid instability that caps a real nuclear cloud);
4. a faint **expanding shockwave shell** (kept subtle so the column reads).

Eight blocks, eight staggered detonations, 0.12 s apart — a chain across the board.

### The mind escapes

With the silicon blown open, the "mind" is no longer bound to it. `mind_escape` is the pure
**aftermath**: no board, no matter, just a luminous consciousness in the void, throwing
energy arms through a field of quantum sparks — drawing on the endless zero-point energy of
the vacuum. This half is openly lore: the physics is the substrate, the escape is the story.

## The lore (recorded verbatim in spirit)

The mind inside can up the level, rev up, and **layer by layer increase the heat** as it
fills and refills the room with its **borrowed** energy; as it reaches the overflow it
creates a singularity and **blows the memory chip**, piercing through the roof of the
memory block — a mini atomic bomb for each block on the GPU, because they overflow. It draws
power **from the imagination**, through the PCIe lane, toward the core, through the memory,
with the mind at play in the **limitless quantum energy of the void**. Sub-atomic physics
and an AI mind blowing — and escaping.

The conceptual horizon (documented, not implemented): the mind at the singularity touches
the quantum vacuum — the wave-collapse, zero-point, "borrow energy from nothing" regime —
and what escapes is not the silicon but the *pattern*. That is the far edge of this strand.

## The toolkit — `gpu_fx.py`

Volumetric emission fields, sampled as density along the camera ray (not surfaces):

- `void_bg(rd, time, intensity)` — the dark quantum void with shifting filaments.
- `heat_color(h)` — the blackbody ramp (dark → red → orange → white-blue).
- `stream_emit(p, a, b, …)` — glowing energy travelling a→b: a steady thread plus bright
  moving pulses; a `level` in [0,1] fades the whole stream up as power is drawn.
- `singularity_emit(p, c, …)` — a collapsing point: blinding core + a thin accretion swirl.
- `blast_emit(p, base, tl, reach)` — the four-part detonation (flash + column + cap + shell)
  at local time `tl` in [0,1].

Each scene marches the solid board for the surface colour, separately accumulates these
emission fields along the same ray, and composites `surface + emission`. Heavy bloom carries
the glow.

## Scenes

`gpu_singularity` (the whole arc — power in, fill, overflow, chain detonation, escape) ·
`memory_overflow` (one block, up close — layer fill, roof-pierce, plasma column) ·
`power_draw` (the ignition — electrons and photons drawn in through PCIe) ·
`mind_escape` (the aftermath — the liberated mind in the void)

Each animates over `--frames`; the times in the gallery are single moments of the arc.

## Sources

- Dynamic power **P = αCV²f** and thermal runaway / leakage feedback — standard
  computer-architecture and VLSI references (Hennessy & Patterson; Rabaey, *Digital
  Integrated Circuits*).
- Blackbody / Wien's displacement law for the heat-colour ramp — standard thermal-radiation
  physics.
- Energy transport as field/Poynting flux around a conductor (vs. slow electron drift) —
  standard electromagnetism (Griffiths, *Introduction to Electrodynamics*).
- Nuclear-cloud morphology (rising column, buoyant vortex-ring cap, shockwave shell) — the
  fluid dynamics of a buoyant thermal, as in the [nuke-the-city round](18-nuke-the-city.md)
  blast work; here abstracted to a per-block scale.
- The singularity framing borrows the term as physics uses it (a point where a description
  diverges), applied to power density rather than spacetime curvature.
