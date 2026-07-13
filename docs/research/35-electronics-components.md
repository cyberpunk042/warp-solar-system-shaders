# Research 35 — Electronics components (silicon to the memory bit)

> Building a computer from the ground up, respecting the physics. Pure silicon, doped
> junctions, discrete parts, packaged chips, and the single-bit memory cells that a
> RAM, an SSD, or a GPU is made of. Everything here is a real thing you could hold (or
> hold under a microscope), and every scene is grounded in how it actually works.

This is the **components** layer: the pieces a memory board or a processor needs, built
bottom-up. The boards themselves (RAM, NVMe, GPU) come later — first the parts.

## Silicon — the material everything is made of

A processor is, physically, patterned silicon. Silicon (Si, group 14) forms four covalent
bonds in a **diamond-cubic** lattice — each atom at the centre of a tetrahedron of
neighbours (the sp³ bonds). It is a **semiconductor**: a filled valence band and an empty
conduction band separated by a ~1.1 eV gap, so pure silicon barely conducts, but small
additions of impurity or a small voltage change that dramatically.

- **Monocrystalline boule.** Ultra-pure silicon is melted and a seed crystal is dipped in
  and slowly pulled while rotating (the **Czochralski** process, 1918), so the whole melt
  solidifies as one continuous crystal — a mirror-grey cylinder up to 300 mm across.
- **Wafer.** The boule is sawn into thin discs, lapped mirror-flat, and printed with
  hundreds of identical **dies** in a grid; a rim **notch** marks the crystal orientation.
  The micron-thin oxide and photoresist films on top interfere with light — the rainbow
  sheen in every wafer photo.

## Doping and the p-n junction

Replace a few silicon atoms with a group-15 donor (P, As) and you get spare electrons —
**n-type**. Use a group-13 acceptor (B) and you get spare **holes** — **p-type**. Put the
two together and, at the boundary, electrons and holes recombine and leave a carrier-free
**depletion region** with a built-in electric field. That junction:

- passes current one way and blocks the other — a **diode**;
- in a direct-bandgap crystal, releases the recombination energy as photons — a **LED**;
- with a third terminal to gate the channel — a **transistor**, the switch everything is
  built from.

## Discrete components

The passive and simple active parts every board carries:

| Part | What it does | Physics |
|---|---|---|
| **Resistor** | turns volts into a proportional current | Ohm's law `V = IR`; value read off the colour bands |
| **Capacitor** | stores charge on two plates | `Q = CV`; electrolytic (big, polarised) vs ceramic (small, not) |
| **Diode / LED** | one-way valve / light source | a single p-n junction |
| **Inductor** | stores energy in a magnetic field, resists current change | Faraday/Lenz; wire wound on a ferrite core |
| **Crystal oscillator** | a precise clock | **piezoelectric** quartz rings at one mechanical frequency |

The colour code on the resistor here is brown-black-red-gold = 10 × 100 = **1 kΩ**, ±5 %.

## Interconnect and packaging

A die is microscopic and fragile, so it is wired out and sealed, then mounted on a board:

- **PCB** — a fibreglass-epoxy board (FR-4) clad with etched copper: **traces** carry
  signals, **vias** jump between layers, **pads** are where parts solder down, and green
  **solder-mask** coats the bare board.
- **Bond wires** — inside the package, hair-thin **gold** wires arc from pads on the die
  to a metal **leadframe**; a machine welds each in a fraction of a second.
- **Package** — the die + leadframe is sealed in moulded black epoxy. A **DIP** brings the
  connections out as two rows of legs; a **BGA** studs the whole underside with a grid of
  **solder balls** (hundreds of short connections that self-align when they melt at reflow).

## The memory / logic cells (the bit)

At the bottom: the single cell that, tiled into a grid billions wide, becomes a memory
chip — or, wired into gates, a processor.

- **CMOS inverter** — a complementary **PMOS + NMOS** pair sharing a gate (input) and a
  drain node (output): a NOT gate that burns almost no power at rest. Every logic gate is
  built from these pairs.
- **SRAM cell (6T)** — two cross-coupled inverters latch a bit in one of two stable states
  (no refresh), plus two access transistors to the bit lines. Six transistors per bit
  makes it big but **fastest** — processor caches.
- **DRAM cell (1T1C)** — **one transistor, one capacitor**: charge the capacitor for a 1,
  drain it for a 0. Tiny, so main memory is DRAM, but the charge leaks in milliseconds, so
  the chip must **refresh** every cell constantly (the "dynamic").
- **NAND flash cell** — a transistor with an extra, isolated **floating gate**. Electrons
  tunnelled onto it are trapped for years with **no power**, shifting the threshold — the
  **non-volatile** bit inside an SSD / NVMe drive.

| Cell | Transistors | Volatile? | Where |
|---|---|---|---|
| SRAM | 6 | yes (needs power) | CPU / GPU cache |
| DRAM | 1 + 1 cap | yes (needs refresh) | main memory (RAM) |
| NAND flash | 1 (floating gate) | **no** | SSD / NVMe storage |

## Scenes

`silicon_ingot` · `silicon_crystal` · `silicon_wafer` · `pn_junction` · `resistor` ·
`capacitor` · `led` · `inductor` · `crystal_oscillator` · `pcb` · `ic_package` · `bga` ·
`bond_wire` · `dram_cell` · `nand_flash_cell` · `cmos_inverter` · `sram_cell`

## Sources

- J. Czochralski, *Ein neues Verfahren zur Messung der Kristallisationsgeschwindigkeit der
  Metalle* (1918) — the crystal-pulling method.
- W. Shockley, *Electrons and Holes in Semiconductors* (1950) — junctions and transistors.
- Standard semiconductor-device and IC-packaging references (Sze; JEDEC package standards).
