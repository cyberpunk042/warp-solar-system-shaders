# Research 36 — Boards & memory blocks (assembling the machine)

> The payoff of the components round: the actual **boards** a computer is made of,
> assembled from the parts. A RAM stick, an NVMe SSD, a CPU, a graphics card, the
> motherboard that ties them together — and a look **inside** the GPU die at the
> compute fabric. Everything here is a real board you could pull out of a PC.

The [components round](35-electronics-components.md) built the pieces bottom-up. This
round puts them together into the modules and boards those pieces exist to make.

## Memory blocks — tiling a cell into a board

A single memory cell holds one bit. A **chip** tiles billions of cells into a grid with
row/column addressing; a **module** carries several chips on a small PCB with an edge
connector. Two kinds, from the two cells:

- **RAM (DRAM).** The `dram_cell` (1T1C) is tiled into DRAM chips, eight-plus of which sit
  on a **DIMM** stick with a gold edge connector and a key notch. Fast and cheap per bit,
  but volatile and needing constant refresh — so it is the computer's **working memory**.
- **SSD (NAND flash).** The `nand_flash_cell` (floating gate) is tiled — and stacked in 3D
  — into big flash packages. An **M.2 NVMe** drive carries a few of them plus a controller
  and a DRAM cache, and talks to the CPU over PCIe. Non-volatile: it is **storage**.

| | RAM (DIMM) | SSD (NVMe) |
|---|---|---|
| Cell | DRAM 1T1C | NAND flash (floating gate) |
| Volatile | yes (needs refresh) | **no** |
| Role | working memory | permanent storage |
| Connector | DIMM slot | M.2 / PCIe |

## The processor and its cooling

- **CPU.** One big silicon die (millions of CMOS gates) flip-chip mounted on a package
  substrate and capped with a nickel-plated **integrated heat spreader** (IHS). Underneath,
  an LGA pad array mates with the socket. Tiny capacitors ring the substrate to steady the
  supply as billions of transistors switch.
- **Heatsink.** A processor dumps its whole power budget as heat in a tiny die, so it needs
  a **heatsink**: a stack of thin aluminium fins (huge surface area) fed by copper **heat
  pipes** — sealed tubes where fluid boils at the hot base, condenses in the cool fins, and
  wicks back, moving heat far faster than solid metal. A fan blows air through the fins.

## The graphics card

The round's centrepiece. A GPU is a huge die that is nearly all **compute**, so it needs:

- **On-package memory** (`gpu_package`) — the die is ringed by GDDR / HBM memory stacks on
  the same substrate, for the bandwidth to feed thousands of cores.
- **A board** (`graphics_card`) — the GPU package and its memory on a long PCB, with a bank
  of **VRM** power stages, a **PCIe** edge connector, an **8-pin power** connector for the
  extra watts, and a metal **I/O bracket** for the display outputs.
- **A cooler** — a shroud and finned heatsink with two or three **fans** over the whole thing.

### Cooling styles

The same board can wear different coolers, and the choice changes how the card behaves:

| Style | How it moves air | Best for | Scene |
|---|---|---|---|
| **Open-air** | 2–3 **axial** fans push air *down* through the fins; hot air spills into the case | most desktop cards; quietest, coolest | `graphics_card` |
| **Blower** | one **centrifugal** fan *induces* air along a sealed tunnel and exhausts it out the bracket | small cases, dense multi-GPU servers | `gpu_blower` |
| **Passive (fanless)** | no fan at all — a big heatsink sheds heat by convection alone | silent / low-power cards | `gpu_open` |

A **blower** is self-contained (no hot air dumped in the case) but louder; **open-air** is
quieter and cooler but recirculates heat; **passive** is silent but limits how hard the die
can run. `gpu_open` also strips the cooler away entirely so the board itself is visible —
the GPU die, the ring of GDDR, the VRM (chokes + capacitors), the PCIe fingers, and the
copper traces wiring them.

### The board itself (no cosmetics)

`gpu_board` drops the coolers entirely and shows a **workstation-class** populated PCB (in
the spirit of an RTX 6000 Pro Blackwell) — the hardware, not the cover:

- an **enormous exposed die** flip-chip on its substrate, dominating the centre;
- a full **ring of GDDR7** memory packages on three sides for the memory bandwidth;
- a heavy **multi-phase VRM** — bank after bank of **chokes** with their **MOSFET** power
  stages and driver ICs — because the die draws hundreds of amps at ~1 V, converted down
  from the 12 V input;
- dense **capacitor** arrays (tiny **MLCC**, polymer **POSCAP**, and bulk cans) packed
  around the die to hold that low voltage steady through huge, fast current swings;
- a **12VHPWR** power connector, gold **PCIe x16** edge fingers, mount holes, and heavy
  copper **routing** threading it all together.

The difference between a cheap-looking render and a real board is **density**: a serious
GPU is almost entirely memory, power delivery, and decoupling wrapped tightly around one
very large, very hungry die.

## The platform

- **Motherboard.** The big PCB everything plugs into: the **CPU socket** (with its VRM
  heatsinks), the **RAM slots**, the long **PCIe slot** for the graphics card, an **M.2**
  slot for the SSD, and the **chipset** that routes everything else. Copper traces on the
  inner layers wire them together.

## Inside the GPU — the compute fabric

`gpu_floorplan` opens the graphics die: a huge regular **grid of shader cores** (streaming
multiprocessors), each a bundle of the logic cells, all running the same instruction on
different data (**SIMT**). A shared **cache** spine runs down the middle; **memory
controllers** line the edges, talking to the GDDR. This massively-parallel array is what
makes a GPU fast at graphics *and* at the general parallel maths (and neural nets) that run
on it — and it is exactly the fabric a **virtual GPU** — a graphics card simulated inside a
graphics card — would have to model.

## Scenes

`ram_stick` · `nvme_ssd` · `cpu` · `heatsink` · `gpu_package` · `graphics_card` ·
`gpu_blower` · `gpu_open` · `gpu_board` · `motherboard` · `gpu_floorplan`

## Sources

- JEDEC standards for DDR DIMMs and the M.2 / PCIe form factors.
- Standard computer-architecture references (Hennessy & Patterson) for cache hierarchy,
  memory, and the SIMT/GPU execution model.
- Vendor GPU whitepapers (SM/compute-unit grid, HBM on-package memory).
